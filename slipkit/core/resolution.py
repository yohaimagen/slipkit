"""
Analytical resolution and covariance for a linear regularized slip inversion.

This is the analytical half of the resolution-analysis toolkit (the empirical
Monte-Carlo / checkerboard half lives elsewhere). Everything here is built from
the **regularized generalized inverse** of the *weighted* linear operator, so
the covariance is already expressed in the unit-variance data metric.

For the augmented, data-weighted, Tikhonov-regularized system

    minimize  || G_w m - d_w ||^2  +  lambda^2 || L m ||^2

with ``G_w = Sigma^-1 G`` and ``L`` the block Laplacian, the generalized inverse is

    G_g = (G_w^T G_w + lambda^2 L^T L)^-1 G_w^T          # (P, N)

and every analytical quantity below is a product of ``G_g`` and ``G_w``:

    R    = G_g G_w      model resolution   (P, P)   m_hat = R m_true
    C_m  = G_g G_g^T    model covariance   (P, P)
    N_d  = G_w G_g      data resolution    (N, N)   the "hat"/importance matrix

The non-negativity/bounds enforced by the production solvers are *nonlinear*, so
these matrices describe the underlying **linear** operator and are exact only on
the free (non-binding) set. They are fast, constraint-free diagnostics; the
Monte-Carlo ensemble (Phase 2) is the authoritative bound-respecting uncertainty.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel, SlipComponent
from slipkit.core.noise import DiagonalNoise, NoiseModel
from slipkit.core.regularization import LaplacianSmoothing, RegularizationManager
from slipkit.core.solvers import SolverStrategy
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.inversion import SlipDistribution


class ResolutionAnalyzer:
    """Analytical resolution/covariance for a linear regularized inversion.

    Build it from an :class:`~slipkit.core.inversion.InversionOrchestrator` with
    :meth:`from_orchestrator` (preferred), or directly from the raw weighted
    operator ``(G_w, L, lambda)``. The generalized inverse ``G_g`` is computed
    once (lazily) and the derived matrices are cached.
    """

    def __init__(
        self,
        g_weighted: np.ndarray,
        l_base: "Union[np.ndarray, object]",
        lambda_spatial: float,
        faults: List[AbstractFaultModel],
        *,
        d_weighted: Optional[np.ndarray] = None,
        max_dense_p: int = 20_000,
    ):
        """Initializes the analyzer from the weighted linear operator.

        Args:
            g_weighted: ``(N, P)`` weighted Green's-function matrix ``G_w``.
            l_base: ``(R, P)`` base regularization matrix ``L`` (the Laplacian at
                ``lambda = 1``); may be sparse or dense.
            lambda_spatial: The spatial regularization weight ``lambda``.
            faults: The fault model(s), for component/patch slicing of the
                per-patch summaries. Column blocks are laid out fault by fault,
                matching :class:`~slipkit.core.inversion.SlipDistribution`.
            d_weighted: Optional ``(N,)`` weighted data vector ``d_w``; only
                needed if the estimate ``m_hat = G_g d_w`` is requested.
            max_dense_p: Guard-rail; materialising a full ``(P, P)`` or ``(N, N)``
                matrix above this size warns before allocating.
        """
        self.g_weighted = np.asarray(g_weighted, dtype=float)
        if self.g_weighted.ndim != 2:
            raise ValueError("g_weighted must be a 2-D (N, P) array.")
        self.l_base = l_base.toarray() if hasattr(l_base, "toarray") else np.asarray(
            l_base, dtype=float
        )
        if self.l_base.shape[1] != self.g_weighted.shape[1]:
            raise ValueError(
                f"G_w has {self.g_weighted.shape[1]} columns but L has "
                f"{self.l_base.shape[1]}; they must match the number of unknowns P."
            )
        if lambda_spatial < 0.0:
            raise ValueError("lambda_spatial must be non-negative.")

        self.lambda_spatial = float(lambda_spatial)
        self.faults = list(faults)
        self.d_weighted = None if d_weighted is None else np.asarray(d_weighted, float)
        self.max_dense_p = int(max_dense_p)

        self.n_data, self.n_params = self.g_weighted.shape

        expected = sum(f.num_components() * f.num_patches() for f in self.faults)
        if expected != self.n_params:
            raise ValueError(
                f"Operator has {self.n_params} columns but the faults imply "
                f"{expected} unknowns (sum of num_components * num_patches)."
            )

        # Per-fault (start, width) blocks in the global P-vector.
        self._fault_offsets: List[Tuple[int, int]] = []
        offset = 0
        for fault in self.faults:
            width = fault.num_components() * fault.num_patches()
            self._fault_offsets.append((offset, width))
            offset += width

        # Caches.
        self._g_generalized: Optional[np.ndarray] = None
        self._resolution: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._svd_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_orchestrator(
        cls,
        inversion: "object",
        lambda_spatial: float,
        *,
        force_recompute: bool = False,
        max_dense_p: int = 20_000,
    ) -> "ResolutionAnalyzer":
        """Builds an analyzer from a configured :class:`InversionOrchestrator`.

        Extracts the weighted operator with the same three-line pattern
        ``run_l_curve`` uses -- assemble at ``lambda = 1`` and read the top ``N``
        rows back out -- so **no assembler change is required**. The kernels are
        computed and cached on the orchestrator's assembler if not already.

        Args:
            inversion: An ``InversionOrchestrator`` with faults, datasets, engine
                and a regularization manager set.
            lambda_spatial: The regularization weight to analyze.
            force_recompute: If True, rebuild the cached elastic kernels first.
            max_dense_p: Guard-rail passed through to the analyzer.

        Returns:
            A :class:`ResolutionAnalyzer` for ``(G_w, L, lambda_spatial)``.
        """
        if not inversion.faults:
            raise ValueError("Inversion has no fault models.")
        if not inversion.datasets:
            raise ValueError("Inversion has no datasets.")
        if inversion.engine is None:
            raise ValueError("Inversion has no GreenFunctionBuilder engine set.")

        assembler = inversion.assembler
        reg_manager = inversion._regularization_manager

        if force_recompute or assembler._G_elastic_cache is None:
            assembler._compute_elastic_kernels(
                inversion.faults, inversion.datasets, inversion.engine
            )

        n_data = sum(len(ds.data) for ds in inversion.datasets)
        # Assemble at lambda = 1: A_ref = [G_w; 1.0 * L], b_ref = [d_w; 0].
        a_ref, b_ref = assembler.assemble(
            inversion.faults,
            inversion.datasets,
            inversion.engine,
            reg_manager,
            1.0,
            force_recompute=False,
        )
        # Keep only the slip columns: nuisance (ramp) unknowns are unregularized
        # and are not part of the slip resolution operator.
        n_slip = sum(
            f.num_components() * f.num_patches() for f in inversion.faults
        )
        g_weighted = np.ascontiguousarray(a_ref[:n_data, :n_slip])
        d_weighted = np.ascontiguousarray(b_ref[:n_data])
        l_base = reg_manager.build_smoothing_matrix(inversion.faults, 1.0)

        return cls(
            g_weighted,
            l_base,
            lambda_spatial,
            inversion.faults,
            d_weighted=d_weighted,
            max_dense_p=max_dense_p,
        )

    # ------------------------------------------------------------------ #
    # Core analytical products
    # ------------------------------------------------------------------ #
    def generalized_inverse(self) -> np.ndarray:
        """Returns the regularized generalized inverse ``G_g`` (shape ``(P, N)``).

        Computed as ``solve(G_w^T G_w + lambda^2 L^T L, G_w^T)`` via a Cholesky
        factorization of the symmetric-positive-(semi)definite normal matrix,
        falling back to a pseudo-inverse if it is singular (e.g. ``lambda = 0``
        with a rank-deficient ``G_w``). Cached after the first call.
        """
        if self._g_generalized is None:
            g = self.g_weighted
            gtg = g.T @ g                                     # (P, P)
            lam2 = self.lambda_spatial ** 2
            normal = gtg + lam2 * (self.l_base.T @ self.l_base)
            gt = g.T                                          # (P, N)
            try:
                c, low = cho_factor(normal, check_finite=False)
                self._g_generalized = cho_solve((c, low), gt, check_finite=False)
            except np.linalg.LinAlgError:
                # Singular normal matrix: minimum-norm solution.
                self._g_generalized = np.linalg.pinv(normal) @ gt
        return self._g_generalized

    def model_resolution(self) -> np.ndarray:
        """Returns the model resolution matrix ``R = G_g G_w`` (shape ``(P, P)``).

        ``m_hat = R m_true``: an identity ``R`` is perfect resolution; smoothing
        pulls it toward a smeared, sub-identity operator. ``R`` is **not**
        symmetric -- rows are averaging kernels, columns are point-spread
        functions (see :meth:`resolution_kernel`).
        """
        if self._resolution is None:
            self._warn_if_large(self.n_params, "model_resolution (P x P)")
            self._resolution = self.generalized_inverse() @ self.g_weighted
        return self._resolution

    def model_covariance(self) -> np.ndarray:
        """Returns the model covariance ``C_m = G_g G_g^T`` (shape ``(P, P)``).

        Because ``G_w`` is already whitened, the data covariance is the identity
        and ``C_m`` is the posterior covariance in slip units squared.
        """
        if self._covariance is None:
            self._warn_if_large(self.n_params, "model_covariance (P x P)")
            g_g = self.generalized_inverse()
            self._covariance = g_g @ g_g.T
        return self._covariance

    def data_resolution(self) -> np.ndarray:
        """Returns the full data resolution / hat matrix ``N_d = G_w G_g``.

        Shape ``(N, N)``. Most users only want its diagonal (the per-point
        importance) -- prefer :meth:`data_resolution_diagonal`, which never forms
        the full matrix.
        """
        self._warn_if_large(self.n_data, "data_resolution (N x N)")
        return self.g_weighted @ self.generalized_inverse()

    def data_resolution_diagonal(self) -> np.ndarray:
        """Returns ``diag(N_d)`` (per-point importance) without forming ``N_d``.

        Uses ``einsum('ij,ji->i', G_w, G_g)``, an ``O(N P)`` contraction instead
        of the ``O(N^2 P)`` full product.
        """
        return np.einsum("ij,ji->i", self.g_weighted, self.generalized_inverse())

    def model_estimate(self) -> np.ndarray:
        """Returns the unconstrained estimate ``m_hat = G_g d_w`` (shape ``(P,)``).

        Requires ``d_weighted`` to have been supplied (it is, via
        :meth:`from_orchestrator`). This is the *linear* estimate; the production
        solvers additionally impose bounds.
        """
        if self.d_weighted is None:
            raise ValueError(
                "model_estimate needs the weighted data vector; construct via "
                "from_orchestrator or pass d_weighted=..."
            )
        return self.generalized_inverse() @ self.d_weighted

    # ------------------------------------------------------------------ #
    # Per-patch / per-component summaries
    # ------------------------------------------------------------------ #
    def resolution_diagonal(
        self, fault_index: int = 0, component: "Optional[Union[SlipComponent, str]]" = None
    ) -> np.ndarray:
        """Returns ``diag(R)`` restricted to one fault (and optionally component).

        Each value is a scalar in ``[0, 1]``: near 1 is well-resolved, near 0 is
        damped by regularization. Mapped patch-by-patch onto the mesh.

        Args:
            fault_index: Which fault's block to slice.
            component: A single component (``'ss'``/``'ds'`` or ``SlipComponent``)
                to restrict to; ``None`` returns the whole fault block (all its
                active components concatenated).

        Returns:
            An ``(M,)`` array for a single component, or ``(num_components * M,)``
            for the whole fault block.
        """
        sl = self._global_slice(fault_index, component)
        return np.diag(self.model_resolution())[sl]

    def resolution_kernel(
        self,
        patch_index: int,
        fault_index: int = 0,
        component: "Optional[Union[SlipComponent, str]]" = None,
        kind: str = "column",
    ) -> np.ndarray:
        """Returns one resolution kernel (row or column of ``R``) as a ``(P,)`` vector.

        ``R`` is not symmetric, so rows and columns answer different questions:

        * ``kind="column"`` -- the **point-spread function**: how a unit spike of
          *true* slip at this patch smears across the whole estimate. This is the
          object the empirical spike test reproduces.
        * ``kind="row"`` -- the **averaging kernel**: how the *estimate* at this
          patch averages true slip over its neighborhood.

        Args:
            patch_index: Local patch index within the selected (fault, component)
                block (``0 <= patch_index < M``).
            fault_index: Which fault the patch belongs to.
            component: Which component the patch belongs to. Required if the fault
                has more than one active component; may be ``None`` for a
                single-component fault.
            kind: ``"column"`` (default) or ``"row"``.

        Returns:
            The full ``(P,)`` kernel across every unknown, so cross-component and
            (future) cross-fault leakage is visible; slice it with
            :meth:`_global_slice` to map onto one component's patches.
        """
        if kind not in ("column", "row"):
            raise ValueError("kind must be 'column' or 'row'.")
        global_idx = self._global_index(patch_index, fault_index, component)
        r = self.model_resolution()
        return r[:, global_idx] if kind == "column" else r[global_idx, :]

    def spread_length(
        self,
        fault_index: int = 0,
        component: "Optional[Union[SlipComponent, str]]" = None,
        mask_below: float = 0.05,
    ) -> np.ndarray:
        """Returns the Backus-Gilbert resolution length per patch, in mesh units.

        For each averaging kernel (row of the fault/component block of ``R``) the
        spread collapses to a single physical length using inter-centroid
        distances::

            spread(i) = sqrt( sum_j R_ij^2 dist(i, j)^2 / sum_j R_ij^2 )

        The ``sqrt`` keeps the units as length (km on a UTM-km mesh), not
        length-squared. This is the headline analytical resolution map.

        Where ``diag(R)`` is tiny the kernel is pure regularization and its spread
        is meaningless, so patches with ``diag(R) < mask_below`` are returned as
        ``NaN`` rather than a misleading number.

        Args:
            fault_index: Which fault to analyze.
            component: Which single component; required for a multi-component
                fault (the averaging kernel is defined within one component's
                block).
            mask_below: ``diag(R)`` threshold below which the spread is masked to
                ``NaN``.

        Returns:
            An ``(M,)`` array of resolution lengths, ``NaN`` where masked.
        """
        comp = self._require_single_component(fault_index, component)
        sl = self._global_slice(fault_index, comp)
        r_block = self.model_resolution()[sl, sl]           # (M, M) averaging kernels
        centroids = self.faults[fault_index].get_centroids()
        dist = cdist(centroids, centroids)                  # (M, M) inter-centroid

        w = r_block ** 2                                     # R_ij^2, rows = estimate i
        num = np.sum(w * dist ** 2, axis=1)
        den = np.sum(w, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.sqrt(num / den)

        diag_r = np.diag(r_block)
        spread[diag_r < mask_below] = np.nan
        return spread

    def cross_component_leakage(self, fault_index: int = 0) -> np.ndarray:
        """Returns the per-patch cross-component leakage (rake trade-off) map.

        For a multi-component fault, ``R`` has off-diagonal blocks coupling the
        components. Row ``i`` of the block ``R[comp_a, comp_b]`` says how true slip
        in component ``b`` leaks into the estimate of component ``a`` at patch
        ``i``. This reports, per component's estimate, the total absolute leakage
        from every *other* component -- a map of where rake is genuinely
        constrained (low leakage) versus assumed (high leakage).

        Args:
            fault_index: Which fault (must invert for >= 2 components).

        Returns:
            A ``(num_components, M)`` array; row ``k`` is the leakage into
            component ``k``'s estimate summed over all other components' blocks.
        """
        fault = self.faults[fault_index]
        comps = fault.active_components()
        if len(comps) < 2:
            raise ValueError(
                "cross_component_leakage needs a fault with >= 2 active "
                f"components; fault {fault_index} has {len(comps)}."
            )
        r = self.model_resolution()
        m = fault.num_patches()
        out = np.zeros((len(comps), m))
        for a, comp_a in enumerate(comps):
            row_sl = self._global_slice(fault_index, comp_a)
            leakage = np.zeros(m)
            for comp_b in comps:
                if comp_b == comp_a:
                    continue
                col_sl = self._global_slice(fault_index, comp_b)
                leakage += np.sum(np.abs(r[row_sl, col_sl]), axis=1)
            out[a] = leakage
        return out

    def model_std(
        self, fault_index: int = 0, component: "Optional[Union[SlipComponent, str]]" = None
    ) -> np.ndarray:
        """Returns per-patch model standard deviation ``sqrt(diag(C_m))``.

        Args:
            fault_index: Which fault's block to slice.
            component: A single component, or ``None`` for the whole fault block.

        Returns:
            An ``(M,)`` (single component) or ``(num_components * M,)`` array of
            standard deviations, in slip units.
        """
        sl = self._global_slice(fault_index, component)
        return np.sqrt(np.diag(self.model_covariance())[sl])

    def model_correlation(self) -> np.ndarray:
        """Returns the ``(P, P)`` model correlation matrix ``C_m / (sigma sigma^T)``.

        Reveals inter-patch trade-offs (e.g. shallow/deep anticorrelation). Rows
        or columns with zero variance yield ``NaN`` correlations.
        """
        c_m = self.model_covariance()
        std = np.sqrt(np.diag(c_m))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = c_m / np.outer(std, std)
        return corr

    def resolved_parameter_count(self) -> float:
        """Returns ``trace(R)`` -- the effective number of resolved parameters.

        A scalar ``<= P`` summarising how many independent quantities the data +
        regularization actually constrain.
        """
        return float(np.trace(self.model_resolution()))

    # ------------------------------------------------------------------ #
    # Optional: TSVD / Picard (regularization-independent geometry view)
    # ------------------------------------------------------------------ #
    def _svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thin SVD of the weighted operator ``G_w = U S V^T`` (cached)."""
        if self._svd_cache is None:
            self._svd_cache = np.linalg.svd(self.g_weighted, full_matrices=False)
        return self._svd_cache

    def singular_values(self) -> np.ndarray:
        """Returns the singular-value spectrum of ``G_w`` (descending).

        A regularization-independent view of what the geometry alone can resolve:
        the decay rate and any spectral gap bound the number of recoverable modes.
        """
        return self._svd()[1]

    def tsvd_model_resolution(self, k: int) -> np.ndarray:
        """Returns the TSVD model resolution ``V_k V_k^T`` for ``k`` retained modes.

        Unlike the Tikhonov ``R``, this depends only on the geometry (``G_w``), not
        on ``lambda``: retaining the top ``k`` right-singular vectors gives the
        resolution of a rank-``k`` truncated inverse. ``diag`` near 1 marks the
        subspace the data constrain directly.

        Args:
            k: Number of singular modes to retain (``1 <= k <= min(N, P)``).

        Returns:
            The ``(P, P)`` matrix ``V[:, :k] @ V[:, :k].T``.
        """
        _, s, vt = self._svd()
        if not (1 <= k <= s.shape[0]):
            raise ValueError(f"k must be in [1, {s.shape[0]}]; got {k}.")
        self._warn_if_large(self.n_params, "tsvd_model_resolution (P x P)")
        v_k = vt[:k].T
        return v_k @ v_k.T

    def picard_coefficients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the discrete Picard-condition arrays for the weighted data.

        For ``G_w = U S V^T`` and weighted data ``d_w``, returns
        ``(s, |u_i^T d_w|, |u_i^T d_w| / s_i)``. The solution is trustworthy only
        while the Fourier coefficients ``|u_i^T d_w|`` decay at least as fast as
        the singular values ``s_i``; where the ratio blows up, that mode is noise.

        Returns:
            A tuple ``(singular_values, fourier_coefficients, ratios)``.

        Raises:
            ValueError: If ``d_weighted`` was not supplied.
        """
        if self.d_weighted is None:
            raise ValueError(
                "picard_coefficients needs the weighted data vector; construct via "
                "from_orchestrator or pass d_weighted=..."
            )
        u, s, _ = self._svd()
        coeffs = np.abs(u.T @ self.d_weighted)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = coeffs / s
        return s, coeffs, ratios

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _global_slice(
        self, fault_index: int, component: "Optional[Union[SlipComponent, str]]"
    ) -> slice:
        """Maps a (fault, component) selection to a slice into the global P-vector."""
        if not (0 <= fault_index < len(self.faults)):
            raise IndexError(f"fault_index {fault_index} out of range.")
        start, width = self._fault_offsets[fault_index]
        if component is None:
            return slice(start, start + width)
        fault = self.faults[fault_index]
        local = fault.component_slice(component)
        if local is None:
            active = [str(c) for c in fault.active_components()]
            raise ValueError(
                f"Component '{component}' is not active on fault {fault_index}. "
                f"Active components: {active}."
            )
        return slice(start + local.start, start + local.stop)

    def _global_index(
        self,
        patch_index: int,
        fault_index: int,
        component: "Optional[Union[SlipComponent, str]]",
    ) -> int:
        """Maps a local patch index to its position in the global P-vector."""
        comp = self._require_single_component(fault_index, component)
        sl = self._global_slice(fault_index, comp)
        m = self.faults[fault_index].num_patches()
        if not (0 <= patch_index < m):
            raise IndexError(
                f"patch_index {patch_index} out of range for a fault with {m} patches."
            )
        return sl.start + patch_index

    def _require_single_component(
        self, fault_index: int, component: "Optional[Union[SlipComponent, str]]"
    ) -> SlipComponent:
        """Resolves ``component``, defaulting a single-component fault to its component."""
        fault = self.faults[fault_index]
        active = fault.active_components()
        if component is None:
            if len(active) == 1:
                return active[0]
            raise ValueError(
                f"Fault {fault_index} has multiple active components "
                f"({[str(c) for c in active]}); specify `component`."
            )
        return SlipComponent.coerce(component)

    def _warn_if_large(self, dim: int, what: str) -> None:
        """Warns before materialising a dense ``dim x dim`` matrix past the guard."""
        if dim > self.max_dense_p:
            warnings.warn(
                f"Materialising {what} with dimension {dim} exceeds max_dense_p="
                f"{self.max_dense_p}; this allocates a dense {dim}x{dim} matrix. "
                "Use the diagonal accessors or raise max_dense_p if intended.",
                UserWarning,
            )


# ===========================================================================
# Empirical resolution (Phase 2): synthetic recovery + Monte-Carlo
#
# Every empirical test runs data through the SAME fault(s), engine, lambda,
# bounds and solver as the real inversion, so it inherits the true geometry and
# constraints. Whitening and the added noise both come from the per-dataset
# NoiseModel (the plan's "noise streams up into the resolution" rule): Sigma uses
# the model's diagonal sigma; perturbations use its (possibly correlated) draws.
# ===========================================================================


def _as_fault_list(faults) -> List[AbstractFaultModel]:
    if isinstance(faults, AbstractFaultModel):
        return [faults]
    return list(faults)


def _default_noise_models(
    datasets: Sequence[GeodeticDataSet],
    noise_models: Optional[Sequence[NoiseModel]],
) -> List[NoiseModel]:
    """Returns one NoiseModel per dataset, defaulting to reported-sigma diagonals.

    If ``noise_models`` is omitted, each dataset gets ``DiagonalNoise.from_dataset``
    (its reported sigma). Correlated InSAR models (``EmpiricalInsarNoise``) are not
    auto-built here -- the datasets carry no type flag and quiet-region fitting can
    fail -- so pass them explicitly when you want correlated draws.
    """
    if noise_models is None:
        return [DiagonalNoise.from_dataset(ds) for ds in datasets]
    models = list(noise_models)
    if len(models) != len(datasets):
        raise ValueError(
            f"noise_models has {len(models)} entries but there are "
            f"{len(datasets)} datasets."
        )
    for nm, ds in zip(models, datasets):
        if len(nm.sigma()) != len(ds):
            raise ValueError(
                f"NoiseModel for '{ds.name}' has length {len(nm.sigma())}, "
                f"but the dataset has {len(ds)} points."
            )
    return models


def _build_dataset_kernels(
    faults: List[AbstractFaultModel],
    datasets: Sequence[GeodeticDataSet],
    engine: GreenFunctionBuilder,
) -> List[np.ndarray]:
    """Builds the per-dataset raw kernel ``(N_i, P)`` (fault blocks concatenated)."""
    blocks = []
    for ds in datasets:
        parts = [engine.build_kernel(f, ds) for f in faults]
        blocks.append(np.concatenate(parts, axis=1))
    return blocks


def _l_base_dense(
    faults: List[AbstractFaultModel],
    manager: Optional[RegularizationManager] = None,
) -> np.ndarray:
    """The base regularization block (at ``lambda = 1``) as a dense array.

    Defaults to a plain Laplacian; pass the inversion's own manager so extra
    blocks (e.g. :class:`DeepEdgeDamping`) are reflected in the diagnostics.
    """
    manager = manager if manager is not None else LaplacianSmoothing()
    l_sparse = manager.build_smoothing_matrix(faults, 1.0)
    return l_sparse.toarray() if hasattr(l_sparse, "toarray") else np.asarray(l_sparse)


def _weighted_operator(
    raw_blocks: Sequence[np.ndarray], sigmas: Sequence[np.ndarray]
) -> np.ndarray:
    """Stacks ``Sigma^-1 G`` over datasets into the weighted operator ``G_w``."""
    return np.vstack([g / s[:, None] for g, s in zip(raw_blocks, sigmas)])


def _solve_weighted(
    g_w: np.ndarray,
    d_w: np.ndarray,
    l_base: np.ndarray,
    lam: float,
    solver: SolverStrategy,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Solves the augmented ``[G_w; lambda L] m = [d_w; 0]`` with the real solver."""
    a_aug = np.vstack([g_w, lam * l_base])
    b_aug = np.concatenate([d_w, np.zeros(l_base.shape[0])])
    return solver.solve(a_aug, b_aug, bounds)


def _fault_local_coords(centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Projects patch centroids onto the fault plane's two principal axes.

    Returns ``(xi, zeta)`` -- along-strike and down-dip coordinates -- via PCA of
    the centroids (exact for a planar mesh, a good approximation otherwise). Both
    are shifted to start at 0 so integer block indices begin at the mesh corner.
    """
    c = np.asarray(centroids, dtype=float)
    centered = c - c.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xi = centered @ vt[0]
    zeta = centered @ vt[1]
    return xi - xi.min(), zeta - zeta.min()


class RecoveryResult:
    """Outcome of one synthetic recovery test (checkerboard / target / spike)."""

    def __init__(
        self,
        input_slip: np.ndarray,
        recovered: SlipDistribution,
        synthetic_datasets: List[GeodeticDataSet],
        difference: np.ndarray,
        recovery_ratio: np.ndarray,
        metrics: Dict[str, float],
        faults: List[AbstractFaultModel],
    ):
        self.input_slip = np.asarray(input_slip)
        self.recovered = recovered
        self.synthetic_datasets = synthetic_datasets
        self.difference = np.asarray(difference)      # recovered - input, (P,)
        self.recovery_ratio = np.asarray(recovery_ratio)  # recovered/input, NaN off-pattern
        self.metrics = metrics
        self.faults = faults

    def input_distribution(self) -> SlipDistribution:
        """Returns the input (target) slip as a :class:`SlipDistribution`."""
        return SlipDistribution(self.input_slip, self.faults)

    def difference_distribution(self) -> SlipDistribution:
        """Returns the (recovered - input) difference as a :class:`SlipDistribution`."""
        return SlipDistribution(self.difference, self.faults)


class SyntheticRecoveryTest:
    """Checkerboard / restoration / spike tests through the real solver.

    Forward-models a known target to the *actual* observation coordinates, adds a
    realization from each dataset's :class:`NoiseModel`, and re-inverts with the
    production ``lambda`` / bounds / solver, so the recovered pattern degrades
    exactly where each dataset's SNR is genuinely low.

    The raw kernels are built once at construction and reused for every target, so
    a size/location sweep costs one solve per target (no kernel rebuilds).
    """

    def __init__(
        self,
        faults: Union[AbstractFaultModel, Sequence[AbstractFaultModel]],
        datasets: Sequence[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        solver: SolverStrategy,
        *,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        noise_models: Optional[Sequence[NoiseModel]] = None,
        regularization: Optional[RegularizationManager] = None,
    ):
        """Initializes the test harness.

        Args:
            faults: The fault model(s) used in the real inversion.
            datasets: The observed datasets (their coords/unit-vecs define the
                synthetic observation geometry).
            engine: The Green's-function engine.
            solver: The production solver strategy.
            bounds: Optional ``(lower, upper)`` bounds, as in the real inversion.
                Used to validate checkerboard feasibility (§2.1).
            noise_models: One :class:`NoiseModel` per dataset; defaults to
                ``DiagonalNoise.from_dataset`` (see :func:`_default_noise_models`).
            regularization: The inversion's regularization manager; defaults to
                plain :class:`LaplacianSmoothing`. Pass the orchestrator's own
                manager when it is not the default (e.g. wrapped in
                :class:`DeepEdgeDamping`) so the recovery test re-inverts with
                the same regularization as production.
        """
        self.faults = _as_fault_list(faults)
        self.datasets = list(datasets)
        self.engine = engine
        self.solver = solver
        self.bounds = bounds
        self.noise_models = _default_noise_models(self.datasets, noise_models)

        self._raw = _build_dataset_kernels(self.faults, self.datasets, engine)
        self._sigmas = [nm.sigma() for nm in self.noise_models]
        self._l_base = _l_base_dense(self.faults, regularization)
        self._n_params = sum(
            f.num_components() * f.num_patches() for f in self.faults
        )
        # Per-fault (start, width) offsets in the global slip vector.
        self._offsets: List[Tuple[int, int]] = []
        off = 0
        for f in self.faults:
            w = f.num_components() * f.num_patches()
            self._offsets.append((off, w))
            off += w

    # -- pattern builders -------------------------------------------------- #
    def make_checkerboard(
        self,
        block_size: float,
        *,
        component: "Optional[Union[SlipComponent, str]]" = None,
        amplitude: float = 1.0,
        baseline: float = 0.0,
    ) -> np.ndarray:
        """Builds a fault-local checkerboard target (§2.1).

        Each patch is assigned the parity of ``floor(xi/block) + floor(zeta/block)``
        in fault-local (along-strike ``xi``, down-dip ``zeta``) coordinates, so
        ``block_size`` is a real length on an irregular mesh -- not a patch count.
        Patch values are ``{baseline, baseline + amplitude}``.

        The default ``baseline=0`` / ``amplitude=1`` gives a ``0 / A`` alternation,
        which is feasible under non-negativity. A ``±A`` pattern
        (``baseline=-A, amplitude=2A``) is unrecoverable under ``slip >= 0`` and is
        rejected when ``bounds`` forbid it.

        Args:
            block_size: Checkerboard block size, in the mesh's coordinate units.
            component: Component(s) to imprint the pattern on; ``None`` (default)
                uses every active component of each fault.
            amplitude: Peak-minus-baseline slip of the pattern.
            baseline: Baseline slip added to every patch.

        Returns:
            A ``(P,)`` target slip vector.

        Raises:
            ValueError: If the resulting pattern violates ``self.bounds``.
        """
        target = np.zeros(self._n_params)
        for fault, (start, _) in zip(self.faults, self._offsets):
            xi, zeta = _fault_local_coords(fault.get_centroids())
            parity = (np.floor(xi / block_size) + np.floor(zeta / block_size)) % 2
            pattern = baseline + amplitude * parity
            comps = (
                fault.active_components()
                if component is None
                else [SlipComponent.coerce(component)]
            )
            for comp in comps:
                local = fault.component_slice(comp)
                if local is None:
                    active = [str(c) for c in fault.active_components()]
                    raise ValueError(
                        f"Component '{comp}' is not active on this fault. "
                        f"Active components: {active}."
                    )
                target[start + local.start:start + local.stop] = pattern
        self._validate_bounds(target)
        return target

    def make_point_source(
        self,
        patch_index: int,
        fault_index: int = 0,
        *,
        component: "Optional[Union[SlipComponent, str]]" = None,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Builds a single-patch spike target for the empirical PSF test (§2.3).

        Unit slip on one patch/component; forward-modelling and re-inverting this
        yields the bound-respecting sibling of a *column* of the resolution matrix
        ``R`` (:meth:`ResolutionAnalyzer.resolution_kernel` with ``kind="column"``).

        Args:
            patch_index: Local patch index within the (fault, component) block.
            fault_index: Which fault the patch belongs to.
            component: Which component; required for a multi-component fault.
            amplitude: Spike amplitude (default unit).

        Returns:
            A ``(P,)`` target slip vector, zero except at the chosen patch.
        """
        fault = self.faults[fault_index]
        active = fault.active_components()
        if component is None:
            if len(active) != 1:
                raise ValueError(
                    f"Fault {fault_index} has multiple components; specify one."
                )
            comp = active[0]
        else:
            comp = SlipComponent.coerce(component)
        local = fault.component_slice(comp)
        if local is None:
            raise ValueError(f"Component '{comp}' is not active on fault {fault_index}.")
        m = fault.num_patches()
        if not (0 <= patch_index < m):
            raise IndexError(f"patch_index {patch_index} out of range (0..{m - 1}).")
        target = np.zeros(self._n_params)
        start = self._offsets[fault_index][0]
        target[start + local.start + patch_index] = amplitude
        self._validate_bounds(target)
        return target

    # -- the test ---------------------------------------------------------- #
    def run(
        self,
        target_slip: np.ndarray,
        lambda_spatial: float,
        *,
        noise: bool = True,
        seed: Optional[int] = None,
    ) -> RecoveryResult:
        """Forward-models ``target_slip``, (optionally) adds noise, and re-inverts.

        Args:
            target_slip: The ``(P,)`` input slip to recover.
            lambda_spatial: The production regularization weight.
            noise: If True, add one realization drawn from each dataset's
                :class:`NoiseModel`; if False, run the clean (noise-free) test.
            seed: Seed for the noise realization (reproducibility).

        Returns:
            A :class:`RecoveryResult` with the recovered distribution, synthetic
            datasets, per-patch difference and recovery ratio, and summary metrics.
        """
        target = np.asarray(target_slip, dtype=float).ravel()
        if target.shape[0] != self._n_params:
            raise ValueError(
                f"target_slip length {target.shape[0]} != number of unknowns "
                f"{self._n_params}."
            )
        rng = np.random.default_rng(seed)

        synthetic = []
        weighted_data = []
        for ds, g_raw, sigma, nm in zip(
            self.datasets, self._raw, self._sigmas, self.noise_models
        ):
            predicted = g_raw @ target
            if noise:
                d_i = predicted + nm.sample(rng)
                nm.free_cache()   # release any O(N^2) correlated factor at once
            else:
                d_i = predicted
            synthetic.append(
                GeodeticDataSet(
                    ds.coords, d_i, ds.unit_vecs, sigma, name=f"{ds.name}_synthetic"
                )
            )
            weighted_data.append(d_i / sigma)

        g_w = _weighted_operator(self._raw, self._sigmas)
        d_w = np.concatenate(weighted_data)
        m = _solve_weighted(
            g_w, d_w, self._l_base, lambda_spatial, self.solver, self.bounds
        )
        recovered = SlipDistribution(m, self.faults)

        difference = m - target
        on_pattern = np.abs(target) > 1e-12
        recovery_ratio = np.full(self._n_params, np.nan)
        recovery_ratio[on_pattern] = m[on_pattern] / target[on_pattern]

        # Summary metrics over the imprinted patches (correlation over all).
        if np.ptp(target) > 0 and np.ptp(m) > 0:
            correlation = float(np.corrcoef(target, m)[0, 1])
        else:
            correlation = float("nan")
        metrics = {
            "rms_difference": float(np.sqrt(np.mean(difference ** 2))),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "correlation": correlation,
            "median_recovery_ratio": float(np.nanmedian(recovery_ratio))
            if np.any(on_pattern)
            else float("nan"),
        }
        return RecoveryResult(
            target, recovered, synthetic, difference, recovery_ratio, metrics, self.faults
        )

    # -- helpers ----------------------------------------------------------- #
    def _validate_bounds(self, target: np.ndarray) -> None:
        """Raises if ``target`` violates ``self.bounds`` (feasibility, §2.1)."""
        if self.bounds is None:
            return
        lower, upper = self.bounds
        lower = np.broadcast_to(np.asarray(lower, dtype=float), target.shape)
        upper = np.broadcast_to(np.asarray(upper, dtype=float), target.shape)
        tol = 1e-9
        if np.any(target < lower - tol) or np.any(target > upper + tol):
            raise ValueError(
                "Synthetic pattern violates the solver bounds and is unrecoverable "
                "by construction (e.g. a +/-A checkerboard under slip >= 0). Use a "
                "0/A alternation (baseline >= amplitude for +/-, or the defaults)."
            )


# ---------------------------------------------------------------------------
# Monte-Carlo parallel workers (module scope so they pickle under "spawn").
#
# The perturbations are pre-drawn in the parent (see perturb_data) so the
# workers only *solve* a supplied weighted-data vector. This is deliberate: if
# each worker drew its own correlated noise it would build a dense (N_i x N_i)
# covariance + Cholesky for every dataset, i.e. n_workers x sum_i N_i**2 memory
# -- tens of GB for thousands of points per track, which OOMs the machine. The
# workers here hold only G_w (and L), never a covariance.
# ---------------------------------------------------------------------------

_MC_G_W: Optional[np.ndarray] = None
_MC_L_BASE: Optional[np.ndarray] = None
_MC_LAM: Optional[float] = None
_MC_SOLVER: Optional[SolverStrategy] = None
_MC_BOUNDS: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _mc_worker_init(g_w, l_base, lam, solver, bounds):
    """Initializer: cache the shared (read-only) solve operands in the worker."""
    global _MC_G_W, _MC_L_BASE, _MC_LAM, _MC_SOLVER, _MC_BOUNDS
    _MC_G_W = g_w
    _MC_L_BASE = l_base
    _MC_LAM = lam
    _MC_SOLVER = solver
    _MC_BOUNDS = bounds


def _mc_solve_dw(d_w: np.ndarray) -> np.ndarray:
    """Re-inverts one pre-perturbed weighted-data vector ``d_w``."""
    return _solve_weighted(_MC_G_W, d_w, _MC_L_BASE, _MC_LAM, _MC_SOLVER, _MC_BOUNDS)


class EnsembleResult:
    """A slip ensemble (Monte-Carlo, jackknife or bootstrap) with summary stats."""

    def __init__(
        self,
        samples: np.ndarray,
        faults: List[AbstractFaultModel],
        reference: Optional[np.ndarray] = None,
    ):
        """Initializes the ensemble.

        Args:
            samples: ``(K, P)`` array of solved slip vectors.
            faults: The fault model(s), for mapping vectors onto the mesh.
            reference: Optional ``(P,)`` reference estimate (the production
                ``m_hat``); enables :attr:`constraint_bias`.
        """
        self.samples = np.asarray(samples, dtype=float)
        if self.samples.ndim != 2:
            raise ValueError("samples must be a 2-D (K, P) array.")
        self.faults = faults
        self.reference = None if reference is None else np.asarray(reference, float)

        self.mean = self.samples.mean(axis=0)
        self.std = (
            self.samples.std(axis=0, ddof=1)
            if self.samples.shape[0] > 1
            else np.zeros(self.samples.shape[1])
        )
        # constraint_bias = <m> - m_hat: ~0 for an unconstrained linear solver,
        # so any structure maps the constraint-induced (one-sided) bias (§2.4).
        self.constraint_bias = (
            None if self.reference is None else self.mean - self.reference
        )

    @property
    def n_samples(self) -> int:
        """Number of ensemble members ``K``."""
        return self.samples.shape[0]

    def percentile(self, q: "Union[float, Sequence[float]]") -> np.ndarray:
        """Returns per-patch percentile(s) across the ensemble (``axis=0``)."""
        return np.percentile(self.samples, q, axis=0)

    def mean_distribution(self) -> SlipDistribution:
        """Maps the ensemble mean onto the fault mesh(es)."""
        return SlipDistribution(self.mean, self.faults)

    def std_distribution(self) -> SlipDistribution:
        """Maps the per-patch std (the headline uncertainty) onto the mesh(es)."""
        return SlipDistribution(self.std, self.faults)

    def running_std(self, patch_index: Optional[int] = None) -> np.ndarray:
        """Returns the running std as ``K`` grows -- a convergence diagnostic.

        The std stabilizes as ``1/sqrt(K)``. With ``patch_index`` given, tracks
        that patch; otherwise returns the mean over patches at each ``k``.

        Args:
            patch_index: Global unknown index to track, or ``None`` for the
                across-patch mean.

        Returns:
            A ``(K,)`` array; entry ``k`` is the std computed from the first
            ``k + 1`` samples (0 for ``k = 0``).
        """
        k_total = self.samples.shape[0]
        out = np.zeros(k_total)
        for k in range(1, k_total):
            s = self.samples[: k + 1].std(axis=0, ddof=1)
            out[k] = s[patch_index] if patch_index is not None else float(s.mean())
        return out


class MonteCarloResolution:
    """Data-perturbation ensembles (§2.4) and dataset jackknife/bootstrap (§2.5).

    The authoritative, bound-respecting uncertainty: it runs the real solver with
    the real bounds on perturbed data, so it captures constraint-induced,
    non-Gaussian behaviour that the analytical ``C_m`` cannot. Perturbations come
    from each dataset's :class:`NoiseModel` (correlated InSAR draws included);
    the whitening ``Sigma`` uses the same model's diagonal sigma.
    """

    def __init__(
        self,
        inversion: "object",
        lambda_spatial: float,
        *,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        n_jobs: Optional[int] = -1,
        noise_models: Optional[Sequence[NoiseModel]] = None,
    ):
        """Initializes from a configured :class:`InversionOrchestrator`.

        Args:
            inversion: The orchestrator (faults, datasets, engine, solver set).
            lambda_spatial: The production regularization weight.
            bounds: Optional solver bounds (as in the real inversion).
            n_jobs: Worker processes for :meth:`perturb_data`; ``-1`` uses all
                cores (capped at ``K``), ``1`` runs serially.
            noise_models: One per dataset; defaults to ``DiagonalNoise.from_dataset``.
        """
        if not inversion.faults:
            raise ValueError("Inversion has no fault models.")
        if not inversion.datasets:
            raise ValueError("Inversion has no datasets.")
        if inversion.engine is None:
            raise ValueError("Inversion has no engine set.")
        if inversion.solver is None:
            raise ValueError("Inversion has no solver set.")

        self.faults = list(inversion.faults)
        self.datasets = list(inversion.datasets)
        self.solver = inversion.solver
        self.lambda_spatial = float(lambda_spatial)
        self.bounds = bounds
        self.n_jobs = n_jobs
        self.noise_models = _default_noise_models(self.datasets, noise_models)

        # Reuse the assembler's cached raw kernels (compute once if needed).
        assembler = inversion.assembler
        if assembler._G_elastic_cache is None:
            assembler._compute_elastic_kernels(
                self.faults, self.datasets, inversion.engine
            )
        self._raw = list(assembler._G_elastic_cache)
        self._sigmas = [nm.sigma() for nm in self.noise_models]
        self._l_base = _l_base_dense(
            self.faults, getattr(inversion, "_regularization_manager", None)
        )

        # Per-dataset row slices into the stacked weighted operator.
        self._dataset_info: List[Tuple[int, int, np.ndarray, NoiseModel]] = []
        start = 0
        weighted_data = []
        for ds, sigma, nm in zip(self.datasets, self._sigmas, self.noise_models):
            n_i = len(ds)
            self._dataset_info.append((start, start + n_i, sigma, nm))
            weighted_data.append(np.asarray(ds.data, dtype=float) / sigma)
            start += n_i

        self._g_w = _weighted_operator(self._raw, self._sigmas)
        self._d_w = np.concatenate(weighted_data)
        # The production estimate on the unperturbed data (for constraint bias).
        self._m_hat = _solve_weighted(
            self._g_w, self._d_w, self._l_base, self.lambda_spatial,
            self.solver, self.bounds,
        )

    @property
    def m_hat(self) -> np.ndarray:
        """The production estimate on the observed (unperturbed) data."""
        return self._m_hat

    def perturb_data(
        self, n_realizations: int, *, seed: Optional[int] = None
    ) -> EnsembleResult:
        """Draws ``K`` noise realizations, re-inverts each, and summarises them.

        The perturbations are **drawn in the parent, one dataset at a time**
        (:meth:`NoiseModel.sample_batch`), and each dataset's dense covariance is
        freed before the next. Peak memory is therefore a single track's
        ``O(N_i**2)`` factor -- not ``n_workers x sum_i N_i**2``, which is what a
        per-worker draw would cost and what OOMs the machine on real data. The
        workers receive only the pre-perturbed weighted-data vectors and solve.

        Args:
            n_realizations: Number of perturbed re-inversions ``K``.
            seed: Master seed for the ``SeedSequence`` (reproducibility).

        Returns:
            An :class:`EnsembleResult` (samples, mean, std, percentiles,
            constraint bias vs :attr:`m_hat`).
        """
        if n_realizations < 1:
            raise ValueError("n_realizations must be >= 1.")
        k = int(n_realizations)

        # Pre-draw all whitened perturbations in the parent, per dataset, freeing
        # each track's dense covariance immediately after (bounded peak memory).
        # Independent per-track seeds keep the draw reproducible for a given seed.
        track_seeds = np.random.SeedSequence(seed).spawn(len(self._dataset_info))
        eps_w = np.zeros((k, self._d_w.shape[0]))
        for (start, stop, sigma, nm), tseed in zip(self._dataset_info, track_seeds):
            n_i = stop - start
            if (getattr(nm, "correlated", False) and getattr(nm, "sill", 0.0) > 0.0
                    and n_i > 8000):
                warnings.warn(
                    f"Correlated noise on a dataset with {n_i} points builds a "
                    f"dense {n_i}x{n_i} covariance (~{n_i * n_i * 8 / 1e9:.1f} GB). "
                    "Consider a smaller dataset or correlated=False.",
                    UserWarning,
                )
            rng = np.random.default_rng(tseed)
            block = nm.sample_batch(rng, k)            # (N_i, k)
            eps_w[:, start:stop] = (block / sigma[:, None]).T
            nm.free_cache()                            # release O(N_i^2) now
        d_w_batch = self._d_w[None, :] + eps_w         # (k, N)

        n_jobs = self.n_jobs
        if n_jobs is None or n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_jobs = max(1, min(n_jobs, k))

        init_args = (
            self._g_w, self._l_base, self.lambda_spatial, self.solver, self.bounds,
        )
        if n_jobs == 1:
            _mc_worker_init(*init_args)
            samples = [_mc_solve_dw(row) for row in d_w_batch]
        else:
            # Pin BLAS to one thread per worker so N processes don't
            # oversubscribe cores (set before the children import numpy).
            thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            saved = {v: os.environ.get(v) for v in thread_vars}
            for v in thread_vars:
                os.environ[v] = "1"
            try:
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_mc_worker_init,
                    initargs=init_args,
                ) as executor:
                    samples = list(executor.map(_mc_solve_dw, list(d_w_batch)))
            finally:
                for v, val in saved.items():
                    if val is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = val

        return EnsembleResult(np.array(samples), self.faults, reference=self._m_hat)

    def jackknife_datasets(self) -> EnsembleResult:
        """Drops each dataset in turn and re-inverts (§2.5).

        Measures the solution's sensitivity to any single track. Returns an
        ensemble of ``num_datasets`` leave-one-out solutions.
        """
        if len(self.datasets) < 2:
            raise ValueError("Need >= 2 datasets to jackknife.")
        samples = []
        for i in range(len(self.datasets)):
            keep = [j for j in range(len(self.datasets)) if j != i]
            samples.append(self._solve_subset(keep))
        return EnsembleResult(np.array(samples), self.faults, reference=self._m_hat)

    def bootstrap_datasets(
        self, n_realizations: int, *, seed: Optional[int] = None
    ) -> EnsembleResult:
        """Resamples whole datasets with replacement and re-inverts (§2.5).

        Args:
            n_realizations: Number of bootstrap resamples ``K``.
            seed: Seed for the resampling (reproducibility).

        Returns:
            An :class:`EnsembleResult` over the ``K`` resampled solutions.
        """
        if len(self.datasets) < 2:
            raise ValueError("Need >= 2 datasets to bootstrap.")
        if n_realizations < 1:
            raise ValueError("n_realizations must be >= 1.")
        rng = np.random.default_rng(seed)
        d = len(self.datasets)
        samples = [
            self._solve_subset(list(rng.integers(0, d, size=d)))
            for _ in range(n_realizations)
        ]
        return EnsembleResult(np.array(samples), self.faults, reference=self._m_hat)

    def _solve_subset(self, dataset_indices: Sequence[int]) -> np.ndarray:
        """Solves using only the given datasets (with multiplicity), unperturbed."""
        g_w = np.vstack([self._raw[i] / self._sigmas[i][:, None] for i in dataset_indices])
        d_w = np.concatenate(
            [np.asarray(self.datasets[i].data, float) / self._sigmas[i]
             for i in dataset_indices]
        )
        return _solve_weighted(
            g_w, d_w, self._l_base, self.lambda_spatial, self.solver, self.bounds
        )
