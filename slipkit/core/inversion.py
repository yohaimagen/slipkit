"""
This module contains the InversionOrchestrator, which is the main
user-facing API for setting up and running a slip inversion.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack, vstack as sparse_vstack

from slipkit.core.data import (
    GeodeticDataSet,
    Ramp,
    nuisance_bases,
    nuisance_widths,
)
from slipkit.core.fault import AbstractFaultModel, SlipComponent
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.solvers import NnlsSolver, SolverStrategy
from slipkit.core.regularization import RegularizationManager, LaplacianSmoothing


class SlipDistribution:
    """
    Stores the results of a slip inversion, mapped back to fault components.

    The raw ``slip_vector`` is the concatenation of each fault's block, and each
    fault's block is the concatenation of its active components (in canonical
    order, strike-slip before dip-slip). This class exposes accessors that slice
    the vector back into per-fault, per-component slip using the fault metadata.
    """

    def __init__(
        self,
        slip_vector: np.ndarray,
        faults: List[AbstractFaultModel],
        nuisance: Optional[Dict[str, Ramp]] = None,
    ):
        self.slip_vector = np.asarray(slip_vector)
        self.faults = faults
        # Fitted nuisance ramps, keyed by dataset name. Empty unless a dataset
        # asked for one; the slip vector never includes their coefficients.
        self.nuisance: Dict[str, Ramp] = dict(nuisance or {})

        # Precompute (start, width) of each fault's block in the global vector.
        self._fault_offsets: List[Tuple[int, int]] = []
        offset = 0
        for fault in faults:
            width = fault.num_components() * fault.num_patches()
            self._fault_offsets.append((offset, width))
            offset += width
        self._total_width = offset

        if self.slip_vector.shape[0] != self._total_width:
            raise ValueError(
                f"slip_vector length ({self.slip_vector.shape[0]}) does not match "
                f"the total number of unknowns across faults ({self._total_width})."
            )

    def nuisance_prediction(self, dataset: GeodeticDataSet) -> np.ndarray:
        """
        Returns the fitted nuisance (ramp) displacement for one dataset.

        Args:
            dataset: The dataset to evaluate the ramp for; matched by name.

        Returns:
            An ``(N,)`` array in the data's units, all zeros if no ramp was
            estimated for this dataset.
        """
        ramp = self.nuisance.get(dataset.name)
        if ramp is None:
            return np.zeros(len(dataset))
        return ramp.evaluate(dataset.coords)

    def get_fault_vector(self, fault_index: int = 0) -> np.ndarray:
        """Returns the slip sub-vector for a single fault (all its components)."""
        start, width = self._fault_offsets[fault_index]
        return self.slip_vector[start:start + width]

    def get_component(
        self,
        component: "Union[SlipComponent, str]",
        fault_index: int = 0,
    ) -> np.ndarray:
        """
        Returns the (M,) slip for a single component of a single fault.

        Args:
            component: A SlipComponent or string alias ('ss'/'ds', etc.).
            fault_index: Index of the fault in the inversion.

        Raises:
            ValueError: If the component is not active on that fault.
        """
        component = SlipComponent.coerce(component)
        fault = self.faults[fault_index]
        local = fault.component_slice(component)
        if local is None:
            active = [str(c) for c in fault.active_components()]
            raise ValueError(
                f"Component '{component}' is not active on fault {fault_index}. "
                f"Active components: {active}."
            )
        fstart, _ = self._fault_offsets[fault_index]
        return self.slip_vector[fstart + local.start:fstart + local.stop]

    def total_slip(self, fault_index: int = 0) -> np.ndarray:
        """
        Returns the (M,) total slip magnitude per patch for a single fault.

        The magnitude combines all active slip components in quadrature, e.g.
        ``sqrt(strike_slip**2 + dip_slip**2)`` for a fault inverting for both.
        For a single-component fault it reduces to the absolute slip.

        Args:
            fault_index: Index of the fault in the inversion.

        Returns:
            A ``(M,)`` array of non-negative slip magnitudes, in the same length
            units as the inverted slip (typically metres).
        """
        fault = self.faults[fault_index]
        squared = None
        for component in fault.active_components():
            comp = self.get_component(component, fault_index)
            squared = comp ** 2 if squared is None else squared + comp ** 2
        return np.sqrt(squared)

    def seismic_moment(
        self,
        shear_modulus: float = 3.3e10,
        length_unit: str = "km",
    ) -> float:
        """
        Computes the total scalar seismic moment of the slip distribution.

        The moment is summed over every patch of every fault using
        :math:`M_0 = \\mu \\sum_i A_i s_i`, where :math:`A_i` is the patch area,
        :math:`s_i` its total slip magnitude, and :math:`\\mu` the shear modulus.

        Args:
            shear_modulus: Shear (rigidity) modulus :math:`\\mu` in pascals.
                Defaults to ``3.3e10`` Pa, a common crustal value.
            length_unit: The length unit of the fault-mesh coordinates, used to
                convert patch areas to square metres. ``"km"`` (default, matching
                local UTM-km meshes) or ``"m"``.

        Returns:
            The scalar seismic moment :math:`M_0` in newton-metres (N·m).
        """
        area_to_m2 = {"km": 1.0e6, "m": 1.0}
        if length_unit not in area_to_m2:
            raise ValueError(
                f"Unknown length_unit '{length_unit}'. Use 'km' or 'm'."
            )
        scale = area_to_m2[length_unit]

        moment = 0.0
        for i, fault in enumerate(self.faults):
            get_areas = getattr(fault, "get_areas", None)
            if get_areas is None:
                raise NotImplementedError(
                    f"Fault {i} ({type(fault).__name__}) does not expose "
                    "get_areas(); cannot compute seismic moment."
                )
            areas_m2 = np.asarray(get_areas()) * scale  # (M,) in m^2
            slip = self.total_slip(i)                    # (M,) in m
            moment += float(np.sum(shear_modulus * areas_m2 * slip))
        return moment

    def moment_magnitude(
        self,
        shear_modulus: float = 3.3e10,
        length_unit: str = "km",
    ) -> float:
        """
        Computes the moment magnitude :math:`M_w` of the slip distribution.

        Uses the Hanks & Kanamori (1979) relation
        :math:`M_w = \\tfrac{2}{3}\\,(\\log_{10} M_0 - 9.05)` with :math:`M_0`
        in newton-metres.

        Args:
            shear_modulus: Shear modulus in pascals (see :meth:`seismic_moment`).
            length_unit: Length unit of the mesh coordinates (see
                :meth:`seismic_moment`).

        Returns:
            The moment magnitude :math:`M_w` (dimensionless).

        Raises:
            ValueError: If the seismic moment is non-positive (e.g. zero slip),
                for which the magnitude is undefined.
        """
        m0 = self.seismic_moment(shear_modulus=shear_modulus, length_unit=length_unit)
        if m0 <= 0.0:
            raise ValueError(
                "Seismic moment is non-positive; moment magnitude is undefined "
                "for a zero slip distribution."
            )
        return (2.0 / 3.0) * (np.log10(m0) - 9.05)


class AbstractAssembler(ABC):
    """
    Abstract base class for strategies that assemble the linear system.

    This allows for flexible construction of the augmented matrix `A` and
    data vector `b` to accommodate complex inversion scenarios, such as
    multiple disjoint events or time-dependent problems.
    """

    @abstractmethod
    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles the augmented linear system (A, b).

        Args:
            faults: A list of fault models in the inversion.
            datasets: A list of geodetic datasets.
            engine: The Green's function calculation engine.
            regularization_manager: The manager for building smoothing matrices.
            lambda_spatial: The spatial regularization weight.

        Returns:
            A tuple containing the augmented matrix `A` and the data vector `b`.
        """
        pass


class VanillaAssembler(AbstractAssembler):
    """
    The standard assembler for a simple, single-event inversion with caching.
    
    It calculates the elastic Green's function matrix G once and caches it.
    Subsequent calls to assemble() reuse this matrix unless a rebuild is forced.

    Datasets carrying a nuisance basis (see
    :meth:`~slipkit.core.data.GeodeticDataSet.get_nuisance_basis`) contribute
    extra columns *after* all the slip columns, one block per dataset, placed
    block-diagonally so each dataset's ramp only sees its own rows. Those
    columns are not smoothed: the regularization block is zero-padded to match.
    """

    def __init__(self):
        # Cache storage
        self._G_elastic_cache: Optional[List[np.ndarray]] = None
        self._data_vector_cache: Optional[np.ndarray] = None
        self._sigma_inv_cache: Optional[List[np.ndarray]] = None

    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles the system (A, b), reusing cached Green's functions if available.

        Args:
            faults: List of fault models.
            datasets: List of geodetic datasets.
            engine: Physics engine.
            regularization_manager: Smoothing manager.
            lambda_spatial: Regularization weight.
            force_recompute: If True, clears cache and recalculates G.
        """
        
        # 1. Check if we need to compute the Elastic Kernels (G)
        if self._G_elastic_cache is None or force_recompute:
            self._compute_elastic_kernels(faults, datasets, engine)

        # 2. Apply Data Weights (Sigma)
        # We do this every time because sigma might theoretically change (though rare),
        # but the heavy lifting (calculating G) is already done.
        weighted_G_blocks = []
        weighted_data_blocks = []

        for i, dataset in enumerate(datasets):
            # Retrieve cached raw G for this dataset
            G_raw = self._G_elastic_cache[i]
            
            if dataset.sigma is None:
                raise ValueError(f"Dataset '{dataset.name}' missing sigma.")
            
            # Inverse covariance (weighting)
            # Flatten sigma if it's 1D, or handle full covariance if supported later
            sigma_inv = 1.0 / dataset.sigma
            
            # Apply weighting: W * G
            # Broadcasting: (N_data, 1) * (N_data, N_param)
            G_weighted = G_raw * sigma_inv[:, np.newaxis]
            d_weighted = dataset.data * sigma_inv

            weighted_G_blocks.append(G_weighted)
            weighted_data_blocks.append(d_weighted)

        # Stack the weighted blocks
        G_full_weighted = np.vstack(weighted_G_blocks)
        d_full_weighted = np.concatenate(weighted_data_blocks)

        # 3. Build Regularization Matrix (S)
        # This is fast relative to G, so we usually rebuild it to allow changing lambda
        S_reg = regularization_manager.build_smoothing_matrix(faults, lambda_spatial)

        # Guard: G columns, S columns, and the expected number of unknowns must
        # all agree. This catches any component-ordering / width mismatch between
        # the physics engine and the regularization manager.
        expected_cols = sum(f.num_components() * f.num_patches() for f in faults)
        if not (G_full_weighted.shape[1] == expected_cols == S_reg.shape[1]):
            raise ValueError(
                "Column mismatch while assembling the system: "
                f"G has {G_full_weighted.shape[1]} columns, "
                f"S has {S_reg.shape[1]} columns, "
                f"expected {expected_cols} unknowns "
                "(sum of num_components * num_patches over faults)."
            )

        # 3b. Nuisance (orbital-ramp) columns, block-diagonal over datasets and
        # weighted by the same sigma as their rows. The regularization block is
        # zero-padded so ramp coefficients stay unsmoothed and unpenalized.
        bases = nuisance_bases(datasets)
        widths = nuisance_widths(bases)
        n_nuisance = sum(widths)
        if n_nuisance:
            B_full = np.zeros((G_full_weighted.shape[0], n_nuisance))
            row = 0
            col = 0
            for dataset, basis, width in zip(datasets, bases, widths):
                n_rows = len(dataset)
                if width:
                    B_full[row:row + n_rows, col:col + width] = (
                        basis / dataset.sigma[:, np.newaxis]
                    )
                    col += width
                row += n_rows
            G_full_weighted = np.hstack([G_full_weighted, B_full])
            S_reg = sparse_hstack(
                [S_reg, csr_matrix((S_reg.shape[0], n_nuisance))], format="csr"
            )

        # 4. Assemble Final System
        # Regularization targets are usually zero (smoothness)
        zero_reg_vector = np.zeros(S_reg.shape[0])

        # Combine Data equations and Regularization equations
        # G_total = [ G_weighted ]
        #           [ S_reg      ]
        A_augmented = sparse_vstack([G_full_weighted, S_reg]).toarray()
        
        # b_total = [ d_weighted ]
        #           [ 0          ]
        b_augmented = np.concatenate([d_full_weighted, zero_reg_vector])

        return A_augmented, b_augmented

    def _compute_elastic_kernels(
        self, 
        faults: List[AbstractFaultModel], 
        datasets: List[GeodeticDataSet], 
        engine: GreenFunctionBuilder
    ):
        """
        Internal method to compute raw G matrices and store them in cache.
        This is the expensive step.
        """
        print("Computing elastic Green's functions (kernels)...")
        self._G_elastic_cache = []
        
        for dataset in datasets:
            # For a single dataset, G is [G_fault1 | G_fault2 | ...]
            current_dataset_G_parts = []
            
            for fault in faults:
                # Calculate G for this specific fault-dataset pair
                # Shape: (N_data_points, N_fault_patches * components)
                G_part = engine.build_kernel(fault, dataset)
                current_dataset_G_parts.append(G_part)
            
            # Concatenate horizontally to get G for this dataset across all faults
            G_dataset_full = np.concatenate(current_dataset_G_parts, axis=1)
            self._G_elastic_cache.append(G_dataset_full)
            
        print("Green's functions computed and cached.")

    def clear_cache(self):
        """Manually clears the kernel cache."""
        self._G_elastic_cache = None


# ---------------------------------------------------------------------------
# L-curve parallel workers
#
# These live at module scope (not as closures/methods) so they are importable
# by worker processes under the "spawn" start method used on macOS/Windows.
# The large, read-only arrays are shipped to each worker exactly once via the
# pool initializer and cached in these globals, so each task only receives a
# single scalar lambda.
# ---------------------------------------------------------------------------

_LC_G_WEIGHTED: Optional[np.ndarray] = None
_LC_D_WEIGHTED: Optional[np.ndarray] = None
_LC_L_BASE: Optional[np.ndarray] = None
_LC_SOLVER: Optional[SolverStrategy] = None
_LC_BOUNDS: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _lcurve_worker_init(g_weighted, d_weighted, l_base, solver, bounds):
    """
    Initializer: cache the shared L-curve arrays in the worker process.

    ``l_base`` arrives as a sparse matrix (cheap to ship) and is densified once
    here so the per-lambda solve just scales and stacks it.
    """
    global _LC_G_WEIGHTED, _LC_D_WEIGHTED, _LC_L_BASE, _LC_SOLVER, _LC_BOUNDS
    _LC_G_WEIGHTED = g_weighted
    _LC_D_WEIGHTED = d_weighted
    _LC_L_BASE = l_base.toarray() if hasattr(l_base, "toarray") else l_base
    _LC_SOLVER = solver
    _LC_BOUNDS = bounds


def _lcurve_solve_one(lambda_val: float) -> Tuple[float, float]:
    """
    Solves the regularized system for a single lambda and returns
    (misfit, roughness).

    The augmented system is A = [G_weighted; lambda * L_base], and because the
    smoothing block scales linearly with lambda, roughness = ||L_base @ m||.
    """
    A_aug = np.vstack([_LC_G_WEIGHTED, lambda_val * _LC_L_BASE])
    b_aug = np.concatenate([_LC_D_WEIGHTED, np.zeros(_LC_L_BASE.shape[0])])

    m = _LC_SOLVER.solve(A_aug, b_aug, _LC_BOUNDS)

    misfit = float(np.linalg.norm(_LC_G_WEIGHTED @ m - _LC_D_WEIGHTED))
    roughness = float(np.linalg.norm(_LC_L_BASE @ m))
    return misfit, roughness


def _solver_allows_negative(solver: SolverStrategy) -> bool:
    """
    Whether a solver can return negative unknowns.

    ``scipy.optimize.nnls`` enforces non-negativity intrinsically and ignores
    any bounds, so :class:`~slipkit.core.solvers.NnlsSolver` cannot carry
    free-sign nuisance parameters.
    """
    return not isinstance(solver, NnlsSolver)


def _extend_bounds(
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    n_slip: int,
    n_nuisance: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Pads slip-only bounds with ``(-inf, +inf)`` over the nuisance block.

    Bounds already covering the full width are returned unchanged, so callers
    that constrain the ramp themselves keep control.

    Args:
        bounds: ``(lower, upper)`` arrays, or None.
        n_slip: Number of slip unknowns.
        n_nuisance: Number of nuisance unknowns.

    Returns:
        Bounds of length ``n_slip + n_nuisance``, or None if none were given.

    Raises:
        ValueError: If the bounds match neither the slip width nor the full width.
    """
    if bounds is None or n_nuisance == 0:
        return bounds
    lower, upper = (np.asarray(b, dtype=float).ravel() for b in bounds)
    total = n_slip + n_nuisance
    if lower.shape[0] == total:
        return lower, upper
    if lower.shape[0] != n_slip:
        raise ValueError(
            f"bounds have length {lower.shape[0]}; expected {n_slip} (slip only) "
            f"or {total} (slip + nuisance)."
        )
    return (
        np.concatenate([lower, np.full(n_nuisance, -np.inf)]),
        np.concatenate([upper, np.full(n_nuisance, np.inf)]),
    )


class InversionOrchestrator:
    """
    The user-facing API that orchestrates the slip inversion process.

    This class ties together data, fault models, physics engines, and solvers
    to construct and solve the linear inverse problem. It uses a configurable
    `Assembler` strategy to build the linear system, allowing for flexibility.
    """

    def __init__(self):
        self.faults: List[AbstractFaultModel] = []
        self.datasets: List[GeodeticDataSet] = []
        self.engine: Optional[GreenFunctionBuilder] = None
        self.solver: Optional[SolverStrategy] = None
        self._regularization_manager: RegularizationManager = LaplacianSmoothing()
        self.assembler: AbstractAssembler = VanillaAssembler()  # Default assembler

    def add_fault(self, fault: AbstractFaultModel):
        """Adds a fault model to the inversion."""
        self.faults.append(fault)

    def add_data(self, dataset: GeodeticDataSet):
        """Adds a geodetic dataset to the inversion."""
        self.datasets.append(dataset)

    def set_engine(self, engine: GreenFunctionBuilder):
        """Sets the Green's function engine."""
        self.engine = engine

    def set_solver(self, solver: SolverStrategy):
        """Sets the solver strategy."""
        self.solver = solver
        
    def set_regularization(self, manager: RegularizationManager):
        """
        Sets the regularization strategy.

        Defaults to plain :class:`LaplacianSmoothing`. Wrap it in
        :class:`~slipkit.core.regularization.DeepEdgeDamping` to additionally
        damp slip along the fault's deep edge.

        Args:
            manager: An instance of a class implementing RegularizationManager.
        """
        self._regularization_manager = manager

    def set_assembler(self, assembler: AbstractAssembler):
        """
        Sets the assembly strategy for constructing the linear system.

        This allows the user to define custom logic for how the Green's function
        matrix and data vectors are constructed, for example in multi-event
        or time-dependent inversions.

        Args:
            assembler: An instance of a class implementing AbstractAssembler.
        """
        self.assembler = assembler

    def run_inversion(
        self, lambda_spatial: float, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> SlipDistribution:
        """
        Executes the slip inversion.

        This method orchestrates the assembly of the observation equations,
        regularization, and calls the specified solver.

        If any dataset carries a nuisance basis (an orbital ramp), its
        coefficients are solved jointly with slip and returned in
        :attr:`SlipDistribution.nuisance`; ``slip_vector`` stays slip-only.

        Args:
            lambda_spatial: The weighting parameter for spatial regularization.
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution vector.
                May be given for the slip unknowns alone; it is then extended
                with ``(-inf, +inf)`` over the nuisance block, since ramp
                coefficients are free-sign.

        Returns:
            A SlipDistribution object containing the inverted slip vector.

        Raises:
            ValueError: If nuisance parameters are requested while the solver
                cannot represent negative unknowns (e.g. :class:`NnlsSolver`).
        """
        if not self.faults:
            raise ValueError("No fault models added to the inversion.")
        if not self.datasets:
            raise ValueError("No geodetic datasets added to the inversion.")
        if self.engine is None:
            raise ValueError("No GreenFunctionBuilder engine has been set.")
        if self.solver is None:
            raise ValueError("No SolverStrategy has been set.")

        n_slip = sum(f.num_components() * f.num_patches() for f in self.faults)
        widths = nuisance_widths(nuisance_bases(self.datasets))
        n_nuisance = sum(widths)

        # Ramp coefficients are free-sign; a non-negative solver would silently
        # clamp them to zero (or to a one-sided ramp) instead of failing.
        if n_nuisance and not _solver_allows_negative(self.solver):
            raise ValueError(
                f"{type(self.solver).__name__} constrains every unknown to be "
                "non-negative, but nuisance (ramp) coefficients are free-sign. "
                "Use BoundedLsqSolver, with bounds (0, inf) on the slip block "
                "if you also want non-negative slip."
            )

        # --- 1. Assembly Phase (Delegated to strategy) ---
        A_augmented, b_augmented = self.assembler.assemble(
            self.faults,
            self.datasets,
            self.engine,
            self._regularization_manager,
            lambda_spatial,
        )

        # --- 2. Solve Phase ---
        bounds = _extend_bounds(bounds, n_slip, n_nuisance)
        solution_vector_m = self.solver.solve(A_augmented, b_augmented, bounds)

        # --- 3. Map Phase ---
        # If the solver produced a richer result object (e.g. AltarSlipDistribution
        # with posterior samples), return that directly instead of wrapping the
        # plain mean vector in a base SlipDistribution.
        if hasattr(self.solver, "get_last_result"):
            rich_result = self.solver.get_last_result()
            if rich_result is not None:
                return rich_result

        nuisance: Dict[str, Ramp] = {}
        offset = n_slip
        for dataset, width in zip(self.datasets, widths):
            if width:
                nuisance[dataset.name] = dataset.ramp.with_coeffs(
                    solution_vector_m[offset:offset + width]
                )
                offset += width
        return SlipDistribution(
            solution_vector_m[:n_slip], self.faults, nuisance=nuisance
        )

    def run_l_curve(
        self,
        lambdas: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        force_recompute: bool = False,
        n_jobs: Optional[int] = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the inversion for a range of lambda values to generate an L-curve.

        This method calculates the misfit (solution norm) and roughness (seminorm)
        for each provided regularization parameter. Because each lambda solve is
        fully independent, the sweep is embarrassingly parallel and is dispatched
        across multiple processes by default.

        The weighted Green's-function/data blocks and the base Laplacian are
        built once and reused for every lambda (the smoothing block scales
        linearly, ``S = lambda * L``), so each worker only reassembles and solves.

        Args:
            lambdas: An array of spatial regularization parameters to test.
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution.
            force_recompute: If True, forces re-computation of the elastic kernels.
            n_jobs: Number of worker processes. ``-1`` (default) uses all available
                cores (capped at the number of lambdas); ``1`` runs serially in
                this process (useful for debugging or tiny sweeps).

        Returns:
            A tuple containing:
            - The array of lambdas used.
            - The corresponding misfit array (rho).
            - The corresponding roughness array (eta).
        """
        if not self.faults:
            raise ValueError("No fault models added to the inversion.")
        if not self.datasets:
            raise ValueError("No geodetic datasets added to the inversion.")
        if self.engine is None:
            raise ValueError("No GreenFunctionBuilder engine has been set.")
        if self.solver is None:
            raise ValueError("No SolverStrategy has been set.")

        lambdas = np.asarray(lambdas)

        # Nuisance columns widen the system here exactly as in run_inversion,
        # so slip-only bounds need the same extension.
        n_slip = sum(f.num_components() * f.num_patches() for f in self.faults)
        n_nuisance = sum(nuisance_widths(nuisance_bases(self.datasets)))
        if n_nuisance and not _solver_allows_negative(self.solver):
            raise ValueError(
                f"{type(self.solver).__name__} constrains every unknown to be "
                "non-negative, but nuisance (ramp) coefficients are free-sign. "
                "Use BoundedLsqSolver."
            )
        bounds = _extend_bounds(bounds, n_slip, n_nuisance)

        # Ensure G is computed and cached before assembling.
        if force_recompute or self.assembler._G_elastic_cache is None:
            self.assembler._compute_elastic_kernels(self.faults, self.datasets, self.engine)

        num_data_points = sum(len(ds.data) for ds in self.datasets)

        # Build the shared blocks once by assembling with lambda = 1.0:
        #   A = [G_weighted; 1.0 * L_base],  b = [d_weighted; 0]
        A_ref, b_ref = self.assembler.assemble(
            self.faults,
            self.datasets,
            self.engine,
            self._regularization_manager,
            1.0,
            force_recompute=False,
        )
        g_weighted = np.ascontiguousarray(A_ref[:num_data_points, :])
        d_weighted = np.ascontiguousarray(b_ref[:num_data_points])
        # The rows below the data are exactly S at lambda = 1, i.e. L_base, since
        # S = lambda * L for every manager. Reading them back out (rather than
        # rebuilding from the manager) keeps any nuisance zero-padding in step
        # with the width of g_weighted.
        l_base = np.ascontiguousarray(A_ref[num_data_points:, :])

        # Resolve worker count.
        if n_jobs is None or n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_jobs = max(1, min(n_jobs, len(lambdas)))

        if n_jobs == 1:
            # Serial path (no process-pool overhead).
            _lcurve_worker_init(g_weighted, d_weighted, l_base, self.solver, bounds)
            results = [_lcurve_solve_one(float(lam)) for lam in lambdas]
        else:
            # Keep BLAS single-threaded in the workers so N processes don't
            # oversubscribe the cores. Set before spawning: children inherit the
            # env at interpreter startup, i.e. before they import numpy.
            thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            saved = {v: os.environ.get(v) for v in thread_vars}
            for v in thread_vars:
                os.environ[v] = "1"
            try:
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_lcurve_worker_init,
                    initargs=(g_weighted, d_weighted, l_base, self.solver, bounds),
                ) as executor:
                    # map preserves input order, so results align with `lambdas`.
                    results = list(executor.map(_lcurve_solve_one, [float(l) for l in lambdas]))
            finally:
                # Restore the parent's original environment.
                for v, val in saved.items():
                    if val is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = val

        misfits = np.array([r[0] for r in results])
        roughnesses = np.array([r[1] for r in results])

        return lambdas, misfits, roughnesses
