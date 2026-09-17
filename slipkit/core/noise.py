"""
Per-dataset noise models for slip inversion and resolution analysis.

A :class:`NoiseModel` is the single source of truth for a dataset's noise: it
supplies the diagonal ``sigma`` used to whiten the linear system (feeding the
``Sigma`` matrix in :class:`~slipkit.core.inversion.VanillaAssembler`) and it
draws noise realizations used by every empirical resolution test (checkerboard,
Monte-Carlo error propagation, ...).

Two concrete models are provided:

* :class:`DiagonalNoise` -- uncorrelated, from reported per-point sigma. This is
  the honest model for GNSS (``SigE/SigN/SigU``) or any dataset that already
  carries real uncertainties in ``dataset.sigma``.
* :class:`EmpiricalInsarNoise` -- InSAR noise estimated empirically from a
  low-signal / far-field ("quiet") region where the true displacement is ~0, so
  the scatter there *is* the noise. Both the amplitude (per-point sigma) and the
  spatial correlation (atmosphere/ionosphere) are estimated by fitting a
  nugget + exponential covariance model to the region's empirical semivariogram.

Why the spatial correlation matters: InSAR errors are spatially correlated, so
treating points as independent *underestimates* model uncertainty (a correlated
error field mimics a smooth deformation signal). The correlated ``sample()``
draws are exactly what lets the Monte-Carlo ensemble capture this.

The linear inversion keeps whitening by the *diagonal* ``sigma`` (no assembler
change); the full covariance ``C_d`` is consumed only by the correlated
``sample()`` / :meth:`NoiseModel.covariance` path used by Monte-Carlo.
"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel


class NoiseModel(ABC):
    """Per-dataset noise model.

    Supplies the diagonal ``sigma`` used to whiten the inversion and draws noise
    realizations for the empirical resolution tests. Optionally exposes a full
    ``(N, N)`` covariance for correlated draws.
    """

    @abstractmethod
    def sigma(self) -> np.ndarray:
        """Returns the ``(N,)`` per-point standard deviation (feeds ``Sigma``)."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draws one ``(N,)`` noise realization.

        Args:
            rng: A NumPy random generator (e.g. ``np.random.default_rng(seed)``)
                so ensembles are reproducible and independent across workers.

        Returns:
            A single noise vector, zero-mean, with this model's (co)variance.
        """
        raise NotImplementedError

    def covariance(self) -> Optional[np.ndarray]:
        """Returns the full ``(N, N)`` data covariance ``C_d`` if modelled.

        Returns ``None`` for purely diagonal models, where the covariance is
        implicitly ``diag(sigma**2)`` and never needs to be materialised.
        """
        return None

    def sample_batch(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """Draws ``k`` realizations at once as an ``(N, k)`` array.

        The default stacks ``k`` calls to :meth:`sample`; correlated models
        override this to factor the covariance **once** and draw the whole batch
        with a single matmul (so an ``(N, N)`` factor is built once, not ``k``
        times). Batch-drawing is how callers avoid rebuilding a dense covariance
        per realization / per worker.
        """
        return np.stack([self.sample(rng) for _ in range(int(k))], axis=1)

    def free_cache(self) -> None:
        """Releases any cached dense covariance / factor. No-op by default."""
        return None

    def __len__(self) -> int:
        """Returns the number of data points ``N``."""
        return int(self.sigma().shape[0])


class DiagonalNoise(NoiseModel):
    """Uncorrelated Gaussian noise from reported per-point sigma.

    The right model for GNSS (from ``SigE/SigN/SigU``) or any dataset whose
    ``sigma`` already reflects real uncertainties. ``sample()`` returns
    ``sigma * N(0, I)``.
    """

    def __init__(self, sigma: np.ndarray):
        """Initializes the model from a per-point standard deviation.

        Args:
            sigma: ``(N,)`` per-point standard deviation. Must be positive.
        """
        sigma_arr = np.asarray(sigma, dtype=float).ravel()
        if sigma_arr.ndim != 1 or sigma_arr.size == 0:
            raise ValueError("sigma must be a non-empty 1-D array.")
        if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0.0):
            raise ValueError("sigma must be finite and strictly positive.")
        self._sigma = sigma_arr

    @classmethod
    def from_dataset(cls, dataset: GeodeticDataSet) -> "DiagonalNoise":
        """Builds the model directly from ``dataset.sigma``.

        Args:
            dataset: A dataset carrying real per-point uncertainties (e.g. GNSS).

        Returns:
            A :class:`DiagonalNoise` using the dataset's reported sigma.
        """
        if dataset.sigma is None:
            raise ValueError(
                f"Dataset '{dataset.name}' has no sigma; cannot build "
                "DiagonalNoise. Provide a NoiseModel explicitly (e.g. "
                "EmpiricalInsarNoise.from_quiet_region for InSAR)."
            )
        return cls(dataset.sigma)

    def sigma(self) -> np.ndarray:
        return self._sigma

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self._sigma * rng.standard_normal(self._sigma.shape[0])

    def sample_batch(self, rng: np.random.Generator, k: int) -> np.ndarray:
        return self._sigma[:, None] * rng.standard_normal((self._sigma.shape[0], int(k)))


def _variogram_exponential(h: np.ndarray, nugget: float, sill: float, length: float) -> np.ndarray:
    """Exponential semivariogram model.

    ``gamma(h) = nugget + sill * (1 - exp(-h / length))``. As ``h -> inf`` the
    semivariance approaches the total sill ``nugget + sill`` (the field
    variance); at ``h = 0`` it is the ``nugget``.
    """
    return nugget + sill * (1.0 - np.exp(-h / length))


def detrend_plane(coords_xy: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Removes a best-fit planar ramp ``a + b*x + c*y`` from ``values``.

    Estimating noise from a quiet region assumes stationarity, but a real
    far-field still carries a long-wavelength **orbital ramp** (and broad
    atmosphere). Subtracting only the mean leaves that ramp, so the semivariogram
    never plateaus and the fit diverges. Removing a planar trend restores
    (approximate) stationarity so the variogram saturates.

    Args:
        coords_xy: ``(N, 2)`` horizontal coordinates.
        values: ``(N,)`` field values.

    Returns:
        The ``(N,)`` residual after subtracting the least-squares plane.
    """
    xy = np.asarray(coords_xy, dtype=float)
    v = np.asarray(values, dtype=float).ravel()
    design = np.column_stack([np.ones(xy.shape[0]), xy[:, 0], xy[:, 1]])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ coef


def empirical_variogram(
    coords_xy: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 15,
    max_dist: Optional[float] = None,
    max_pairs: int = 2_000_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes an isotropic empirical semivariogram.

    For every pair of points the semivariance ``0.5 * (z_i - z_j)**2`` is binned
    by horizontal separation distance.

    Args:
        coords_xy: ``(N, 2)`` horizontal coordinates of the points.
        values: ``(N,)`` field values (residuals in a quiet region).
        n_bins: Number of distance bins.
        max_dist: Largest separation to include. Defaults to half the maximum
            pairwise distance (the usual variogram cutoff, where bins stay
            well-populated).
        max_pairs: If the number of pairs exceeds this, a random subset of
            points is used to keep the computation tractable.
        rng: Generator used only for the optional subsampling (reproducibility).

    Returns:
        A tuple ``(bin_centers, gamma, counts)`` of the bin centre distances,
        the mean semivariance per bin, and the pair count per bin. Empty bins
        are dropped.
    """
    xy = np.asarray(coords_xy, dtype=float)
    z = np.asarray(values, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords_xy must have shape (N, 2).")
    if z.shape[0] != xy.shape[0]:
        raise ValueError("values and coords_xy must have the same length N.")
    n = xy.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to estimate a variogram.")

    # Subsample points if the pair count would be excessive (~ n^2 / 2).
    if n * (n - 1) // 2 > max_pairs:
        rng = rng if rng is not None else np.random.default_rng()
        m = int((1 + np.sqrt(1 + 8 * max_pairs)) / 2)
        idx = rng.choice(n, size=min(m, n), replace=False)
        xy = xy[idx]
        z = z[idx]

    dist = pdist(xy)                       # (P,) pairwise horizontal distances
    semi = 0.5 * pdist(z[:, None]) ** 2    # (P,) pairwise semivariances

    if max_dist is None:
        max_dist = float(dist.max()) / 2.0
    if max_dist <= 0.0:
        raise ValueError("All points are coincident; cannot form a variogram.")

    edges = np.linspace(0.0, max_dist, n_bins + 1)
    which = np.digitize(dist, edges) - 1
    in_range = (which >= 0) & (which < n_bins)
    which = which[in_range]
    semi = semi[in_range]

    counts = np.bincount(which, minlength=n_bins).astype(float)
    sums = np.bincount(which, weights=semi, minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    nonempty = counts > 0
    with np.errstate(invalid="ignore"):
        gamma = np.where(nonempty, sums / np.where(nonempty, counts, 1.0), np.nan)
    return centers[nonempty], gamma[nonempty], counts[nonempty]


def fit_variogram(
    coords_xy: np.ndarray,
    values: np.ndarray,
    *,
    model: str = "exponential",
    n_bins: int = 15,
    max_dist: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Fits a nugget + exponential model to the empirical semivariogram.

    Fitting *with* a nugget is important: without it the exponential term is
    forced to absorb the uncorrelated (decorrelation/thermal) noise, biasing the
    correlation length ``L`` low (see the plan's noise-model discussion).

    Args:
        coords_xy: ``(N, 2)`` horizontal coordinates.
        values: ``(N,)`` field values (quiet-region residuals).
        model: Only ``"exponential"`` is supported for now.
        n_bins: Number of variogram bins.
        max_dist: Variogram distance cutoff (see :func:`empirical_variogram`).

    Returns:
        ``(nugget, sill, length)`` -- the uncorrelated variance ``sigma_n**2``,
        the correlated partial sill ``sigma_c**2``, and the correlation length
        ``L`` (in the coordinates' units). The total per-point variance is
        ``nugget + sill``.
    """
    if model != "exponential":
        raise ValueError(f"Unsupported variogram model '{model}'.")

    centers, gamma, counts = empirical_variogram(
        coords_xy, values, n_bins=n_bins, max_dist=max_dist
    )
    if centers.size < 3:
        raise ValueError(
            "Too few populated variogram bins to fit; supply a larger quiet "
            "region or reduce n_bins."
        )

    var = float(np.var(np.asarray(values, dtype=float)))
    extent = float(centers.max())
    eps = np.finfo(float).eps
    # Initial guesses: split variance evenly between nugget and sill; length ~
    # a third of the fitted range.
    p0 = [0.5 * var, 0.5 * var, max(extent / 3.0, eps)]
    # Bound the correlation length at the region extent: a correlation length
    # cannot exceed the data it is measured over. Without this the fit is
    # degenerate on a non-saturating (e.g. ramped) variogram -- for h << L the
    # model is linear with slope sill/L, so sill and L run off to infinity
    # together (giving physically impossible L and metre-scale sigma).
    bounds = ([0.0, 0.0, eps], [np.inf, np.inf, max(extent, eps)])

    try:
        popt, _ = curve_fit(
            _variogram_exponential,
            centers,
            gamma,
            p0=p0,
            bounds=bounds,
            sigma=1.0 / np.sqrt(counts),  # down-weight sparsely-populated bins
            maxfev=10_000,
        )
        nugget, sill, length = (float(popt[0]), float(popt[1]), float(popt[2]))
    except (RuntimeError, ValueError):
        # Fall back to a pure-nugget (white-noise) model from the sample variance.
        nugget, sill, length = var, 0.0, extent

    # A zero total sill is unusable downstream; back off to the sample variance.
    if nugget + sill <= 0.0:
        nugget, sill = var, 0.0

    # If L rails at the extent, the variogram never plateaued: the region still
    # carries long-wavelength signal (orbital ramp / atmosphere / deformation),
    # so it is not noise-stationary and L (and the sill) are unreliable.
    if length >= 0.99 * extent:
        warnings.warn(
            "Variogram did not saturate within the fitting distance: the region "
            "is likely non-stationary (residual ramp/signal), so the correlation "
            "length and sill are unreliable. Use a smaller/known-quiet region, "
            "detrend it, or cap max_dist.",
            UserWarning,
        )
    return nugget, sill, length


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask of which ``points`` fall inside ``polygon`` (ray casting).

    Args:
        points: ``(N, 2)`` query points.
        polygon: ``(K, 2)`` polygon vertices (open or closed ring).

    Returns:
        ``(N,)`` boolean mask, ``True`` for points strictly inside or on the
        boundary of the polygon.
    """
    poly = np.asarray(polygon, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + np.finfo(float).eps) + xi
        )
        inside ^= crosses
        j = i
    return inside


def select_quiet_region(
    dataset: GeodeticDataSet,
    *,
    region: Optional[Union[np.ndarray, Sequence[float]]] = None,
    faults: Optional[Union[AbstractFaultModel, Sequence[AbstractFaultModel]]] = None,
    quantile: float = 0.75,
) -> np.ndarray:
    """Selects the low-signal / far-field points used to estimate InSAR noise.

    Args:
        dataset: The InSAR dataset.
        region: How to choose the quiet points. One of:

            * ``None`` -- auto-select: points farther than ``quantile`` of the
              distance-to-fault distribution *and* below the median ``|LOS|``.
              Requires ``faults`` for the fault distances.
            * a boolean mask ``(N,)`` -- used directly.
            * a bbox ``(xmin, ymin, xmax, ymax)`` -- points inside the box.
            * a polygon ``(K, 2)`` -- points inside the polygon.
        faults: Fault model(s), required only for the auto path (to measure
            distance from the fault).
        quantile: Distance-to-fault quantile above which points count as
            far-field, for the auto path.

    Returns:
        A ``(N,)`` boolean mask selecting the quiet points.
    """
    coords = np.asarray(dataset.coords, dtype=float)
    n = coords.shape[0]

    if region is not None:
        region_arr = np.asarray(region)
        if region_arr.dtype == bool:
            if region_arr.shape[0] != n:
                raise ValueError("Boolean region mask must have length N.")
            mask = region_arr
        elif region_arr.ndim == 1 and region_arr.size == 4:
            xmin, ymin, xmax, ymax = region_arr.astype(float)
            mask = (
                (coords[:, 0] >= xmin)
                & (coords[:, 0] <= xmax)
                & (coords[:, 1] >= ymin)
                & (coords[:, 1] <= ymax)
            )
        elif region_arr.ndim == 2 and region_arr.shape[1] == 2:
            mask = _point_in_polygon(coords[:, :2], region_arr.astype(float))
        else:
            raise ValueError(
                "region must be a boolean mask (N,), a bbox (xmin, ymin, xmax, "
                "ymax), or a polygon (K, 2)."
            )
    else:
        if faults is None:
            raise ValueError(
                "Auto quiet-region selection needs fault geometry; pass "
                "faults=... or an explicit region."
            )
        fault_list: List[AbstractFaultModel] = (
            [faults] if isinstance(faults, AbstractFaultModel) else list(faults)
        )
        centroids = np.vstack([f.get_centroids() for f in fault_list])
        # Nearest-fault distance via a KD-tree: O(N) memory, versus an
        # (N, M) cdist matrix that OOMs on a full-resolution scene (~1e5-1e6
        # points) when estimating noise from the real (un-downsampled) data.
        dist_to_fault, _ = cKDTree(centroids).query(coords, k=1)
        far = dist_to_fault >= np.quantile(dist_to_fault, quantile)
        low = np.abs(np.asarray(dataset.data, dtype=float)) <= np.median(
            np.abs(dataset.data)
        )
        mask = far & low

    if not np.any(mask):
        raise ValueError(
            "Quiet-region selection produced no points; relax the region or "
            "quantile."
        )
    return mask


class EmpiricalInsarNoise(NoiseModel):
    """InSAR noise estimated empirically from a quiet (low-signal) region.

    The amplitude (total sill ``sigma_n**2 + sigma_c**2``) sets the per-point
    ``sigma`` that feeds the whitening ``Sigma``. When ``correlated`` is set, the
    fitted exponential covariogram
    ``C_d(r) = sigma_n**2 * delta(r) + sigma_c**2 * exp(-r / L)`` defines the
    full ``(N, N)`` covariance used for correlated ``sample()`` draws.

    The covariance and its Cholesky factor are built lazily on first use and
    cached, since ``C_d`` is ``(N, N)`` and only needed for Monte-Carlo.
    """

    def __init__(
        self,
        coords_xy: np.ndarray,
        *,
        nugget: float,
        sill: float,
        length: float,
        correlated: bool = True,
        name: str = "InSAR",
    ):
        """Initializes the model from fitted covariance parameters.

        Prefer :meth:`from_quiet_region`, which fits these from data.

        Args:
            coords_xy: ``(N, 2)`` horizontal coordinates of *all* dataset points
                (used to build ``C_d`` for correlated draws).
            nugget: Uncorrelated variance ``sigma_n**2`` (>= 0).
            sill: Correlated partial sill ``sigma_c**2`` (>= 0).
            length: Correlation length ``L`` (> 0), in the coordinates' units.
            correlated: If True, ``sample()`` draws correlated noise from ``C_d``;
                if False it uses the diagonal ``sigma`` only.
            name: Label, for error messages.
        """
        xy = np.asarray(coords_xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coords_xy must have shape (N, 2).")
        if nugget < 0.0 or sill < 0.0:
            raise ValueError("nugget and sill must be non-negative.")
        if length <= 0.0:
            raise ValueError("length must be strictly positive.")
        total = nugget + sill
        if total <= 0.0:
            raise ValueError("Total variance (nugget + sill) must be positive.")

        self._coords_xy = xy
        self.nugget = float(nugget)
        self.sill = float(sill)
        self.length = float(length)
        self.correlated = bool(correlated)
        self.name = name

        self._n = xy.shape[0]
        self._sigma = np.full(self._n, np.sqrt(total))
        self._cov: Optional[np.ndarray] = None
        self._chol: Optional[np.ndarray] = None

    @staticmethod
    def _fit_region(
        dataset, region, faults, model, quantile, n_bins, max_dist, detrend,
        max_points,
    ) -> Tuple[float, float, float]:
        """Selects the quiet region, (de)trends it, and fits the variogram.

        A full-resolution reference can have tens of millions of points, so it is
        randomly **thinned** to ``max_points`` before any O(N) work. Random
        thinning preserves the variogram exactly (a random subsample of a
        stationary field has the same covariance), unlike box-averaging, so a few
        hundred thousand points give the same fit at a tiny fraction of the memory.
        """
        coords = np.asarray(dataset.coords, dtype=float)
        data = np.asarray(dataset.data, dtype=float)
        n = coords.shape[0]

        # Boolean per-point masks can't survive thinning; only thin the auto /
        # bbox / polygon paths (which reselect from whatever points remain).
        is_bool_mask = (
            region is not None and np.asarray(region).dtype == bool
        )
        if max_points is not None and n > int(max_points) and not is_bool_mask:
            # replace=True keeps this O(max_points) in memory (no length-N
            # permutation); a few duplicate points are harmless for a variogram.
            sel = np.random.default_rng(0).integers(0, n, size=int(max_points))
            coords = coords[sel]
            data = data[sel]

        m = coords.shape[0]
        thinned = GeodeticDataSet(
            coords, data, np.zeros((m, 3)), np.ones(m),
            name=getattr(dataset, "name", "reference"),
        )
        mask = select_quiet_region(
            thinned, region=region, faults=faults, quantile=quantile
        )
        quiet_xy = coords[mask, :2]
        quiet_vals = data[mask]
        # Remove a planar ramp (orbital/atmosphere), not just the mean, so the
        # region is (approximately) stationary and the variogram saturates.
        quiet_vals = (
            detrend_plane(quiet_xy, quiet_vals)
            if detrend
            else quiet_vals - quiet_vals.mean()
        )
        return fit_variogram(
            quiet_xy, quiet_vals, model=model, n_bins=n_bins, max_dist=max_dist
        )

    @classmethod
    def from_quiet_region(
        cls,
        dataset: GeodeticDataSet,
        *,
        region: Optional[Union[np.ndarray, Sequence[float]]] = None,
        faults: Optional[Union[AbstractFaultModel, Sequence[AbstractFaultModel]]] = None,
        model: str = "exponential",
        correlated: bool = True,
        quantile: float = 0.75,
        n_bins: int = 15,
        max_dist: Optional[float] = None,
        detrend: bool = True,
        max_points: Optional[int] = 300_000,
    ) -> "EmpiricalInsarNoise":
        """Fits the noise model from a quiet region of an InSAR dataset.

        The quiet region (where true displacement ~ 0) is selected via
        :func:`select_quiet_region`; its scatter *is* the noise. A nugget +
        exponential covariance is fitted to the region's empirical
        semivariogram, and the result parameterises the whole-dataset model.

        .. warning::
            Fit on the **full-resolution** point cloud, not on quadtree /
            box-averaged points: averaging smooths the scatter and destroys the
            short-lag correlation, so the variogram is under-sampled and the fit
            unstable. To estimate on full-res data but apply the covariance to a
            downsampled inversion dataset, use :meth:`from_reference`.

        Args:
            dataset: The InSAR dataset to model.
            region: Quiet-region selector (mask, bbox, polygon, or ``None`` for
                auto). See :func:`select_quiet_region`.
            faults: Fault model(s), needed only for the auto region path.
            model: Covariance model; only ``"exponential"`` is supported.
            correlated: Whether ``sample()`` produces correlated draws.
            quantile: Far-field distance quantile for the auto region path.
            n_bins: Number of variogram bins.
            max_dist: Variogram distance cutoff.
            detrend: If True (default), remove a planar ramp from the region
                before fitting (see :func:`detrend_plane`).
            max_points: Randomly thin the dataset to at most this many points
                before fitting (bounds memory/time on full-resolution scenes;
                thinning preserves the variogram). ``None`` disables thinning.
                Ignored when ``region`` is a boolean per-point mask.

        Returns:
            A fitted :class:`EmpiricalInsarNoise` for the full dataset.
        """
        nugget, sill, length = cls._fit_region(
            dataset, region, faults, model, quantile, n_bins, max_dist, detrend,
            max_points,
        )
        return cls(
            np.asarray(dataset.coords, dtype=float)[:, :2],
            nugget=nugget,
            sill=sill,
            length=length,
            correlated=correlated,
            name=dataset.name,
        )

    @classmethod
    def from_reference(
        cls,
        target_dataset: GeodeticDataSet,
        reference_dataset: GeodeticDataSet,
        *,
        region: Optional[Union[np.ndarray, Sequence[float]]] = None,
        faults: Optional[Union[AbstractFaultModel, Sequence[AbstractFaultModel]]] = None,
        model: str = "exponential",
        correlated: bool = True,
        quantile: float = 0.75,
        n_bins: int = 15,
        max_dist: Optional[float] = None,
        detrend: bool = True,
        max_points: Optional[int] = 300_000,
    ) -> "EmpiricalInsarNoise":
        """Fits the variogram on a full-resolution reference, applies it to a target.

        The correct workflow for a downsampled inversion: estimate the noise on
        the **full-resolution** point cloud (``reference_dataset`` -- read the
        same track with ``downsample_method='uniform', downsample_factor=1`` so no
        averaging occurs), then attach the fitted covariance to the
        (quadtree-)downsampled points actually used in the inversion
        (``target_dataset``). Estimating directly on the averaged points is wrong
        (see :meth:`from_quiet_region`).

        The quiet region is selected on the *reference* dataset; the resulting
        ``(nugget, sill, length)`` parameterise the covariance built on the
        *target* dataset's coordinates.

        .. note::
            A full-resolution reference can have tens of millions of points; it is
            randomly thinned to ``max_points`` before fitting (thinning preserves
            the variogram, so the fit is unchanged but memory stays bounded).
            **Read each reference inside a loop and ``del`` it after** so only one
            full scene is resident at a time.

        .. note::
            The per-point ``sigma`` is the full-resolution (pixel-level) noise;
            applied to box-averaged target points it is a **conservative
            (slightly high) first-order** estimate, since averaging reduces the
            uncorrelated part. For ``L`` >> box size the correlated part
            dominates and the overestimate is small.

        Args:
            target_dataset: The (downsampled) dataset used in the inversion; its
                coordinates define the covariance ``C_d`` and ``sigma`` length.
            reference_dataset: The full-resolution dataset used only to fit the
                variogram. Must be the same track/scene as ``target_dataset``.
            region / faults / quantile: Quiet-region selection on the reference
                (see :func:`select_quiet_region`).
            model / correlated / n_bins / max_dist / detrend / max_points: As in
                :meth:`from_quiet_region`.

        Returns:
            A fitted :class:`EmpiricalInsarNoise` on the target's coordinates.
        """
        nugget, sill, length = cls._fit_region(
            reference_dataset, region, faults, model, quantile, n_bins, max_dist,
            detrend, max_points,
        )
        return cls(
            np.asarray(target_dataset.coords, dtype=float)[:, :2],
            nugget=nugget,
            sill=sill,
            length=length,
            correlated=correlated,
            name=target_dataset.name,
        )

    def sigma(self) -> np.ndarray:
        return self._sigma

    def covariance(self) -> Optional[np.ndarray]:
        """Returns the full ``(N, N)`` covariance ``C_d`` (or ``None`` if diagonal).

        For an uncorrelated model (``correlated=False`` or ``sill == 0``) this
        returns ``None`` -- the covariance is just ``diag(sigma**2)``.
        """
        if not self.correlated or self.sill == 0.0:
            return None
        if self._cov is None:
            dist = squareform(pdist(self._coords_xy))
            cov = self.sill * np.exp(-dist / self.length)
            # Nugget on the diagonal makes the total diagonal variance
            # nugget + sill == sigma**2 and keeps C_d well-conditioned for the
            # Cholesky factorisation.
            cov[np.diag_indices_from(cov)] += self.nugget
            self._cov = cov
        return self._cov

    def _cholesky(self) -> np.ndarray:
        """Lower Cholesky factor of ``C_d``, cached (jittered if needed).

        Frees the dense covariance once factored -- only the factor is needed for
        sampling, so this halves the peak ``O(N**2)`` memory.
        """
        if self._chol is None:
            cov = self.covariance()
            assert cov is not None  # only called on the correlated path
            jitter = 0.0
            base = 1e-10 * float(np.mean(np.diag(cov)))
            while True:
                try:
                    self._chol = np.linalg.cholesky(cov + jitter * np.eye(self._n))
                    break
                except np.linalg.LinAlgError:
                    jitter = base if jitter == 0.0 else jitter * 10.0
                    if jitter > 1e-3 * float(np.mean(np.diag(cov))):
                        raise
            self._cov = None  # drop the covariance; keep only its factor
        return self._chol

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal(self._n)
        if not self.correlated or self.sill == 0.0:
            return self._sigma * z
        return self._cholesky() @ z

    def sample_batch(self, rng: np.random.Generator, k: int) -> np.ndarray:
        k = int(k)
        if not self.correlated or self.sill == 0.0:
            return self._sigma[:, None] * rng.standard_normal((self._n, k))
        # Factor once, draw the whole batch with one matmul.
        return self._cholesky() @ rng.standard_normal((self._n, k))

    def free_cache(self) -> None:
        """Releases the cached dense covariance and Cholesky factor (``O(N**2)``)."""
        self._cov = None
        self._chol = None


def estimate_insar_sigma(
    dataset: GeodeticDataSet,
    *,
    region: Optional[Union[np.ndarray, Sequence[float]]] = None,
    faults: Optional[Union[AbstractFaultModel, Sequence[AbstractFaultModel]]] = None,
    quantile: float = 0.75,
) -> float:
    """Estimates a single InSAR noise standard deviation from a quiet region.

    A convenience wrapper for the common case of replacing the InSAR
    ``sigma = ones`` placeholder with a real scalar amplitude, without fitting
    the full spatial-correlation model.

    Args:
        dataset: The InSAR dataset.
        region: Quiet-region selector (see :func:`select_quiet_region`).
        faults: Fault model(s), needed only for the auto region path.
        quantile: Far-field distance quantile for the auto path.

    Returns:
        The scalar noise standard deviation (the quiet region's scatter).
    """
    mask = select_quiet_region(dataset, region=region, faults=faults, quantile=quantile)
    vals = np.asarray(dataset.data, dtype=float)[mask]
    return float(np.std(vals - vals.mean(), ddof=1))
