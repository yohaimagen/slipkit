"""
Generic containers for observed geodetic displacements, plus the optional
nuisance (orbital-ramp) parameterization that rides along with a dataset.
"""

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Union
import numpy as np


def _ramp_exponents(degree: int) -> List[tuple]:
    """
    Returns the ``(i, j)`` exponents of every monomial ``x**i * y**j`` with
    ``i + j <= degree``, ordered by total degree then by ``i``.

    For ``degree = 2`` this is ``1, y, x, y**2, x*y, x**2`` -- i.e.
    ``P = (degree + 1) * (degree + 2) / 2`` terms.
    """
    if degree < 0:
        raise ValueError(f"Ramp degree must be >= 0; got {degree}.")
    return [(i, t - i) for t in range(degree + 1) for i in range(t + 1)]


@dataclass
class Ramp:
    """
    A polynomial nuisance surface (orbital ramp) attached to one dataset.

    The ramp is evaluated on the dataset's horizontal local coordinates,
    centred and scaled to keep the basis columns of order one::

        u = (x - center[0]) / scale,  v = (y - center[1]) / scale
        basis columns = u**i * v**j   for i + j <= degree

    Without that normalization a quadratic term in local km is ~1e4 times the
    constant term, and the ramp block carries no regularization to absorb the
    resulting conditioning. Because ``center`` and ``scale`` travel with the
    fitted ``coeffs``, the same ramp can later be evaluated at *other* points
    (e.g. a full-resolution LOS table) without any extra bookkeeping.

    Attributes:
        degree: Polynomial degree. 0 = constant LOS offset, 1 = planar ramp,
            2 = quadratic surface.
        center: ``(2,)`` horizontal centre of the observation cloud, in the
            mesh's length units (local km).
        scale: Half-extent used to normalize coordinates, same units.
        coeffs: ``(P,)`` fitted coefficients, or None before the inversion.
    """

    degree: int
    center: np.ndarray
    scale: float
    coeffs: Optional[np.ndarray] = None

    def __post_init__(self):
        self.degree = int(self.degree)
        self.center = np.asarray(self.center, dtype=float).ravel()
        if self.center.shape != (2,):
            raise ValueError(f"Ramp center must have shape (2,); got {self.center.shape}.")
        self.scale = float(self.scale)
        if not self.scale > 0.0:
            raise ValueError(f"Ramp scale must be positive; got {self.scale}.")
        if self.coeffs is not None:
            self.coeffs = np.asarray(self.coeffs, dtype=float).ravel()
            if self.coeffs.shape[0] != self.num_params:
                raise ValueError(
                    f"Ramp of degree {self.degree} needs {self.num_params} "
                    f"coefficients; got {self.coeffs.shape[0]}."
                )

    @classmethod
    def for_dataset(cls, dataset: "GeodeticDataSet", degree: int) -> "Ramp":
        """
        Builds a ramp whose normalization is taken from a dataset's coordinates.

        Args:
            dataset: The dataset the ramp will be estimated for.
            degree: Polynomial degree (0, 1, 2, ...).

        Returns:
            An un-fitted :class:`Ramp` (``coeffs is None``).
        """
        xy = np.asarray(dataset.coords, dtype=float)[:, :2]
        center = xy.mean(axis=0)
        half_extent = float(np.abs(xy - center).max())
        # A degenerate cloud (all points coincident) still needs a usable scale;
        # with degree 0 the basis is constant anyway.
        scale = half_extent if half_extent > 0.0 else 1.0
        return cls(degree=degree, center=center, scale=scale)

    @property
    def num_params(self) -> int:
        """Number of coefficients, ``(degree + 1) * (degree + 2) / 2``."""
        return (self.degree + 1) * (self.degree + 2) // 2

    def basis(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluates the normalized polynomial basis at observation points.

        Args:
            coords: ``(N, 2)`` or ``(N, 3)`` coordinates in the mesh's frame;
                only the horizontal components are used.

        Returns:
            An ``(N, P)`` design matrix.
        """
        xy = np.asarray(coords, dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"coords must be (N, 2) or (N, 3); got {xy.shape}.")
        u, v = ((xy[:, :2] - self.center) / self.scale).T
        return np.column_stack(
            [u ** i * v ** j for i, j in _ramp_exponents(self.degree)]
        )

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluates the fitted ramp at observation points.

        Args:
            coords: ``(N, 2)`` or ``(N, 3)`` coordinates in the mesh's frame.

        Returns:
            An ``(N,)`` array of nuisance displacement, in the data's units.

        Raises:
            ValueError: If the ramp has not been fitted (``coeffs is None``).
        """
        if self.coeffs is None:
            raise ValueError("Ramp has no coefficients; it has not been fitted yet.")
        return self.basis(coords) @ self.coeffs

    def with_coeffs(self, coeffs: np.ndarray) -> "Ramp":
        """Returns a copy of this ramp carrying the given coefficients."""
        return replace(self, coeffs=np.asarray(coeffs, dtype=float).ravel())


class GeodeticDataSet:
    """
    A generic container for observed displacements.

    This class is agnostic to the data source (e.g., InSAR, GNSS).
    """

    def __init__(
        self,
        coords: np.ndarray,
        data: np.ndarray,
        unit_vecs: np.ndarray,
        sigma: np.ndarray,
        name: str,
        ramp: Optional[Union[Ramp, int]] = None,
    ):
        """
        Initializes the GeodeticDataSet.

        Args:
            coords: (N, 3) np.ndarray of observation points (x, y, z) in local coordinates.
            data: (N,) np.ndarray of observed displacement values.
            unit_vecs: (N, 3) np.ndarray of unit vectors for projection (e.g., LOS for InSAR, E,N,U for GNSS).
            sigma: (N,) np.ndarray of data uncertainties.
            name: str identifier for the dataset.
            ramp: Optional nuisance ramp solved for alongside slip. Either a
                :class:`Ramp` or an integer degree (0 = constant offset,
                1 = planar, 2 = quadratic), in which case the normalization is
                derived from ``coords``. ``None`` (default) means no ramp.
        """
        if not (coords.shape[0] == data.shape[0] == unit_vecs.shape[0] == sigma.shape[0]):
            raise ValueError("All input arrays must have the same length (N).")
        
        if coords.shape[1] != 3:
            raise ValueError("Coordinates array must have shape (N, 3).")

        if unit_vecs.shape[1] != 3:
            raise ValueError("Unit vectors array must have shape (N, 3).")

        self.coords = coords
        self.data = data
        self.unit_vecs = unit_vecs
        self.sigma = sigma
        self.name = name
        self.ramp = Ramp.for_dataset(self, ramp) if isinstance(ramp, int) else ramp

    def get_nuisance_basis(self) -> Optional[np.ndarray]:
        """
        Returns the basis for this dataset's nuisance parameters, or None.

        For a dataset carrying a :class:`Ramp` this is the ``(N, P)`` polynomial
        design matrix evaluated at ``coords``; the inversion appends it to the
        Green's function matrix and solves for its coefficients jointly with
        slip. Datasets without a ramp return None and add no unknowns.
        """
        return None if self.ramp is None else self.ramp.basis(self.coords)

    def __len__(self) -> int:
        """Returns the number of data points (N)."""
        return self.coords.shape[0]


def nuisance_bases(
    datasets: Sequence[GeodeticDataSet],
) -> List[Optional[np.ndarray]]:
    """
    Returns each dataset's nuisance basis (or None), aligned with ``datasets``.

    This is the single place the inversion machinery asks "does this dataset add
    nuisance unknowns, and how many", so the assembler, the bounds extension and
    the result mapping can never disagree about the layout.
    """
    return [ds.get_nuisance_basis() for ds in datasets]


def nuisance_widths(bases: Sequence[Optional[np.ndarray]]) -> List[int]:
    """Returns the number of nuisance columns contributed by each dataset."""
    return [0 if b is None else int(b.shape[1]) for b in bases]
