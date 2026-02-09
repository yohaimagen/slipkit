import numpy as np
from typing import Optional

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
    ):
        """
        Initializes the GeodeticDataSet.

        Args:
            coords: (N, 3) np.ndarray of observation points (x, y, z) in local coordinates.
            data: (N,) np.ndarray of observed displacement values.
            unit_vecs: (N, 3) np.ndarray of unit vectors for projection (e.g., LOS for InSAR, E,N,U for GNSS).
            sigma: (N,) np.ndarray of data uncertainties.
            name: str identifier for the dataset.
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

    def get_nuisance_basis(self) -> Optional[np.ndarray]:
        """
        Returns a basis for nuisance parameters (e.g., orbital ramps).
        
        For V1, this is a placeholder and returns None.
        """
        return None

    def __len__(self) -> int:
        """Returns the number of data points (N)."""
        return self.coords.shape[0]
