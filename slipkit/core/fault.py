from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Union

class AbstractFaultModel(ABC):
    """
    Abstract Base Class for any fault geometry.
    """

    @abstractmethod
    def num_patches(self) -> int:
        """
        Returns the total number of sub-faults (M).
        """
        pass

    @abstractmethod
    def get_mesh_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns vertices and faces (or equivalent geometric description).
        """
        pass

    @abstractmethod
    def get_centroids(self) -> np.ndarray:
        """
        Returns (M, 3) coordinates of patch centers (for visualization).
        """
        pass

    @abstractmethod
    def get_smoothing_matrix(self, type: str = 'laplacian') -> np.ndarray:
        """
        Returns the sparse (M, M) regularization matrix L based on topology.
        """
        pass


class TriangularFaultMesh(AbstractFaultModel):
    """
    Implementation for unstructured triangular meshes.
    """

    def __init__(
        self,
        mesh_input: Union[str, Tuple[np.ndarray, np.ndarray]]
    ):
        """
        Initializes the TriangularFaultMesh.

        Args:
            mesh_input: Path to a .msh or .stl file, or a tuple of
                        (vertices, faces) numpy arrays.
        """
        if isinstance(mesh_input, str):
            # Future: Implement mesh file parsing (.msh, .stl)
            raise NotImplementedError("File parsing for .msh or .stl is not yet implemented.")
        elif isinstance(mesh_input, tuple) and len(mesh_input) == 2:
            self.vertices: np.ndarray = mesh_input[0]
            self.faces: np.ndarray = mesh_input[1]
            if self.faces.shape[1] != 3:
                raise ValueError("Faces array must represent triangles (N, 3).")
        else:
            raise ValueError(
                "mesh_input must be a file path (str) or a tuple of (vertices, faces) arrays."
            )

        # For future use: Builds an adjacency graph of triangles.
        # For future use: Computes the Laplacian L where L_ij = -1 if j is neighbor of i, and L_ii = degree(i).

    def num_patches(self) -> int:
        """
        Returns the total number of sub-faults (M), which is the number of faces.
        """
        return self.faces.shape[0]

    def get_mesh_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns vertices and faces.
        """
        raise NotImplementedError("get_mesh_geometry is not yet implemented.")

    def get_centroids(self) -> np.ndarray:
        """
        Returns (M, 3) coordinates of patch centers (for visualization).
        """
        raise NotImplementedError("get_centroids is not yet implemented.")

    def get_smoothing_matrix(self, type: str = 'laplacian') -> np.ndarray:
        """
        Returns the sparse (M, M) regularization matrix L based on topology.
        """
        raise NotImplementedError("get_smoothing_matrix is not yet implemented.")
