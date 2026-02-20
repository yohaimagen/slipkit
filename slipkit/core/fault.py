import enum
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union, Dict, List
import meshio
from scipy.sparse import lil_matrix, csr_matrix


class StrikeSlipType(enum.Enum):
    """
    Defines the strike-slip component of fault motion.
    """
    UNSPECIFIED = "unspecified"
    RIGHT_LATERAL = "right_lateral"
    LEFT_LATERAL = "left_lateral"

    def __str__(self):
        return self.value


class DipSlipType(enum.Enum):
    """
    Defines the dip-slip component of fault motion.
    """
    UNSPECIFIED = "unspecified"
    NORMAL = "normal"
    REVERSE = "reverse"
    THRUST = "thrust" # Alias for REVERSE motion

    def __str__(self: str):
        return self.value


class AbstractFaultModel(ABC):
    """
    Abstract Base Class for any fault geometry.
    """
    def __init__(
        self,
        strike_slip_type: StrikeSlipType = StrikeSlipType.UNSPECIFIED,
        dip_slip_type: DipSlipType = DipSlipType.UNSPECIFIED
    ):
        """
        Initializes the AbstractFaultModel with optional strike-slip and dip-slip types.

        Args:
            strike_slip_type: The primary strike-slip component of the fault's motion.
            dip_slip_type: The primary dip-slip component of the fault's motion.
        """
        self.strike_slip_type = strike_slip_type
        self.dip_slip_type = dip_slip_type

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
    def get_smoothing_matrix(self, type: str = 'laplacian') -> csr_matrix:
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
        mesh_input: Union[str, Tuple[np.ndarray, np.ndarray]],
        strike_slip_type: StrikeSlipType = StrikeSlipType.UNSPECIFIED,
        dip_slip_type: DipSlipType = DipSlipType.UNSPECIFIED
    ):
        """
        Initializes the TriangularFaultMesh from a file or raw arrays.

        Args:
            mesh_input: Path to a mesh file (e.g., .msh, .stl) or a tuple of
                        (vertices, faces) numpy arrays.
            strike_slip_type: The primary strike-slip component of the fault's motion.
            dip_slip_type: The primary dip-slip component of the fault's motion.
        """
        super().__init__(
            strike_slip_type=strike_slip_type,
            dip_slip_type=dip_slip_type
        )

        if isinstance(mesh_input, str):
            mesh = meshio.read(mesh_input)
            self.vertices: np.ndarray = mesh.points
            # Find the triangle cells
            triangle_cells = None
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    triangle_cells = cell_block.data
                    break
            if triangle_cells is None:
                raise ValueError("No triangular faces found in the mesh file.")
            self.faces: np.ndarray = triangle_cells
        elif isinstance(mesh_input, tuple) and len(mesh_input) == 2:
            self.vertices: np.ndarray = mesh_input[0]
            self.faces: np.ndarray = mesh_input[1]
            if self.faces.shape[1] != 3:
                raise ValueError("Faces array must represent triangles (N, 3).")
        else:
            raise ValueError(
                "mesh_input must be a file path (str) or a tuple of (vertices, faces) arrays."
            )
        
        self._build_adjacency()

    def _build_adjacency(self):
        """Builds an adjacency graph of the mesh triangles."""
        self.adjacency: Dict[int, List[int]] = {i: [] for i in range(self.num_patches())}
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}

        for i, face in enumerate(self.faces):
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0]))),
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(i)

        for edge, faces_indices in edge_to_faces.items():
            if len(faces_indices) == 2:
                f1, f2 = faces_indices
                self.adjacency[f1].append(f2)
                self.adjacency[f2].append(f1)
    
    def num_patches(self) -> int:
        """Returns the total number of sub-faults (M)."""
        return self.faces.shape[0]

    def get_mesh_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns vertices and faces."""
        return self.vertices, self.faces

    def get_centroids(self) -> np.ndarray:
        """Calculates and returns the centroids of each triangular patch."""
        return self.vertices[self.faces].mean(axis=1)

    def get_smoothing_matrix(self, type: str = 'laplacian') -> csr_matrix:
        """
        Constructs and returns the sparse Laplacian smoothing matrix.

        The Laplacian L is defined as:
        L_ii = degree of patch i (number of neighbors)
        L_ij = -1 if patches i and j are adjacent
        L_ij = 0 otherwise

        Returns:
            A sparse (M, M) CSR matrix representing the Laplacian operator.
        """
        if type != 'laplacian':
            raise NotImplementedError(f"Smoothing type '{type}' is not supported.")

        n_patches = self.num_patches()
        laplacian = lil_matrix((n_patches, n_patches))

        for i in range(n_patches):
            neighbors = self.adjacency[i]
            laplacian[i, i] = len(neighbors)
            for j in neighbors:
                laplacian[i, j] = -1.0
        
        return laplacian.tocsr()
