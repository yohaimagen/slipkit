import enum
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union, Dict, List, Sequence, Optional
import meshio
from scipy.sparse import lil_matrix, csr_matrix


class SlipComponent(enum.Enum):
    """
    Identifies a slip component that a fault can invert for.

    The value maps to the cutde slip dimension index used by the physics
    engine (strike-slip -> 0, dip-slip -> 1).
    """
    STRIKE_SLIP = "strike_slip"
    DIP_SLIP = "dip_slip"

    def __str__(self):
        return self.value

    @classmethod
    def coerce(cls, value: "Union[SlipComponent, str]") -> "SlipComponent":
        """
        Returns a SlipComponent from either a SlipComponent or a string alias.

        Accepted string aliases (case-insensitive):
            strike-slip: 'strike_slip', 'strike-slip', 'ss', 'strike'
            dip-slip:    'dip_slip', 'dip-slip', 'ds', 'dip'
        """
        if isinstance(value, cls):
            return value
        key = str(value).lower()
        aliases = {
            "strike_slip": cls.STRIKE_SLIP,
            "strike-slip": cls.STRIKE_SLIP,
            "ss": cls.STRIKE_SLIP,
            "strike": cls.STRIKE_SLIP,
            "dip_slip": cls.DIP_SLIP,
            "dip-slip": cls.DIP_SLIP,
            "ds": cls.DIP_SLIP,
            "dip": cls.DIP_SLIP,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(
            f"Unknown slip component '{value}'. "
            "Use 'strike_slip'/'ss' or 'dip_slip'/'ds'."
        )


# Canonical column/row ordering for slip components. The engine and the
# regularization manager both lay components out in this order so that the
# Green's-function columns and the smoothing-matrix rows always line up.
CANONICAL_COMPONENT_ORDER: Tuple[SlipComponent, ...] = (
    SlipComponent.STRIKE_SLIP,
    SlipComponent.DIP_SLIP,
)


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
        dip_slip_type: DipSlipType = DipSlipType.UNSPECIFIED,
        slip_components: Sequence[SlipComponent] = CANONICAL_COMPONENT_ORDER,
    ):
        """
        Initializes the AbstractFaultModel.

        Args:
            strike_slip_type: The primary strike-slip component of the fault's
                motion. Only used to set the *sign* of the strike-slip Green's
                functions when strike-slip is an active component.
            dip_slip_type: The primary dip-slip component of the fault's motion.
                Only used to set the *sign* of the dip-slip Green's functions
                when dip-slip is an active component.
            slip_components: Which slip components this fault inverts for. Defaults
                to both strike-slip and dip-slip. Pass a single component (e.g.
                ``[SlipComponent.STRIKE_SLIP]``) to invert for that component only,
                which reduces the kernel/slip-vector width to ``M`` instead of
                ``2M``.
        """
        self.strike_slip_type = strike_slip_type
        self.dip_slip_type = dip_slip_type
        self._slip_components = self._validate_components(slip_components)

    @staticmethod
    def _validate_components(
        slip_components: Sequence[SlipComponent],
    ) -> Tuple[SlipComponent, ...]:
        """Validates and canonicalizes the requested slip components."""
        comps = tuple(slip_components)
        if len(comps) == 0:
            raise ValueError("slip_components must contain at least one SlipComponent.")
        for c in comps:
            if not isinstance(c, SlipComponent):
                raise ValueError(
                    f"slip_components entries must be SlipComponent, got {c!r}."
                )
        if len(set(comps)) != len(comps):
            raise ValueError("slip_components must not contain duplicate entries.")
        # Canonicalize order so the column layout is deterministic.
        return tuple(c for c in CANONICAL_COMPONENT_ORDER if c in comps)

    def active_components(self) -> Tuple[SlipComponent, ...]:
        """Returns the active slip components in canonical column order."""
        return self._slip_components

    def num_components(self) -> int:
        """Returns the number of active slip components (1 or 2)."""
        return len(self._slip_components)

    def component_slice(self, component: Union[SlipComponent, str]) -> Optional[slice]:
        """
        Returns the column/row `slice` for a component within this fault's block,
        or None if the component is not active on this fault.

        The slice indexes into this fault's local `(num_components * M)`-wide
        block (strike-slip before dip-slip, following the canonical order).
        """
        component = SlipComponent.coerce(component)
        if component not in self._slip_components:
            return None
        idx = self._slip_components.index(component)
        n = self.num_patches()
        return slice(idx * n, (idx + 1) * n)

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
        dip_slip_type: DipSlipType = DipSlipType.UNSPECIFIED,
        slip_components: Sequence[SlipComponent] = CANONICAL_COMPONENT_ORDER,
    ):
        """
        Initializes the TriangularFaultMesh from a file or raw arrays.

        Args:
            mesh_input: Path to a mesh file (e.g., .msh, .stl) or a tuple of
                        (vertices, faces) numpy arrays.
            strike_slip_type: The primary strike-slip component of the fault's motion.
            dip_slip_type: The primary dip-slip component of the fault's motion.
            slip_components: Which slip components to invert for (see
                :class:`AbstractFaultModel`). Defaults to strike-slip + dip-slip.
        """
        super().__init__(
            strike_slip_type=strike_slip_type,
            dip_slip_type=dip_slip_type,
            slip_components=slip_components,
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

    def get_areas(self) -> np.ndarray:
        """
        Calculates and returns the area of each triangular patch.
        
        Returns:
            A numpy array of shape (M,) containing the areas in square units.
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # Area = 0.5 * |(v1 - v0) x (v2 - v0)|
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return areas

    def get_mesh_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns vertices and faces."""
        return self.vertices, self.faces

    def get_centroids(self) -> np.ndarray:
        """Calculates and returns the centroids of each triangular patch."""
        return self.vertices[self.faces].mean(axis=1)

    def deep_patch_indices(self, tol: float = 2.0) -> np.ndarray:
        """
        Returns the indices of the patches lying along the fault's deep edge.

        A patch qualifies if it sits on the mesh boundary (fewer than three
        neighbours) *and* its centroid is within ``tol`` of the deepest centroid
        in the mesh. The boundary test keeps interior patches out on meshes whose
        bottom edge is curved or tapered; the depth test keeps the rest of the
        boundary (the surface trace and the lateral edges) out.

        Args:
            tol: Depth band above the deepest centroid, expressed in the mesh's
                own length units (km for local UTM-km meshes).

        Returns:
            A ``(K,)`` integer array of patch indices, possibly empty.
        """
        depth = -self.get_centroids()[:, 2]  # z is negative downwards
        on_boundary = np.array(
            [len(self.adjacency[i]) < 3 for i in range(self.num_patches())]
        )
        return np.flatnonzero(on_boundary & (depth >= depth.max() - tol))

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
