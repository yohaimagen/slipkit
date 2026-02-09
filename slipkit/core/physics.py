from abc import ABC, abstractmethod
import numpy as np
from slipkit.core.fault import AbstractFaultModel, TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
import cutde.halfspace as HS

class GreenFunctionBuilder(ABC):
    """
    Abstract base class for calculating the elastic response matrix G.
    """

    @abstractmethod
    def build_kernel(
        self, fault: AbstractFaultModel, data: GeodeticDataSet
    ) -> np.ndarray:
        """
        Returns the generic Green's function matrix G.
        """
        pass


class CutdeCpuEngine(GreenFunctionBuilder):
    """
    Green's function engine using the `cutde` library on the CPU.
    """

    def __init__(self, poisson_ratio: float = 0.25):
        """
        Initializes the CutdeCpuEngine.

        Args:
            poisson_ratio: Poisson's ratio for the elastic medium.
        """
        self.nu = poisson_ratio

    def build_kernel(
        self, fault: TriangularFaultMesh, dataset: GeodeticDataSet
    ) -> np.ndarray:
        """
        Builds the Green's function matrix G using cutde.

        This matrix maps fault slip (strike-slip and dip-slip) to
        displacements at observation points.

        Args:
            fault: A TriangularFaultMesh object.
            dataset: A GeodeticDataSet object.

        Returns:
            A numpy array of shape (N_data, 2 * M_patches) representing
            the Green's function matrix.
        """
        obs_pts = dataset.coords
        
        # cutde expects triangles as (M, 3, 3) array of vertex coordinates
        verts, faces = fault.get_mesh_geometry()
        tris = verts[faces]

        # This returns a (N, M, 3, 3) matrix mapping slip to displacement
        # (obs_idx, tri_idx, disp_dim, slip_dim)
        disp_mat = HS.disp_matrix(obs_pts=obs_pts, tris=tris, nu=self.nu)

        n_obs = obs_pts.shape[0]
        n_patches = fault.num_patches()
        
        # Initialize the final (N, 2M) Green's function matrix
        g_matrix = np.zeros((n_obs, 2 * n_patches))

        # Project displacements onto unit vectors (e.g., satellite LOS)
        # dataset.unit_vecs is (N, 3), so we need to align dimensions for broadcasting
        unit_vecs_expanded = dataset.unit_vecs[:, np.newaxis, :] # -> (N, 1, 3)

        # Response to strike-slip component (slip_dim=0)
        # disp_mat is (N, M, 3, 3). We want disp_mat[:, :, :, 0] which is (N, M, 3)
        # Sum over the displacement components (axis=2) after element-wise multiplication
        strike_slip_response = np.sum(
            disp_mat[:, :, :, 0] * unit_vecs_expanded, axis=2
        ) # -> (N, M)
        
        # Response to dip-slip component (slip_dim=1)
        dip_slip_response = np.sum(
            disp_mat[:, :, :, 1] * unit_vecs_expanded, axis=2
        ) # -> (N, M)

        # Populate the G matrix
        # Columns 0 to M-1 are for strike-slip
        g_matrix[:, :n_patches] = strike_slip_response
        # Columns M to 2M-1 are for dip-slip
        g_matrix[:, n_patches:] = dip_slip_response
        
        return g_matrix
