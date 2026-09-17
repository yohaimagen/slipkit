from abc import ABC, abstractmethod
import numpy as np
from slipkit.core.fault import (
    AbstractFaultModel,
    TriangularFaultMesh,
    StrikeSlipType,
    DipSlipType,
    SlipComponent,
)
from slipkit.core.data import GeodeticDataSet
import cutde.halfspace as HS


# Maps each slip component to its cutde slip dimension (last axis of disp_matrix).
_COMPONENT_SLIP_DIM = {
    SlipComponent.STRIKE_SLIP: 0,
    SlipComponent.DIP_SLIP: 1,
}

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

    def predict(
        self,
        fault: AbstractFaultModel,
        data: GeodeticDataSet,
        slip: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts the observed displacement for a *known* slip distribution.

        This is the forward operation ``G @ slip`` projected onto the dataset's
        unit vectors, i.e. it returns the same ``(N,)`` values as
        ``build_kernel(fault, data) @ slip``. Because the slip is known, engines
        may implement this far more cheaply than building the full matrix (see
        :class:`CutdeCpuEngine`). This generic fallback simply builds the kernel
        and multiplies.

        Args:
            fault: The fault model.
            data: The observation dataset.
            slip: The fault's slip block, of length
                ``fault.num_components() * fault.num_patches()`` in canonical
                component order (strike-slip before dip-slip).

        Returns:
            An ``(N,)`` array of predicted (projected) displacements.
        """
        return self.build_kernel(fault, data) @ np.asarray(slip).ravel()


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
        Builds the Green's function matrix G using cutde, incorporating kinematic constraints.

        This matrix maps fault slip to displacements at observation points, with
        one column block per *active* slip component (see
        ``fault.active_components()``). The signs of each block are adjusted
        based on the fault's specified kinematic type.

        Args:
            fault: A TriangularFaultMesh object.
            dataset: A GeodeticDataSet object.

        Returns:
            A numpy array of shape (N_data, k * M_patches) where
            ``k = fault.num_components()``. Column blocks follow the canonical
            component order (strike-slip before dip-slip), each of width M, with
            signs adjusted for kinematic type.
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

        # Initialize the final (N, k*M) Green's function matrix.
        g_matrix = np.zeros((n_obs, fault.num_components() * n_patches))

        # Project displacements onto unit vectors (e.g., satellite LOS).
        # dataset.unit_vecs is (N, 3); align dimensions for broadcasting against
        # disp_mat's (N, 3, M) slices -> unit_vecs needs to be (N, 3, 1).
        unit_vecs_expanded = dataset.unit_vecs[:, :, np.newaxis]

        # Build one M-wide block per active component, in canonical order.
        for component in fault.active_components():
            slip_dim = _COMPONENT_SLIP_DIM[component]

            # disp_mat[:, :, :, slip_dim] is (N, 3, M); multiply by (N, 3, 1)
            # and sum over the displacement components (axis=1) -> (N, M).
            response = np.sum(
                disp_mat[:, :, :, slip_dim] * unit_vecs_expanded, axis=1
            )

            # Apply the sign convention for this component (only when active).
            if (
                component == SlipComponent.STRIKE_SLIP
                and fault.strike_slip_type == StrikeSlipType.LEFT_LATERAL
            ):
                response = -response
            elif (
                component == SlipComponent.DIP_SLIP
                and fault.dip_slip_type == DipSlipType.NORMAL
            ):
                response = -response

            g_matrix[:, fault.component_slice(component)] = response

        return g_matrix

    def predict(
        self,
        fault: TriangularFaultMesh,
        dataset: GeodeticDataSet,
        slip: np.ndarray,
    ) -> np.ndarray:
        """
        Matrix-free forward model: predicts LOS displacement for a known slip.

        Equivalent to ``build_kernel(fault, dataset) @ slip`` but computed with
        cutde's matrix-free ``disp_free``, so the full ``(N, k*M)`` Green's
        function matrix is never formed. This is both cheaper (one displacement
        evaluation per obs/patch pair for the *actual* slip, versus one per unit
        slip component) and O(N) in memory, which is essential at full data
        resolution where the dense kernel would be enormous.

        The kinematic sign conventions are applied identically to
        :meth:`build_kernel`: the strike-slip block is negated for
        ``LEFT_LATERAL`` faults and the dip-slip block for ``NORMAL`` faults, by
        folding those signs into the slip vector handed to ``disp_free``.

        Args:
            fault: The triangular fault mesh.
            dataset: The observation dataset (coords + projection unit vectors).
            slip: The fault's slip block, of length
                ``fault.num_components() * fault.num_patches()`` in canonical
                component order (strike-slip before dip-slip).

        Returns:
            An ``(N,)`` array of predicted displacements projected onto
            ``dataset.unit_vecs``.
        """
        slip = np.asarray(slip, dtype=float).ravel()
        n_patches = fault.num_patches()
        expected = fault.num_components() * n_patches
        if slip.shape[0] != expected:
            raise ValueError(
                f"slip length ({slip.shape[0]}) does not match this fault's "
                f"{expected} unknowns (num_components * num_patches)."
            )

        # Assemble the cutde (M, 3) slip array [strike, dip, tensile], placing
        # each active component in its cutde slot with the kinematic sign folded
        # in so the result matches build_kernel(...) @ slip exactly.
        slip_vecs = np.zeros((n_patches, 3))
        for component in fault.active_components():
            block = slip[fault.component_slice(component)]
            if (
                component == SlipComponent.STRIKE_SLIP
                and fault.strike_slip_type == StrikeSlipType.LEFT_LATERAL
            ):
                block = -block
            elif (
                component == SlipComponent.DIP_SLIP
                and fault.dip_slip_type == DipSlipType.NORMAL
            ):
                block = -block
            slip_vecs[:, _COMPONENT_SLIP_DIM[component]] = block

        verts, faces = fault.get_mesh_geometry()
        tris = verts[faces]

        # (N, 3) displacement summed over all patches, matrix-free.
        disp = HS.disp_free(dataset.coords, tris, slip_vecs, self.nu)

        # Project onto the per-point unit vectors (e.g. satellite LOS).
        return np.sum(disp * dataset.unit_vecs, axis=1)
