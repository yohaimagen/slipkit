from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union
import warnings
import numpy as np
from scipy.sparse import block_diag, csr_matrix, vstack
from slipkit.core.fault import AbstractFaultModel, SlipComponent

class RegularizationManager(ABC):
    """
    Abstract base class for constructing regularization matrices.
    """

    @abstractmethod
    def build_smoothing_matrix(
        self, faults: List[AbstractFaultModel], lambda_spatial: float
    ) -> csr_matrix:
        """
        Builds the global sparse regularization matrix S.
        """
        pass


class LaplacianSmoothing(RegularizationManager):
    """
    Standard implementation using topological Laplacian operators.
    """

    def build_smoothing_matrix(
        self, faults: List[AbstractFaultModel], lambda_spatial: float
    ) -> csr_matrix:
        """
        Constructs the global block-diagonal smoothing matrix.

        This implementation assumes independent faults and smooths each active
        slip component independently.

        For V1, multi-fault stitching is not supported and will raise a warning.

        The structure for a single fault repeats ``lambda * L`` once per active
        component on the diagonal, e.g. for two components::

            S_full = [[lambda * L, 0],
                      [0, lambda * L]]

        and for a single-component fault it is simply ``lambda * L``.

        Args:
            faults: A list of fault models.
            lambda_spatial: The spatial smoothing weight.

        Returns:
            The global sparse smoothing matrix S. Its width per fault matches the
            fault's kernel width, ``fault.num_components() * M``.
        """
        if len(faults) > 1:
            warnings.warn(
                "Multi-fault stitching is not yet implemented. "
                "Smoothing will be applied to each fault independently.",
                UserWarning
            )

        all_laplacians = []
        for fault in faults:
            # Get the single-component Laplacian for the fault
            l_single = fault.get_smoothing_matrix(type='laplacian')

            # Apply the smoothing weight
            weighted_l = lambda_spatial * l_single

            # Repeat the Laplacian once per active component so the block width
            # matches the kernel width. Components are smoothed independently.
            l_block = block_diag(
                [weighted_l] * fault.num_components(), format='csr'
            )
            all_laplacians.append(l_block)

        # Combine the matrices for all faults into one large block-diagonal matrix
        if not all_laplacians:
            return csr_matrix((0, 0))
            
        return block_diag(all_laplacians, format='csr')


class DeepEdgeDamping(RegularizationManager):
    """
    Wraps another regularization manager and damps slip on the fault's deep edge.

    On top of the base (usually Laplacian) equations it appends one equation
    ``alpha * lambda * m_j = 0`` per deep-edge patch and per constrained slip
    component, which *pulls* slip towards zero at the bottom of the fault rather
    than forbidding it outright:

    * ``alpha = 0`` disables the constraint,
    * ``alpha`` of order a Laplacian row norm (~3.5 for an interior triangle)
      makes it about as strong as one smoothing equation,
    * large ``alpha`` (~100) approaches a hard zero.

    By default every *active* component of each fault is constrained, so a fault
    inverting for both strike-slip and dip-slip gets both damped, and a
    single-component fault gets only the component it solves for. Pass
    ``components`` to restrict it further.

    The constraint is scaled by ``lambda_spatial`` so an L-curve sweep
    (:meth:`~slipkit.core.inversion.InversionOrchestrator.run_l_curve`, which
    rescales the whole regularization block) keeps a fixed ratio between
    smoothing and deep-edge damping. ``alpha`` is therefore relative to lambda,
    not an absolute weight.
    """

    def __init__(
        self,
        base: Optional[RegularizationManager] = None,
        alpha: float = 1.0,
        tol: float = 2.0,
        components: Optional[Sequence[Union[SlipComponent, str]]] = None,
    ):
        """
        Initializes the deep-edge damping wrapper.

        Args:
            base: The regularization manager providing the smoothing block.
                Defaults to :class:`LaplacianSmoothing`.
            alpha: Damping strength, relative to ``lambda_spatial`` (see above).
            tol: Depth band defining the deep edge, in mesh length units; passed
                to ``fault.deep_patch_indices``.
            components: Which slip components to damp. ``None`` (default) damps
                every active component of every fault. Entries may be
                :class:`SlipComponent` values or string aliases ('ss'/'ds').
        """
        self.base = base if base is not None else LaplacianSmoothing()
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.components = (
            None if components is None
            else tuple(SlipComponent.coerce(c) for c in components)
        )

    def constrained_columns(self, faults: List[AbstractFaultModel]) -> np.ndarray:
        """
        Returns the global column indices damped by this constraint.

        Indices refer to the concatenated slip vector (fault blocks in order,
        components in canonical order within each block), i.e. the same layout
        as :class:`~slipkit.core.inversion.SlipDistribution`.
        """
        cols: List[int] = []
        offset = 0
        for fault in faults:
            finder = getattr(fault, "deep_patch_indices", None)
            if finder is None:
                raise TypeError(
                    f"{type(fault).__name__} does not implement "
                    "deep_patch_indices(); DeepEdgeDamping cannot locate its "
                    "deep edge."
                )
            deep = np.asarray(finder(self.tol), dtype=int)
            for component in fault.active_components():
                if self.components is not None and component not in self.components:
                    continue
                start = fault.component_slice(component).start
                cols.extend(offset + start + deep)
            offset += fault.num_components() * fault.num_patches()
        return np.asarray(cols, dtype=int)

    def build_smoothing_matrix(
        self, faults: List[AbstractFaultModel], lambda_spatial: float
    ) -> csr_matrix:
        """
        Builds the base smoothing block with the deep-edge equations appended.

        Args:
            faults: A list of fault models.
            lambda_spatial: The spatial smoothing weight.

        Returns:
            The sparse matrix ``[base ; alpha * lambda * E]``, where ``E`` selects
            the deep-edge columns. Reduces to the base block when ``alpha`` is
            zero or no patch qualifies as deep.
        """
        s_base = self.base.build_smoothing_matrix(faults, lambda_spatial)
        if self.alpha == 0.0:
            return s_base

        cols = self.constrained_columns(faults)
        if cols.size == 0:
            warnings.warn(
                f"DeepEdgeDamping selected no patches (tol={self.tol}); "
                "the constraint has no effect. Increase tol or check the mesh "
                "depth units.",
                UserWarning,
            )
            return s_base

        selector = csr_matrix(
            (np.full(cols.size, self.alpha * lambda_spatial),
             (np.arange(cols.size), cols)),
            shape=(cols.size, s_base.shape[1]),
        )
        return vstack([s_base, selector], format="csr")
