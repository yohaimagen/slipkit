from abc import ABC, abstractmethod
from typing import List
import warnings
from scipy.sparse import block_diag, csr_matrix
from slipkit.core.fault import AbstractFaultModel

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

        This implementation assumes independent faults and smooths strike-slip
        and dip-slip components independently.

        For V1, multi-fault stitching is not supported and will raise a warning.

        The structure for a single fault is:
            S_full = [[lambda * L, 0],
                      [0, lambda * L]]

        Args:
            faults: A list of fault models.
            lambda_spatial: The spatial smoothing weight.

        Returns:
            The global sparse smoothing matrix S.
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
            
            # Create the 2-component block diagonal matrix for this fault
            # This smooths strike and dip components independently
            l_double = block_diag([weighted_l, weighted_l], format='csr')
            all_laplacians.append(l_double)

        # Combine the matrices for all faults into one large block-diagonal matrix
        if not all_laplacians:
            return csr_matrix((0, 0))
            
        return block_diag(all_laplacians, format='csr')