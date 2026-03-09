from typing import List, Tuple, Optional
import numpy as np

from slipkit.core.inversion import AbstractAssembler
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.regularization import RegularizationManager

class BayesianAssembler(AbstractAssembler):
    """
    Assembler for Bayesian Static Inversion.
    
    This assembler handles the preparation of matrices specifically for the Bayesian framework.
    It transforms raw Strike-Slip and Dip-Slip kernels into Rake-Parallel and 
    Rake-Perpendicular kernels based on a target rake angle. 
    
    It bypasses traditional augmented-matrix Tikhonov regularization by ignoring 
    the RegularizationManager and returns raw components needed for the 
    probabilistic likelihood.
    """

    def __init__(self, target_rake_deg: float):
        """
        Initializes the BayesianAssembler.

        Args:
            target_rake_deg: The target rake angle in degrees.
        """
        self.target_rake_rad = np.radians(target_rake_deg)
        self._G_par_cache: Optional[np.ndarray] = None
        self._G_perp_cache: Optional[np.ndarray] = None
        self._d_obs_cache: Optional[np.ndarray] = None
        self._sigma_cache: Optional[np.ndarray] = None

    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles the components for the Bayesian model.
        
        Args:
            faults: List of fault models.
            datasets: List of geodetic datasets.
            engine: Physics engine for calculating Green's functions.
            regularization_manager: Ignored in this strategy.
            lambda_spatial: Ignored in this strategy.
            force_recompute: If True, clears cache and recalculates G.

        Returns:
            A tuple containing:
            - Augmented matrix A: Horizontal concatenation [G_par, G_perp].
            - Augmented vector b: Stacked [d_obs, sigma_data].
        """
        
        # 1. Compute or retrieve kernels
        if self._G_par_cache is None or force_recompute:
            self._compute_bayesian_components(faults, datasets, engine)
            
        # 2. Final Stacking
        # A = [G_par, G_perp]
        A_augmented = np.hstack([self._G_par_cache, self._G_perp_cache])
        
        # b = [d_obs, sigma_data]
        b_augmented = np.concatenate([self._d_obs_cache, self._sigma_cache])
        
        return A_augmented, b_augmented

    def _compute_bayesian_components(
        self, 
        faults: List[AbstractFaultModel], 
        datasets: List[GeodeticDataSet], 
        engine: GreenFunctionBuilder
    ):
        """
        Internal method to compute G_par, G_perp, d_obs, and sigma_data.
        """
        g_parallel_dataset_blocks = []
        g_perp_dataset_blocks = []
        d_obs_blocks = []
        sigma_blocks = []

        for dataset in datasets:
            current_dataset_par_parts = []
            current_dataset_perp_parts = []
            
            for fault in faults:
                # Calculate raw Strike-Slip and Dip-Slip kernel for this specific fault-dataset pair
                # Shape: (N_data_points, 2 * N_fault_patches)
                G_raw = engine.build_kernel(fault, dataset)
                n_patches = fault.num_patches()
                
                G_ss = G_raw[:, :n_patches]
                G_ds = G_raw[:, n_patches:]
                
                # Perform rotation based on target rake
                # G_par = G_ss * cos(rake) + G_ds * sin(rake)
                # G_perp = -G_ss * sin(rake) + G_ds * cos(rake)
                G_par = G_ss * np.cos(self.target_rake_rad) + G_ds * np.sin(self.target_rake_rad)
                G_perp = -G_ss * np.sin(self.target_rake_rad) + G_ds * np.cos(self.target_rake_rad)
                
                current_dataset_par_parts.append(G_par)
                current_dataset_perp_parts.append(G_perp)
            
            # Combine all faults for this dataset
            g_dataset_par = np.concatenate(current_dataset_par_parts, axis=1)
            g_dataset_perp = np.concatenate(current_dataset_perp_parts, axis=1)
            
            g_parallel_dataset_blocks.append(g_dataset_par)
            g_perp_dataset_blocks.append(g_dataset_perp)
            d_obs_blocks.append(dataset.data)
            sigma_blocks.append(dataset.sigma)

        # Cache the stacked components across all datasets
        self._G_par_cache = np.vstack(g_parallel_dataset_blocks)
        self._G_perp_cache = np.vstack(g_perp_dataset_blocks)
        self._d_obs_cache = np.concatenate(d_obs_blocks)
        self._sigma_cache = np.concatenate(sigma_blocks)

    def clear_cache(self):
        """Manually clears the kernel cache."""
        self._G_par_cache = None
        self._G_perp_cache = None
        self._d_obs_cache = None
        self._sigma_cache = None
