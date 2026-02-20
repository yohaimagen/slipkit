"""
This module contains the InversionOrchestrator, which is the main
user-facing API for setting up and running a slip inversion.
"""

from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import vstack as sparse_vstack

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.solvers import SolverStrategy
from slipkit.core.regularization import RegularizationManager, LaplacianSmoothing


class SlipDistribution:
    """
    Placeholder class for storing the results of a slip inversion.

    In future iterations, this will likely hold the slip vector, mapped
    back to fault geometry, and other derived quantities.
    """

    def __init__(self, slip_vector: np.ndarray, faults: List[AbstractFaultModel]):
        self.slip_vector = slip_vector
        self.faults = faults
        # TODO: Implement mapping of slip_vector back to individual fault patches


class AbstractAssembler(ABC):
    """
    Abstract base class for strategies that assemble the linear system.

    This allows for flexible construction of the augmented matrix `A` and
    data vector `b` to accommodate complex inversion scenarios, such as
    multiple disjoint events or time-dependent problems.
    """

    @abstractmethod
    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles the augmented linear system (A, b).

        Args:
            faults: A list of fault models in the inversion.
            datasets: A list of geodetic datasets.
            engine: The Green's function calculation engine.
            regularization_manager: The manager for building smoothing matrices.
            lambda_spatial: The spatial regularization weight.

        Returns:
            A tuple containing the augmented matrix `A` and the data vector `b`.
        """
        pass


class VanillaAssembler(AbstractAssembler):
    """
    The standard assembler for a simple, single-event inversion with caching.
    
    It calculates the elastic Green's function matrix G once and caches it.
    Subsequent calls to assemble() reuse this matrix unless a rebuild is forced.
    """

    def __init__(self):
        # Cache storage
        self._G_elastic_cache: Optional[List[np.ndarray]] = None
        self._data_vector_cache: Optional[np.ndarray] = None
        self._sigma_inv_cache: Optional[List[np.ndarray]] = None

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
        Assembles the system (A, b), reusing cached Green's functions if available.

        Args:
            faults: List of fault models.
            datasets: List of geodetic datasets.
            engine: Physics engine.
            regularization_manager: Smoothing manager.
            lambda_spatial: Regularization weight.
            force_recompute: If True, clears cache and recalculates G.
        """
        
        # 1. Check if we need to compute the Elastic Kernels (G)
        if self._G_elastic_cache is None or force_recompute:
            self._compute_elastic_kernels(faults, datasets, engine)

        # 2. Apply Data Weights (Sigma)
        # We do this every time because sigma might theoretically change (though rare),
        # but the heavy lifting (calculating G) is already done.
        weighted_G_blocks = []
        weighted_data_blocks = []

        for i, dataset in enumerate(datasets):
            # Retrieve cached raw G for this dataset
            G_raw = self._G_elastic_cache[i]
            
            if dataset.sigma is None:
                raise ValueError(f"Dataset '{dataset.name}' missing sigma.")
            
            # Inverse covariance (weighting)
            # Flatten sigma if it's 1D, or handle full covariance if supported later
            sigma_inv = 1.0 / dataset.sigma
            
            # Apply weighting: W * G
            # Broadcasting: (N_data, 1) * (N_data, N_param)
            G_weighted = G_raw * sigma_inv[:, np.newaxis]
            d_weighted = dataset.data * sigma_inv

            weighted_G_blocks.append(G_weighted)
            weighted_data_blocks.append(d_weighted)

        # Stack the weighted blocks
        G_full_weighted = np.vstack(weighted_G_blocks)
        d_full_weighted = np.concatenate(weighted_data_blocks)

        # 3. Build Regularization Matrix (S)
        # This is fast relative to G, so we usually rebuild it to allow changing lambda
        S_reg = regularization_manager.build_smoothing_matrix(faults, lambda_spatial)
        
        # 4. Assemble Final System
        # Regularization targets are usually zero (smoothness)
        zero_reg_vector = np.zeros(S_reg.shape[0])

        # Combine Data equations and Regularization equations
        # G_total = [ G_weighted ]
        #           [ S_reg      ]
        A_augmented = sparse_vstack([G_full_weighted, S_reg]).toarray()
        
        # b_total = [ d_weighted ]
        #           [ 0          ]
        b_augmented = np.concatenate([d_full_weighted, zero_reg_vector])

        return A_augmented, b_augmented

    def _compute_elastic_kernels(
        self, 
        faults: List[AbstractFaultModel], 
        datasets: List[GeodeticDataSet], 
        engine: GreenFunctionBuilder
    ):
        """
        Internal method to compute raw G matrices and store them in cache.
        This is the expensive step.
        """
        print("Computing elastic Green's functions (kernels)...")
        self._G_elastic_cache = []
        
        for dataset in datasets:
            # For a single dataset, G is [G_fault1 | G_fault2 | ...]
            current_dataset_G_parts = []
            
            for fault in faults:
                # Calculate G for this specific fault-dataset pair
                # Shape: (N_data_points, N_fault_patches * components)
                G_part = engine.build_kernel(fault, dataset)
                current_dataset_G_parts.append(G_part)
            
            # Concatenate horizontally to get G for this dataset across all faults
            G_dataset_full = np.concatenate(current_dataset_G_parts, axis=1)
            self._G_elastic_cache.append(G_dataset_full)
            
        print("Green's functions computed and cached.")

    def clear_cache(self):
        """Manually clears the kernel cache."""
        self._G_elastic_cache = None


class InversionOrchestrator:
    """
    The user-facing API that orchestrates the slip inversion process.

    This class ties together data, fault models, physics engines, and solvers
    to construct and solve the linear inverse problem. It uses a configurable
    `Assembler` strategy to build the linear system, allowing for flexibility.
    """

    def __init__(self):
        self.faults: List[AbstractFaultModel] = []
        self.datasets: List[GeodeticDataSet] = []
        self.engine: Optional[GreenFunctionBuilder] = None
        self.solver: Optional[SolverStrategy] = None
        self._regularization_manager: RegularizationManager = LaplacianSmoothing()
        self.assembler: AbstractAssembler = VanillaAssembler()  # Default assembler

    def add_fault(self, fault: AbstractFaultModel):
        """Adds a fault model to the inversion."""
        self.faults.append(fault)

    def add_data(self, dataset: GeodeticDataSet):
        """Adds a geodetic dataset to the inversion."""
        self.datasets.append(dataset)

    def set_engine(self, engine: GreenFunctionBuilder):
        """Sets the Green's function engine."""
        self.engine = engine

    def set_solver(self, solver: SolverStrategy):
        """Sets the solver strategy."""
        self.solver = solver
        
    def set_assembler(self, assembler: AbstractAssembler):
        """
        Sets the assembly strategy for constructing the linear system.

        This allows the user to define custom logic for how the Green's function
        matrix and data vectors are constructed, for example in multi-event
        or time-dependent inversions.

        Args:
            assembler: An instance of a class implementing AbstractAssembler.
        """
        self.assembler = assembler

    def run_inversion(
        self, lambda_spatial: float, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> SlipDistribution:
        """
        Executes the slip inversion.

        This method orchestrates the assembly of the observation equations,
        regularization, and calls the specified solver.

        Args:
            lambda_spatial: The weighting parameter for spatial regularization.
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution vector.

        Returns:
            A SlipDistribution object containing the inverted slip vector.
        """
        if not self.faults:
            raise ValueError("No fault models added to the inversion.")
        if not self.datasets:
            raise ValueError("No geodetic datasets added to the inversion.")
        if self.engine is None:
            raise ValueError("No GreenFunctionBuilder engine has been set.")
        if self.solver is None:
            raise ValueError("No SolverStrategy has been set.")

        # --- 1. Assembly Phase (Delegated to strategy) ---
        A_augmented, b_augmented = self.assembler.assemble(
            self.faults,
            self.datasets,
            self.engine,
            self._regularization_manager,
            lambda_spatial,
        )

        # --- 2. Solve Phase ---
        solution_vector_m = self.solver.solve(A_augmented, b_augmented, bounds)

        # --- 3. Map Phase ---
        return SlipDistribution(solution_vector_m, self.faults)

    def run_l_curve(
        self,
        lambdas: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the inversion for a range of lambda values to generate an L-curve.

        This method calculates the misfit (solution norm) and roughness (seminorm)
        for each provided regularization parameter.

        Args:
            lambdas: An array of spatial regularization parameters to test.
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution.
            force_recompute: If True, forces re-computation of the elastic kernels.

        Returns:
            A tuple containing:
            - The array of lambdas used.
            - The corresponding misfit array (rho).
            - The corresponding roughness array (eta).
        """
        if not self.faults:
            raise ValueError("No fault models added to the inversion.")
        if not self.datasets:
            raise ValueError("No geodetic datasets added to the inversion.")
        if self.engine is None:
            raise ValueError("No GreenFunctionBuilder engine has been set.")
        if self.solver is None:
            raise ValueError("No SolverStrategy has been set.")

        misfits = []
        roughnesses = []

        # Ensure G is computed and cached before the loop
        if force_recompute or self.assembler._G_elastic_cache is None:
            self.assembler._compute_elastic_kernels(self.faults, self.datasets, self.engine)

        num_data_points = sum(len(ds.data) for ds in self.datasets)

        for lambda_val in lambdas:
            # 1. Assemble the system for the current lambda
            A_aug, b_aug = self.assembler.assemble(
                self.faults,
                self.datasets,
                self.engine,
                self._regularization_manager,
                lambda_val,
                force_recompute=False,  # Use cached G
            )

            # 2. Solve for the slip vector m
            m = self.solver.solve(A_aug, b_aug, bounds)

            # 3. Calculate misfit and roughness
            G_weighted = A_aug[:num_data_points, :]
            d_weighted = b_aug[:num_data_points]
            
            # The regularization part of the matrix
            S_reg = A_aug[num_data_points:, :]

            misfit = np.linalg.norm(G_weighted @ m - d_weighted)
            
            # Roughness is ||L*m||, but S_reg = lambda * L
            # So, ||L*m|| = ||S_reg*m|| / lambda
            if lambda_val > 0:
                roughness = np.linalg.norm(S_reg @ m) / lambda_val
            else:
                # If lambda is zero, roughness is not well-defined in this context
                # but can be considered as ||L*m|| where L is part of the matrix
                # For simplicity, we can get L from the regularization manager
                L_matrix = self._regularization_manager.build_smoothing_matrix(self.faults, 1.0)
                roughness = np.linalg.norm(L_matrix @ m)


            misfits.append(misfit)
            roughnesses.append(roughness)

        return np.array(lambdas), np.array(misfits), np.array(roughnesses)
