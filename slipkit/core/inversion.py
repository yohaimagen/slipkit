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
    The standard assembler for a simple, single-event inversion.

    This assembler assumes that every fault contributes to every dataset. It builds
    a single Green's function matrix relating all slip parameters to all
    data points.
    """

    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assembles the system for a standard single-event inversion.
        """
        # 1.1. Collect all data and build the elastic Green's function matrix G
        data_vector_d = []
        weighted_G_blocks = []

        for dataset in datasets:
            current_dataset_G_blocks = []
            for fault in faults:
                G_elastic_fault = engine.build_kernel(fault, dataset)
                current_dataset_G_blocks.append(G_elastic_fault)

            G_dataset = np.concatenate(current_dataset_G_blocks, axis=1)

            if dataset.sigma is None:
                raise ValueError(
                    f"Dataset '{dataset.name}' is missing uncertainty (sigma) information."
                )
            sigma_inv = 1.0 / dataset.sigma
            
            weighted_G_dataset = G_dataset * sigma_inv[:, np.newaxis]
            weighted_data_d = dataset.data * sigma_inv

            weighted_G_blocks.append(weighted_G_dataset)
            data_vector_d.append(weighted_data_d)

        G_full_weighted = np.vstack(weighted_G_blocks)
        d_full_weighted = np.concatenate(data_vector_d)

        # 1.2. Build the regularization matrix S
        S_reg = regularization_manager.build_smoothing_matrix(
            faults, lambda_spatial
        )

        # 1.3. Assemble the augmented system
        zero_reg_vector = np.zeros(S_reg.shape[0])

        A_augmented = sparse_vstack([G_full_weighted, S_reg.toarray()]).toarray()
        b_augmented = np.concatenate([d_full_weighted, zero_reg_vector])
        
        return A_augmented, b_augmented


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
