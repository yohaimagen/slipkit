"""
This module provides strategies for solving the inverse problem.

It defines an abstract base class `SolverStrategy` and concrete implementations
like `NnlsSolver` for Non-Negative Least Squares and `BoundedLsqSolver`
for bounded least squares problems using `scipy.optimize`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import nnls, lsq_linear


class SolverStrategy(ABC):
    """
    Abstract base class for numerical solvers.

    Defines the interface for any solver strategy used in the inversion process.
    Concrete implementations must provide a `solve` method.
    """

    @abstractmethod
    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Solves the linear system Ax = b.

        Args:
            A: The coefficient matrix of shape (M, N).
            b: The dependent variable vector of shape (M,).
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution vector.
                    Each array should be of shape (N,).

        Returns:
            The solution vector x of shape (N,).
        """
        pass


class NnlsSolver(SolverStrategy):
    """
    Solver strategy using Non-Negative Least Squares.

    This solver is suitable for problems where the solution (slip) is
    expected to be non-negative (e.g., pure thrust or strike-slip in a known direction).
    It wraps `scipy.optimize.nnls`.
    """

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Solves the linear system Ax = b using non-negative least squares.

        Note:
            `nnls` intrinsically enforces non-negativity. Any `bounds` provided
            will be ignored by this specific solver implementation as `nnls`
            only supports non-negative constraints.

        Args:
            A: The coefficient matrix of shape (M, N).
            b: The dependent variable vector of shape (M,).
            bounds: Optional tuple of (lower_bounds, upper_bounds).
                    These bounds are ignored by `scipy.optimize.nnls` as it
                    only supports non-negative constraints.

        Returns:
            The solution vector x of shape (N,).
        """
        x, _ = nnls(A, b)
        return x


class BoundedLsqSolver(SolverStrategy):
    """
    Solver strategy using bounded least squares.

    This solver allows for general lower and upper bounds on the solution
    variables, making it suitable for mixed-mode slip or when specific
    physical constraints are known. It wraps `scipy.optimize.lsq_linear`.
    """

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Solves the linear system Ax = b using bounded least squares.

        Args:
            A: The coefficient matrix of shape (M, N).
            b: The dependent variable vector of shape (M,).
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the solution vector.
                    If None, no bounds are applied. Each array should be of shape (N,).

        Returns:
            The solution vector x of shape (N,).

        Raises:
            ValueError: If bounds are provided but are not a tuple of two arrays.
        """
        if bounds is None:
            # lsq_linear expects (lower_bounds, upper_bounds) as a tuple
            # If no bounds, pass (-inf, inf) implicitly by not providing them.
            res = lsq_linear(A, b)
        else:
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    "Bounds must be a tuple of (lower_bounds, upper_bounds)."
                )
            res = lsq_linear(A, b, bounds=bounds)

        return res.x
