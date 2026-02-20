# slipkit/utils/visualizers/l_curve.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from typing import Tuple, Optional

class LCurveVisualizer:
    """
    A visualizer for plotting L-curves and finding the optimal regularization parameter
    using linear scaling with internal normalization.
    """

    @staticmethod
    def find_corner(
        lambdas: np.ndarray,
        misfits: np.ndarray,
        roughnesses: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Finds the corner of the L-curve using maximum curvature on normalized data.
        
        We normalize the data to [0, 1] before calculating curvature to prevent 
        axis scaling (e.g., small misfit vs large roughness) from biasing the result.
        """
        
        # 1. Sort by misfit (essential for spline fitting)
        # We assume the curve is roughly monotonic: as misfit increases, roughness decreases.
        sort_indices = np.argsort(misfits)
        m_sorted = misfits[sort_indices]
        r_sorted = roughnesses[sort_indices]
        l_sorted = lambdas[sort_indices]

        # 2. Normalize data to [0, 1] range for calculation
        # This is the critical fix for linear plots.
        m_min, m_max = m_sorted.min(), m_sorted.max()
        r_min, r_max = r_sorted.min(), r_sorted.max()

        # Avoid division by zero if data is constant
        if m_max == m_min or r_max == r_min:
             # Fallback to the middle point if curve is flat
            mid_idx = len(l_sorted) // 2
            return l_sorted[mid_idx], m_sorted[mid_idx], r_sorted[mid_idx]

        m_norm = (m_sorted - m_min) / (m_max - m_min)
        r_norm = (r_sorted - r_min) / (r_max - r_min)

        # 3. Fit spline to NORMALIZED data
        # k=3 is cubic, s=0 implies interpolation (passes through all points).
        # Use a small 's' if data is very noisy.
        tck = splrep(m_norm, r_norm, k=3, s=0)

        # 4. Calculate curvature on fine grid
        t = np.linspace(0, 1, 1000)
        
        # Derivatives of the normalized curve
        dy = splev(t, tck, der=1)
        ddy = splev(t, tck, der=2)

        # Standard curvature formula: k = |y''| / (1 + y'^2)^1.5
        curvature = np.abs(ddy) / np.power(1 + np.square(dy), 1.5)
        
        # 5. Find index of max curvature
        corner_idx_fine = np.argmax(curvature)
        
        # 6. Map back to original data
        # Find which original point is closest to the fine-grid corner
        # We compare the normalized misfit values
        closest_idx = np.abs(m_norm - t[corner_idx_fine]).argmin()

        optimal_lambda = l_sorted[closest_idx]
        corner_misfit = m_sorted[closest_idx]
        corner_roughness = r_sorted[closest_idx]

        return optimal_lambda, corner_misfit, corner_roughness

    @staticmethod
    def plot_l_curve(
        lambdas: np.ndarray,
        misfits: np.ndarray,
        roughnesses: np.ndarray,
        ax: Optional[plt.Axes] = None,
        plot_corner: bool = True,
        **kwargs
    ):
        """
        Plots the L-curve in Linear scale.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot raw data (Linear Scale)
        sc = ax.scatter(misfits, roughnesses, c=np.log10(lambdas), cmap='viridis', **kwargs)
        
        # Optional: Add a line connecting the dots to visualize the curve order
        # sort for plotting the line so it doesn't zigzag
        sort_idx = np.argsort(misfits)
        ax.plot(misfits[sort_idx], roughnesses[sort_idx], 'k--', alpha=0.3)

        ax.set_xlabel("Misfit (Residual Norm)")
        ax.set_ylabel("Roughness (Model Norm)")
        ax.set_title("L-Curve (Linear Scale)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Log10(Lambda)")

        if plot_corner:
            optimal_lambda, corner_misfit, corner_roughness = LCurveVisualizer.find_corner(
                lambdas, misfits, roughnesses
            )
            ax.plot(
                corner_misfit, 
                corner_roughness, 
                'r*', 
                markersize=15, 
                label=rf'Corner ($\lambda$={optimal_lambda:.2e})'
            )
            ax.legend()
        
        if 'fig' in locals():
            plt.show()