"""
ALTar slip distribution result container.

Stores the full posterior sample arrays from an ALTar static slip inversion
and provides methods for uncertainty quantification, convergence diagnostics,
and visualisation.
"""

import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slipkit.core.fault import AbstractFaultModel
from slipkit.core.inversion import SlipDistribution


class AltarSlipDistribution(SlipDistribution):
    """Posterior slip distribution produced by the ALTar CATMIP backend.

    Extends :class:`~slipkit.core.inversion.SlipDistribution` to hold the
    full sample arrays from all Markov chains at ``beta = 1``, plus the
    annealing trajectory recorded in ``BetaStatistics.txt``.

    Attributes:
        ss_samples: Strike-slip posterior samples, shape ``(N_chains, M)``.
        ds_samples: Dip-slip posterior samples, shape ``(N_chains, M)``.
        beta_statistics: DataFrame with columns
            ``[iteration, beta, scaling, accepted, invalid, rejected]``.
        final_beta: The beta value at the last sampling step (should be 1.0).
        step_files: Paths to each ``step_nnn.h5`` intermediate output file.
    """

    def __init__(
        self,
        slip_vector: np.ndarray,
        faults: List[AbstractFaultModel],
        ss_samples: np.ndarray,
        ds_samples: np.ndarray,
        beta_statistics: pd.DataFrame,
        final_beta: float,
        step_files: Optional[List[str]] = None,
    ) -> None:
        """Initialises the distribution container.

        Args:
            slip_vector: Posterior mean slip ``[mean_ss | mean_ds]``,
                shape ``(2 * M,)``.
            faults: Fault models used in the inversion.
            ss_samples: Strike-slip samples ``(N_chains, M)``.
            ds_samples: Dip-slip samples ``(N_chains, M)``.
            beta_statistics: Annealing trajectory DataFrame.
            final_beta: Final beta reached by the sampler.
            step_files: Optional list of intermediate HDF5 step file paths.
        """
        super().__init__(slip_vector, faults)
        self.ss_samples = ss_samples
        self.ds_samples = ds_samples
        self.beta_statistics = beta_statistics
        self.final_beta = final_beta
        self.step_files: List[str] = step_files or []

    # ------------------------------------------------------------------
    # Posterior statistics
    # ------------------------------------------------------------------

    def get_mean_slip(self) -> np.ndarray:
        """Returns the posterior mean slip vector.

        Returns:
            Array of shape ``(2 * M,)`` with strike-slip means followed by
            dip-slip means.
        """
        return np.concatenate(
            [self.ss_samples.mean(axis=0), self.ds_samples.mean(axis=0)]
        )

    def get_posterior_std(self) -> np.ndarray:
        """Returns the posterior standard deviation for each slip component.

        Returns:
            Array of shape ``(2 * M,)`` with strike-slip standard deviations
            followed by dip-slip standard deviations.
        """
        return np.concatenate(
            [self.ss_samples.std(axis=0), self.ds_samples.std(axis=0)]
        )

    def get_credible_intervals(self, hdi_prob: float = 0.95) -> np.ndarray:
        """Computes the highest density interval (HDI) for each slip component.

        Uses :func:`arviz.hdi` applied patch-by-patch on the stacked sample
        array.

        Args:
            hdi_prob: Probability mass to enclose (e.g., ``0.95`` for 95%).

        Returns:
            Array of shape ``(2 * M, 2)`` where column 0 is the lower bound
            and column 1 the upper bound, ordered strike-slip then dip-slip.
        """
        try:
            import arviz as az
        except ImportError:
            warnings.warn(
                "arviz is not installed; falling back to quantile-based intervals.",
                ImportWarning,
                stacklevel=2,
            )
            return self._quantile_intervals(hdi_prob)

        alpha = (1.0 - hdi_prob) / 2.0
        all_samples = np.hstack([self.ss_samples, self.ds_samples])
        hdi = az.hdi(all_samples, hdi_prob=hdi_prob)
        return hdi

    def get_slip_magnitude_stats(self) -> Dict[str, np.ndarray]:
        """Computes per-patch slip magnitude statistics across all samples.

        Slip magnitude is ``sqrt(ss^2 + ds^2)`` for each chain sample and
        each patch.

        Returns:
            A dict with keys ``"mean"``, ``"std"``, ``"hdi_lower"``,
            ``"hdi_upper"``, each an array of shape ``(M,)``.
        """
        magnitudes = np.sqrt(self.ss_samples ** 2 + self.ds_samples ** 2)
        result: Dict[str, np.ndarray] = {
            "mean": magnitudes.mean(axis=0),
            "std": magnitudes.std(axis=0),
        }
        try:
            import arviz as az
            hdi = az.hdi(magnitudes, hdi_prob=0.95)
            result["hdi_lower"] = hdi[:, 0]
            result["hdi_upper"] = hdi[:, 1]
        except ImportError:
            q_lo = np.percentile(magnitudes, 2.5, axis=0)
            q_hi = np.percentile(magnitudes, 97.5, axis=0)
            result["hdi_lower"] = q_lo
            result["hdi_upper"] = q_hi
        return result

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    def is_converged(self, tolerance: float = 1e-3) -> bool:
        """Returns True if the sampler reached beta = 1 within tolerance.

        Args:
            tolerance: Acceptable deviation from 1.0.

        Returns:
            True when ``|final_beta - 1.0| <= tolerance``.
        """
        return abs(self.final_beta - 1.0) <= tolerance

    @property
    def n_beta_steps(self) -> int:
        """Total number of beta steps recorded in ``BetaStatistics``."""
        return len(self.beta_statistics)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_annealing_convergence(self) -> None:
        """Plots the annealing trajectory and chain acceptance rate.

        Two panels are produced:
        - **Top**: Beta vs. iteration (log scale on the y-axis).
        - **Bottom**: Acceptance rate (accepted / (accepted + rejected))
          vs. iteration.
        """
        if self.beta_statistics.empty:
            warnings.warn("No BetaStatistics data available to plot.", stacklevel=2)
            return

        df = self.beta_statistics
        total = df["accepted"] + df["rejected"]
        acc_rate = np.where(total > 0, df["accepted"] / total, 0.0)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        axes[0].semilogy(df["iteration"], df["beta"], marker="o", ms=3)
        axes[0].set_ylabel("Beta")
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="beta=1")
        axes[0].legend(fontsize=8)
        axes[0].set_title("ALTar annealing convergence")

        axes[1].plot(df["iteration"], acc_rate, marker="s", ms=3, color="tab:orange")
        axes[1].set_ylabel("Acceptance rate")
        axes[1].set_xlabel("Beta step iteration")
        axes[1].axhline(0.23, color="gray", linestyle=":", linewidth=0.8, label="optimal")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_slip_marginals(
        self,
        patch_indices: Optional[List[int]] = None,
        bins: int = 40,
    ) -> None:
        """Plots 1-D posterior histograms for selected patches.

        Args:
            patch_indices: Indices of patches to plot. Defaults to the first
                min(6, M) patches.
            bins: Number of histogram bins.
        """
        m = self.ss_samples.shape[1]
        if patch_indices is None:
            patch_indices = list(range(min(6, m)))

        n = len(patch_indices)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)

        for row, idx in enumerate(patch_indices):
            axes[row, 0].hist(self.ss_samples[:, idx], bins=bins, density=True)
            axes[row, 0].set_title(f"Patch {idx} — strike-slip")
            axes[row, 0].set_xlabel("Slip (m)")

            axes[row, 1].hist(self.ds_samples[:, idx], bins=bins, density=True, color="tab:orange")
            axes[row, 1].set_title(f"Patch {idx} — dip-slip")
            axes[row, 1].set_xlabel("Slip (m)")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quantile_intervals(self, hdi_prob: float) -> np.ndarray:
        """Computes quantile-based credible intervals as fallback.

        Args:
            hdi_prob: Probability mass to enclose.

        Returns:
            Array of shape ``(2 * M, 2)``.
        """
        alpha = (1.0 - hdi_prob) / 2.0
        all_samples = np.hstack([self.ss_samples, self.ds_samples])
        lower = np.percentile(all_samples, 100 * alpha, axis=0)
        upper = np.percentile(all_samples, 100 * (1 - alpha), axis=0)
        return np.column_stack([lower, upper])
