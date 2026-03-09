import numpy as np
import arviz as az
from typing import List, Optional
import matplotlib.pyplot as plt

from slipkit.core.inversion import SlipDistribution
from slipkit.core.fault import AbstractFaultModel

class BayesianSlipDistribution(SlipDistribution):
    """
    Results container for Bayesian Slip Inversion.
    
    Extends SlipDistribution to store the full posterior trace (InferenceData)
    and provide methods for uncertainty quantification and diagnostics.
    """

    def __init__(
        self, 
        slip_vector: np.ndarray, 
        faults: List[AbstractFaultModel],
        inference_data: az.InferenceData
    ):
        """
        Initializes the BayesianSlipDistribution.

        Args:
            slip_vector: Posterior mean slip vector.
            faults: List of fault models used in the inversion.
            inference_data: ArviZ InferenceData object containing the posterior.
        """
        super().__init__(slip_vector, faults)
        self.inference_data = inference_data

    def get_posterior_stats(self, variable: str = "u_par") -> az.InferenceData:
        """Returns summary statistics for a given variable."""
        return az.summary(self.inference_data, var_names=[variable])

    def get_credible_intervals(self, hdi_prob: float = 0.95) -> np.ndarray:
        """
        Calculates the Highest Density Interval (HDI) for each patch.

        Returns:
            A (2 * N_patches, 2) array where [i, 0] is the lower bound 
            and [i, 1] is the upper bound for the i-th slip component.
        """
        hdi_par = az.hdi(self.inference_data, hdi_prob=hdi_prob).u_par.values
        hdi_perp = az.hdi(self.inference_data, hdi_prob=hdi_prob).u_perp.values
        
        return np.vstack([hdi_par, hdi_perp])

    def plot_diagnostics(self):
        """
        Generates standard ArviZ diagnostics (trace and rank plots).
        """
        # 1. Trace plot with rank vlines (standard for SMC)
        az.plot_trace(self.inference_data, kind="rank_vlines")
        plt.tight_layout()
        plt.show()

    def plot_alpha_posterior(self):
        """Plots the posterior distribution of the model error parameter alpha."""
        az.plot_posterior(self.inference_data, var_names=["alpha"])
        plt.show()

    def get_patch_posterior(self, patch_idx: int, component: str = "par"):
        """
        Returns the samples for a specific patch and component.
        
        Args:
            patch_idx: Index of the fault patch.
            component: 'par' for rake-parallel, 'perp' for rake-perpendicular.
        """
        var_name = "u_par" if component == "par" else "u_perp"
        return self.inference_data.posterior[var_name].sel(u_par_dim_0=patch_idx).values
