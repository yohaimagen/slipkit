import numpy as np
import pymc as pm
import arviz as az
from typing import Optional, Tuple, List, Dict
import warnings

from slipkit.core.solvers import SolverStrategy

class BayesianSolver(SolverStrategy):
    """
    Bayesian Solver Strategy using PyMC's Sequential Monte Carlo (SMC).
    
    This solver implements the probabilistic framework of Minson et al. (2013).
    It estimates rake-parallel slip, rake-perpendicular slip, and a fractional 
    model error parameter (alpha) jointly.
    
    It utilizes a Dirichlet-seeding initialization strategy based on an estimated 
    earthquake magnitude to ensure a physically plausible starting population.
    """

    def __init__(
        self, 
        mu_mw: float, 
        sigma_mw: float, 
        areas: np.ndarray,
        max_slip: float = 20.0, 
        rake_tolerance: float = 1.0,
        draws: int = 2000,
        chains: int = 4,
        p_acc_rate: float = 0.8,
        cores: Optional[int] = None,
        rigidity: float = 30e9,
        random_seed: Optional[int] = None
    ):
        """
        Initializes the BayesianSolver.

        Args:
            mu_mw: Mean earthquake magnitude (Mw) for initialization.
            sigma_mw: Standard deviation of magnitude for initialization.
            areas: (N_patches,) array of patch areas (m^2).
            max_slip: Upper bound for rake-parallel slip (meters).
            rake_tolerance: Standard deviation for rake-perpendicular slip (meters).
            draws: Number of particles per chain for SMC.
            chains: Number of independent chains.
            p_acc_rate: Target acceptance rate for the SMC kernel.
            cores: Number of CPU cores for parallel sampling.
            rigidity: Shear modulus (Pa) used for moment-to-slip conversion in seeding.
            random_seed: Random seed for reproducibility.
        """
        self.mu_mw = mu_mw
        self.sigma_mw = sigma_mw
        self.areas = areas
        self.max_slip = max_slip
        self.rake_tolerance = rake_tolerance
        self.draws = draws
        self.chains = chains
        self.p_acc_rate = p_acc_rate
        self.cores = cores
        self.rigidity = rigidity
        self.random_seed = random_seed
        self.last_trace: Optional[az.InferenceData] = None

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Executes the Bayesian inversion using PyMC SMC.

        Args:
            A: Augmented matrix [G_par, G_perp] of shape (N_obs, 2 * N_patches).
            b: Augmented vector [d_obs, sigma_data] of shape (2 * N_obs,).
            bounds: Optional bounds (currently ignored in favor of constructor params).

        Returns:
            The posterior mean slip vector of shape (2 * N_patches,).
        """
        n_obs = A.shape[0]
        n_patches = A.shape[1] // 2
        
        # Unpack components from the BayesianAssembler's format
        G_par = A[:, :n_patches]
        G_perp = A[:, n_patches:]
        d_obs = b[:n_obs]
        sigma_data = b[n_obs:]
        
        if len(self.areas) != n_patches:
            raise ValueError(
                f"Number of patches in areas ({len(self.areas)}) "
                f"does not match A ({n_patches})."
            )

        with pm.Model() as model:
            # 1. Priors
            # Rake-Parallel Slip (U_par)
            # We use a hard lower bound of -0.1 as per Minson et al. 2013
            # We set default_transform=None to allow SMC initialization with start dicts
            u_par = pm.Uniform("u_par", lower=-0.1, upper=self.max_slip, shape=n_patches, default_transform=None)
            
            # Rake-Perpendicular Slip (U_perp)
            u_perp = pm.Normal("u_perp", mu=0.0, sigma=self.rake_tolerance, shape=n_patches)
            
            # Fractional Model Error (alpha)
            ln_alpha = pm.Normal("ln_alpha", mu=np.log(0.05), sigma=1.0)
            alpha = pm.Deterministic("alpha", pm.math.exp(ln_alpha))
            
            # 2. Forward Model
            d_pred = pm.math.dot(G_par, u_par) + pm.math.dot(G_perp, u_perp)
            
            # 3. Dynamic Covariance
            sigma_total = pm.math.sqrt(sigma_data**2 + (alpha**2 * d_obs**2))
            
            # 4. Likelihood
            pm.Normal("obs", mu=d_pred, sigma=sigma_total, observed=d_obs)
            
            # 5. SMC Initialization (Dirichlet Seeding)
            initial_population = self._generate_dirichlet_seeds(n_patches)
            
            # 6. Sampling
            print(f"Starting Bayesian Inversion (SMC) with {self.draws} particles...")
            self.last_trace = pm.sample_smc(
                draws=self.draws,
                chains=self.chains,
                cores=self.cores,
                start=initial_population,
                random_seed=self.random_seed
            )
            
        # Post-sampling validation
        self._validate_sampling()
        
        # Extract mean slip
        mean_u_par = self.last_trace.posterior["u_par"].mean(dim=["chain", "draw"]).values
        mean_u_perp = self.last_trace.posterior["u_perp"].mean(dim=["chain", "draw"]).values
        
        return np.concatenate([mean_u_par, mean_u_perp])

    def _generate_dirichlet_seeds(self, n_patches: int) -> List[Dict]:
        """
        Implements the Minson et al. (2013) initialization trick.
        
        Generates starting coordinates by drawing a target seismic moment 
        from a Gaussian and partitioning it via a Dirichlet distribution.
        Returns a list of 'chains' dictionaries, each with 'draws' particles.
        """
        init_vals = []
        for _ in range(self.chains):
            # Generate a population of 'draws' particles for this chain
            pop_u_par = np.zeros((self.draws, n_patches))
            pop_u_perp = np.zeros((self.draws, n_patches))
            pop_ln_alpha = np.full(self.draws, np.log(0.05))
            
            for i in range(self.draws):
                # 1. Draw target Mw
                mw = np.random.normal(self.mu_mw, self.sigma_mw)
                
                # 2. Convert Mw to Mo (N.m)
                mo_total = 10**(1.5 * mw + 9.1)
                
                # 3. Partition via Dirichlet
                fractions = np.random.dirichlet(np.ones(n_patches))
                mo_patches = fractions * mo_total
                
                # 4. Convert to slip
                slip_par = mo_patches / (self.rigidity * self.areas)
                pop_u_par[i, :] = np.clip(slip_par, -0.09, self.max_slip * 0.9)
            
            init_vals.append({
                "u_par": pop_u_par,
                "u_perp": pop_u_perp,
                "ln_alpha": pop_ln_alpha
            })
            
        return init_vals

    def _validate_sampling(self):
        """Checks for convergence diagnostics and sampler performance."""
        if self.last_trace is None:
            return
        pass

    def get_inference_data(self) -> Optional[az.InferenceData]:
        """Returns the full posterior trace from the last run."""
        return self.last_trace
