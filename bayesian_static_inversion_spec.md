# System Directive: Implementation of Bayesian Static Slip Inversion (Minson et al., 2013)

**Role:** You are an expert computational geophysicist and PyMC developer. 
**Task:** Implement a purely static Bayesian earthquake slip inversion module for the `SlipKit` package. This module will map the mathematical framework from Minson, Simons, and Beck (2013) into a modern PyMC probabilistic programming architecture.

## 1. Scientific Framework & Mathematical Formulation
The problem is a highly underdetermined linear inverse problem where surface displacement data ($\mathbf{d}$) is used to infer fault slip ($\mathbf{\theta}_s$) via a pre-computed Green's functions matrix ($\mathbf{G}_s$).

**The Forward Model:**
$$\mathbf{d}_{pred} = \mathbf{G}_{\parallel} \mathbf{U}_{\parallel} + \mathbf{G}_{\perp} \mathbf{U}_{\perp}$$

**The Error Model (Crucial Contribution of Minson et al.):**
Unlike traditional least-squares which assumes a fixed data covariance ($\mathbf{C}_d$), this approach introduces a dynamic *Model Prediction Error* ($\mathbf{C}_p$) that scales with the amplitude of the observed data to account for inaccuracies in the Green's functions (e.g., due to 1D velocity structure approximations).
$$\mathbf{C}_x = \mathbf{C}_d + \mathbf{C}_p$$
$$\mathbf{C}_p = \alpha^2 \text{diag}(\mathbf{d}_{obs}^2)$$
Where $\alpha$ is an unknown fractional error parameter to be estimated jointly with the slip.

## 2. PyMC Implementation Architecture

The agent must structure the PyMC model exactly according to the following probabilistic graph. 

### A. Data Inputs (Passed as parameters to the model builder)
* `G_par`: Rake-parallel Green's function matrix (shape: `N_obs` x `N_patches`).
* `G_perp`: Rake-perpendicular Green's function matrix (shape: `N_obs` x `N_patches`).
* `d_obs`: 1D array of observed static displacements.
* `Cd_var`: 1D array of the diagonal elements of the data covariance matrix $\mathbf{C}_d$ (variances of the observations).

### B. Priors (Random Variables)
1.  **Rake-Parallel Slip ($\mathbf{U}_{\parallel}$)**:
    * *Scientific constraint:* Prevent back-slip while allowing expected motion.
    * *PyMC implementation:* `pm.Uniform("U_par", lower=0.0, upper=max_expected_slip, shape=N_patches)` OR `pm.TruncatedNormal`.
2.  **Rake-Perpendicular Slip ($\mathbf{U}_{\perp}$)**:
    * *Scientific constraint:* Allow minor deviations from the assumed rake direction.
    * *PyMC implementation:* `pm.Normal("U_perp", mu=0.0, sigma=rake_tolerance, shape=N_patches)`.
3.  **Fractional Model Error ($\alpha$)**:
    * *Scientific constraint:* Must be strictly positive. The paper uses a log-normal distribution.
    * *PyMC implementation:* ```python
        ln_alpha = pm.Normal("ln_alpha", mu=np.log(0.05), sigma=1.0) # Centered on 5% error
        alpha = pm.Deterministic("alpha", pm.math.exp(ln_alpha))
        ```

### C. Deterministic Forward Model (PyTensor)
Use PyTensor (`pt`) or `pm.math` for the linear algebra to ensure the computational graph can be evaluated efficiently.
```python
d_pred = pm.math.dot(G_par, U_par) + pm.math.dot(G_perp, U_perp)