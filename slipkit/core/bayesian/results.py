"""Generic posterior records and geometry-aware public slip results."""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
from slipkit.core.inversion import SlipDistribution


@dataclass
class AltarPosterior:
    samples: np.ndarray
    final_beta: float
    beta_statistics: pd.DataFrame = field(default_factory=pd.DataFrame)
    probabilities: dict = field(default_factory=dict)
    layout: list = field(default_factory=list)
    run_path: str = None
    step_files: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=lambda: {'sampling_adequacy': 'not assessed'})

    exact_mean: np.ndarray = None
    precision_factor: np.ndarray = None

    def __post_init__(self):
        from .problem import validate_layout
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 2 or self.samples.shape[0] < 2 or self.samples.shape[1] == 0 or not np.isfinite(self.samples).all():
            raise ValueError('Posterior samples must be a finite nonempty particle-by-parameter matrix.')
        validate_layout(self.layout, self.samples.shape[1])

    @property
    def mean(self):
        return self.samples.mean(axis=0) if self.exact_mean is None else self.exact_mean

    @property
    def covariance(self):
        if self.precision_factor is not None:
            from scipy.linalg import cho_solve
            return cho_solve((self.precision_factor, False), np.eye(self.samples.shape[1]))
        return np.atleast_2d(np.cov(self.samples, rowvar=False, ddof=1))

    @property
    def std(self):
        if self.precision_factor is None:
            return self.samples.std(axis=0, ddof=1)
        from scipy.linalg import solve_triangular
        factor_solve = solve_triangular(self.precision_factor, np.eye(self.samples.shape[1]))
        return np.linalg.norm(factor_solve, axis=1)

    @property
    def annealing_complete(self):
        return bool(np.isfinite(self.final_beta) and abs(self.final_beta-1.) <= 1e-8)


def intervals(samples, probability, method):
    if not 0 < probability < 1:
        raise ValueError('Interval probability must lie strictly between zero and one.')
    if method == 'equal_tailed':
        alpha = (1-probability)/2
        return np.quantile(samples, [alpha, 1-alpha], axis=0).T
    if method == 'hdi':
        import arviz as az
        # One particle axis, never interpreted as chains x draws.
        return np.array([az.hdi(samples[:, j], hdi_prob=probability) for j in range(samples.shape[1])])
    raise ValueError("Interval method must be 'equal_tailed' or 'hdi'.")


class AltarSlipDistribution(SlipDistribution):
    def __init__(self, posterior, faults, datasets=()):
        if not posterior.annealing_complete or posterior.diagnostics.get('diagnostic_only'):
            raise ValueError('Incomplete diagnostic samples cannot construct a public slip posterior.')
        n_slip = sum(f.num_patches()*f.num_components() for f in faults)
        from slipkit.core.data import nuisance_bases, nuisance_widths
        widths = nuisance_widths(nuisance_bases(datasets))
        if posterior.samples.shape[1] != n_slip + sum(widths):
            raise ValueError('Posterior width does not match faults and nuisance parameters.')
        from .problem import parameter_layout
        expected_layout = parameter_layout(faults, datasets)
        if not posterior.layout or posterior.layout != expected_layout:
            raise ValueError('Posterior semantic layout/geometry does not match the supplied faults and ramps.')
        nuisance, self_nuisance_samples = {}, {}
        offset = n_slip
        for dataset, width in zip(datasets, widths):
            if width:
                nuisance[dataset.name] = dataset.ramp.with_coeffs(posterior.mean[offset:offset+width])
                self_nuisance_samples[dataset.name] = posterior.samples[:, offset:offset+width]
                offset += width
        super().__init__(posterior.mean[:n_slip], faults, nuisance=nuisance)
        self.posterior = posterior
        self.samples = posterior.samples[:, :n_slip]
        self.nuisance_samples = self_nuisance_samples
        self.final_beta = posterior.final_beta
        self.beta_statistics = posterior.beta_statistics
        self.step_files = posterior.step_files
        self.interval_method = 'equal_tailed'

    def get_component_samples(self, component, fault_index=0):
        fault = self.faults[fault_index]
        local = fault.component_slice(component)
        if local is None:
            raise ValueError(f'Component {component} is not active on fault {fault_index}.')
        start, _ = self._fault_offsets[fault_index]
        return self.samples[:, start+local.start:start+local.stop]

    @property
    def ss_samples(self):
        """First fault only; absent components raise ValueError."""
        return self.get_component_samples('ss')

    @property
    def ds_samples(self):
        return self.get_component_samples('ds')

    def get_mean_slip(self):
        return self.slip_vector.copy()

    def get_posterior_std(self):
        return self.posterior.std[:self._total_width]

    def get_credible_intervals(self, hdi_prob=.95, method='equal_tailed'):
        self.interval_method = method
        return intervals(self.samples, hdi_prob, method)

    def _magnitudes(self, fault_index):
        return np.sqrt(sum(self.get_component_samples(c, fault_index)**2
                           for c in self.faults[fault_index].active_components()))

    def get_slip_magnitude_stats(self, fault_index=0, probability=.95, method='equal_tailed'):
        samples = self._magnitudes(fault_index)
        ci = intervals(samples, probability, method)
        return dict(mean=samples.mean(axis=0), std=samples.std(axis=0, ddof=1),
                    lower=ci[:, 0], upper=ci[:, 1], interval_method=method)

    def seismic_moment_samples(self, shear_modulus=3.3e10, length_unit='km'):
        if length_unit not in ('km', 'm') or not np.isfinite(shear_modulus) or shear_modulus <= 0:
            raise ValueError('Use positive finite rigidity and mesh units km or m.')
        scale = 1e6 if length_unit == 'km' else 1.
        return sum(shear_modulus * self._magnitudes(i) @ (f.get_areas()*scale)
                   for i, f in enumerate(self.faults))

    @property
    def annealing_complete(self):
        return self.posterior.annealing_complete

    def is_converged(self, tolerance=1e-3):
        warnings.warn('is_converged only checks beta; use annealing_complete. Sampling adequacy is separate.',
                      DeprecationWarning, stacklevel=2)
        return abs(self.final_beta-1.) <= tolerance

    @property
    def n_beta_steps(self):
        return len(self.beta_statistics)

    def plot_annealing_convergence(self):
        import matplotlib.pyplot as plt
        df = self.beta_statistics
        if df.empty:
            return
        total = df.accepted + df.invalid + df.rejected
        _, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(df.iteration, df.beta)
        axes[0].set_ylabel('Beta')
        axes[1].plot(df.iteration, df.accepted / total.replace(0, np.nan))
        axes[1].set_ylabel('Acceptance rate')
        axes[1].set_xlabel('Annealing stage')
        plt.show()

    def plot_slip_marginals(self, patch_indices=None, bins=40, fault_index=0):
        import matplotlib.pyplot as plt
        fault = self.faults[fault_index]
        indices = list(range(min(6, fault.num_patches()))) if patch_indices is None else patch_indices
        _, axes = plt.subplots(len(indices), fault.num_components(), squeeze=False)
        for col, component in enumerate(fault.active_components()):
            samples = self.get_component_samples(component, fault_index)
            for row, index in enumerate(indices):
                axes[row, col].hist(samples[:, index], bins=bins, density=True)
                axes[row, col].set_title(f'Fault {fault_index}, patch {index}, {component}')
        plt.show()
