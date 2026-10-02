"""Assemble raw kernels, covariance, ramps and Gaussian smoothing penalties."""
import numpy as np
from slipkit.core.data import nuisance_bases, nuisance_widths
from slipkit.core.inversion import AbstractAssembler
from .problem import AltarProblem, validate_covariance, parameter_layout


class AltarAssembler(AbstractAssembler):
    def __init__(self, covariance=None, noise_models=None, length_unit='km'):
        if covariance is not None and noise_models is not None:
            raise ValueError('Supply covariance or noise_models, not both.')
        if length_unit not in ('km', 'm'):
            raise ValueError("length_unit must be 'km' or 'm'.")
        self.covariance = covariance
        self.noise_models = noise_models
        self.length_unit = length_unit

    def assemble_problem(self, faults, datasets, engine, regularization_manager, lambda_spatial):
        if not faults or not datasets:
            raise ValueError('At least one fault and dataset are required.')
        if not np.isfinite(lambda_spatial) or lambda_spatial < 0:
            raise ValueError('lambda_spatial must be finite and nonnegative.')
        for dataset in datasets:
            for name in ('coords', 'unit_vecs', 'data'):
                if not np.isfinite(np.asarray(getattr(dataset, name))).all():
                    raise ValueError(f'Dataset {dataset.name} has nonfinite {name}.')
        for fault in faults:
            if hasattr(fault, 'vertices') and not np.isfinite(fault.vertices).all():
                raise ValueError('Fault geometry must be finite.')
        bases = nuisance_bases(datasets)
        widths = nuisance_widths(bases)
        if sum(widths) and len({d.name for d in datasets}) != len(datasets):
            raise ValueError('Dataset names must be unique when fitting ramps.')
        layout = parameter_layout(faults, datasets)
        n_slip = sum(f.num_patches()*f.num_components() for f in faults)
        offset = n_slip + sum(widths)
        # Recompute: mutable geometry/coordinates make identity-based caches unsafe.
        g = np.zeros((sum(len(d) for d in datasets), offset))
        row = ramp_col = 0
        for dataset, basis, width in zip(datasets, bases, widths):
            col, stop = 0, row+len(dataset)
            for fault in faults:
                part = np.asarray(engine.build_kernel(fault, dataset), dtype=float)
                size = fault.num_patches()*fault.num_components()
                if part.shape != (len(dataset), size):
                    raise ValueError('Kernel shape does not match its fault and dataset.')
                g[row:stop, col:col+size] = part
                col += size
                del part
            if width:
                g[row:stop, n_slip+ramp_col:n_slip+ramp_col+width] = basis
            ramp_col += width
            row = stop
        data = np.concatenate([d.data for d in datasets])
        if self.covariance is not None:
            covariance = self.covariance
        else:
            if self.noise_models is not None and len(self.noise_models) != len(datasets):
                raise ValueError('One NoiseModel is required per dataset.')
            blocks = []
            for i, dataset in enumerate(datasets):
                model = None if self.noise_models is None else self.noise_models[i]
                sigma = np.asarray(dataset.sigma if model is None else model.sigma(), dtype=float)
                if sigma.shape != (len(dataset),) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                    raise ValueError('sigma must be finite and strictly positive per observation.')
                c = None if model is None else model.covariance()
                if c is not None and np.shape(c) != (len(dataset), len(dataset)):
                    raise ValueError('NoiseModel covariance must match its dataset.')
                blocks.append(sigma**2 if c is None else np.asarray(c))
            covariance = (np.concatenate(blocks) if all(c.ndim == 1 for c in blocks) else
                          blocks[0] if len(blocks) == 1 else tuple(blocks))
        smoothing = None
        if lambda_spatial:
            if regularization_manager is None:
                raise ValueError('A regularization manager is required for smoothing.')
            s = regularization_manager.build_smoothing_matrix(faults, lambda_spatial).toarray()
            smoothing = np.pad(s, ((0, 0), (0, offset-n_slip)))
        return AltarProblem(g, data, covariance, layout, smoothing)

    def assemble(self, faults, datasets, engine, regularization_manager, lambda_spatial, force_recompute=False):
        """Legacy packed adapter: diagonal noise, no ramps or smoothing only."""
        problem = self.assemble_problem(faults, datasets, engine, regularization_manager, lambda_spatial)
        if problem.smoothing is not None or any(e['kind'] == 'ramp' for e in problem.layout):
            raise ValueError('Ramps/smoothing require assemble_problem and the Bayesian orchestrator.')
        if isinstance(problem.covariance, tuple):
            raise ValueError('Block covariance requires assemble_problem.')
        if problem.covariance.ndim == 2 and np.any(problem.covariance != np.diag(np.diag(problem.covariance))):
            raise ValueError('Full covariance requires assemble_problem.')
        return problem.G, np.concatenate((problem.data, np.sqrt(problem.covariance if problem.covariance.ndim == 1 else np.diag(problem.covariance))))

    def get_areas(self, faults):
        return np.concatenate([f.get_areas() for f in faults]) * (1e6 if self.length_unit == 'km' else 1.)

    def clear_cache(self):
        pass
