"""Plain-text inputs required by the pinned AlTar CPU linear loader."""
import os
import numpy as np
from .problem import AltarProblem, validate_covariance


class AltarDataExporter:
    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _write(self, values, filename):
        values = np.asarray(values, dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise ValueError('Exported arrays must be nonempty and finite.')
        path = os.path.join(self.output_dir, filename)
        np.savetxt(path, values)
        return path

    def export_greens_function(self, G, filename='green.txt'):
        if np.asarray(G).ndim != 2:
            raise ValueError('G must be two-dimensional.')
        return self._write(G, filename)

    def export_data(self, d_obs, filename='data.txt'):
        if np.asarray(d_obs).ndim != 1:
            raise ValueError('Data must be one-dimensional.')
        return self._write(d_obs, filename)

    def export_covariance(self, sigma=None, d_obs=None, alpha_cp=0., filename='cd.txt', *, covariance=None):
        if not np.isfinite(alpha_cp) or alpha_cp < 0:
            raise ValueError('alpha_cp must be finite and nonnegative.')
        d = np.asarray(d_obs, dtype=float)
        if covariance is None:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.ndim != 1 or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError('sigma must be finite and positive.')
            covariance = sigma**2
        c = validate_covariance(covariance, len(covariance)).copy()
        if alpha_cp:
            if d.shape != (len(c),) or not np.isfinite(d).all():
                raise ValueError('Fixed Cp requires finite observations matching covariance.')
            if c.ndim == 1:
                c += (alpha_cp*d)**2
            else:
                c[np.diag_indices(len(c))] += (alpha_cp*d)**2
        return self._write(np.diag(c) if c.ndim == 1 else c, filename)

    def export_whitened(self, G, data):
        """Whitening already validated the frozen physical covariance."""
        if np.asarray(G).ndim != 2 or np.asarray(data).shape != (np.asarray(G).shape[0],):
            raise ValueError('Whitened G/data must share the observation axis.')
        return dict(green=self.export_greens_function(G), data=self.export_data(data),
                    cd=self._write(np.eye(len(data)), 'cd.txt'))

    def export_cuda(self, G, data):
        """Binary whitened inputs for native static CUDA; no identity matrix/file."""
        import h5py
        g, d = np.asarray(G), np.asarray(data)
        if g.ndim != 2 or d.shape != (len(g),) or not g.size or not np.isfinite(g).all() or not np.isfinite(d).all():
            raise ValueError('CUDA G/data must be finite and share the observation axis.')
        paths = {}
        for key, values in dict(green=g, data=d).items():
            path = os.path.join(self.output_dir, key+'.h5')
            with h5py.File(path, 'w') as output:
                output[key] = values
            paths[key] = path
        return paths

    def export_all(self, G, d_obs, sigma=None, areas_m2=None, alpha_cp=0., *, covariance=None):
        if areas_m2 is not None:
            raise ValueError('CPU linear does not consume areas; no area file is exported.')
        if covariance is None:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.ndim != 1 or not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError('sigma must be finite and positive.')
        c = sigma**2 if covariance is None else covariance
        problem = AltarProblem(G, d_obs, c)
        return dict(green=self.export_greens_function(problem.G), data=self.export_data(problem.data),
                    cd=self.export_covariance(sigma, d_obs, alpha_cp, covariance=problem.covariance))
