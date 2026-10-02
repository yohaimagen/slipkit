"""Frozen physical inputs and shared Gaussian linear algebra."""
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import numpy as np
from scipy.linalg import cho_solve, solve_triangular


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def parameter_layout(faults, datasets=()):
    """Semantic column identity, including geometry and normalized ramp bases."""
    from slipkit.core.data import nuisance_bases, nuisance_widths
    layout, offset = [], 0
    for index, fault in enumerate(faults):
        geometry = None
        if hasattr(fault, 'vertices') and hasattr(fault, 'faces'):
            geometry = array_hash(np.asarray(fault.vertices, dtype='<f8'), np.asarray(fault.faces, dtype='<i8'))
        for component in fault.active_components():
            width = fault.num_patches()
            entry = dict(kind='slip', fault_index=index, component=component.value,
                         start=offset, stop=offset+width, units='m', geometry_sha256=geometry,
                         sign_convention=dict(strike_slip=str(fault.strike_slip_type), dip_slip=str(fault.dip_slip_type)))
            layout.append(entry)
            offset += width
    bases = nuisance_bases(datasets)
    for index, (dataset, basis, width) in enumerate(zip(datasets, bases, nuisance_widths(bases))):
        if width:
            layout.append(dict(kind='ramp', dataset_index=index, name=dataset.name,
                               start=offset, stop=offset+width, units='m',
                               dataset_sha256=array_hash(np.asarray(dataset.coords, dtype='<f8'), np.asarray(dataset.unit_vecs, dtype='<f8')),
                               basis_sha256=array_hash(np.asarray(basis, dtype='<f8')),
                               ramp=dict(degree=dataset.ramp.degree, center=dataset.ramp.center.tolist(), scale=dataset.ramp.scale)))
            offset += width
    return layout


def validate_covariance(covariance, n, *, return_factor=False):
    c = np.array(covariance, dtype=float, copy=True)
    if c.shape not in ((n,), (n, n)) or not np.isfinite(c).all():
        raise ValueError('Covariance must be finite variances (N_obs,) or a matrix (N_obs,N_obs).')
    if c.ndim == 2 and np.count_nonzero(c) == np.count_nonzero(np.diag(c)):
        c = np.diag(c).copy()
    if c.ndim == 1:
        if np.any(c <= 0):
            raise ValueError('Variances must be strictly positive.')
        factor = np.sqrt(c)
    else:
        if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
            raise ValueError('Covariance must be symmetric.')
        try:
            factor = np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Covariance must be positive definite.') from exc
    c.setflags(write=False)
    factor.setflags(write=False)
    return (c, factor) if return_factor else c


def validate_layout(layout, p):
    if not layout:
        return
    cursor = 0
    for entry in layout:
        if entry['start'] != cursor or entry['stop'] <= cursor:
            raise ValueError('Layout must cover parameters contiguously in column order.')
        cursor = entry['stop']
    if cursor != p:
        raise ValueError('Layout width does not match parameter count.')


@dataclass(frozen=True)
class AltarProblem:
    G: np.ndarray
    data: np.ndarray
    covariance: np.ndarray
    layout: list = field(default_factory=list)
    smoothing: np.ndarray = None

    def __post_init__(self):
        g, d = np.array(self.G, dtype=float, copy=True), np.array(self.data, dtype=float, copy=True)
        if g.ndim != 2 or min(g.shape) == 0 or not np.isfinite(g).all():
            raise ValueError('G must be a finite nonempty observation-by-parameter matrix.')
        n, p = g.shape
        if d.shape != (n,) or not np.isfinite(d).all():
            raise ValueError('Data must be a finite vector with N_obs entries.')
        if isinstance(self.covariance, tuple) and all(isinstance(c, np.ndarray) for c in self.covariance):
            if sum(len(c) for c in self.covariance) != n:
                raise ValueError('Covariance blocks must cover all observations.')
            validated = [validate_covariance(c, len(c), return_factor=True) for c in self.covariance]
            covariance, factor = tuple(c for c, _ in validated), tuple(f for _, f in validated)
        else:
            covariance, factor = validate_covariance(self.covariance, n, return_factor=True)
        validate_layout(self.layout, p)
        smoothing = None
        if self.smoothing is not None:
            smoothing = np.array(self.smoothing, dtype=float, copy=True)
            if smoothing.ndim != 2 or smoothing.shape[1] != p or not np.isfinite(smoothing).all():
                raise ValueError('Smoothing must be finite with N_param columns.')
            smoothing.setflags(write=False)
        g.setflags(write=False); d.setflags(write=False)
        for name, value in dict(G=g, data=d, covariance=covariance, smoothing=smoothing,
                                layout=deepcopy(self.layout), _noise_factor=factor).items():
            object.__setattr__(self, name, value)

    def whiten(self, alpha_cp=0.):
        """Ceff includes fixed Cp; return L^-1 G, L^-1 d and log(det Ceff)."""
        if not np.isfinite(alpha_cp) or alpha_cp < 0:
            raise ValueError('alpha_cp must be finite and nonnegative.')
        covariances = self.covariance if isinstance(self.covariance, tuple) else (self.covariance,)
        factors = self._noise_factor if isinstance(self._noise_factor, tuple) else (self._noise_factor,)
        row, logdet = 0, 0.
        gw, dw = np.empty_like(self.G), np.empty_like(self.data)
        for covariance, factor in zip(covariances, factors):
            stop = row+len(covariance)
            g, d = self.G[row:stop], self.data[row:stop]
            if alpha_cp:
                c = covariance.copy()
                if c.ndim == 1:
                    c += (alpha_cp*d)**2
                else:
                    c[np.diag_indices(len(c))] += (alpha_cp*d)**2
                _, factor = validate_covariance(c, len(c), return_factor=True)
            if factor.ndim == 1:
                np.divide(g, factor[:, None], out=gw[row:stop])
                np.divide(d, factor, out=dw[row:stop])
                logdet += 2*np.log(factor).sum()
            else:
                whitened = solve_triangular(factor, np.column_stack((g, d)), lower=True)
                gw[row:stop], dw[row:stop] = whitened[:, :-1], whitened[:, -1]
                logdet += 2*np.log(np.diag(factor)).sum()
            row = stop
        return gw, dw, float(logdet)

    @property
    def covariance_bytes(self):
        return sum(c.nbytes for c in self.covariance) if isinstance(self.covariance, tuple) else self.covariance.nbytes

    @property
    def covariance_arrays(self):
        if isinstance(self.covariance, tuple):
            return {f'covariance_{i}': c for i, c in enumerate(self.covariance)}
        return {'covariance': self.covariance}

    def gaussian_factor(self, mean=0., scales=.5, alpha_cp=0., *, beta=1.):
        """Exact posterior mean and upper precision factor, without forming covariance."""
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('beta must lie in [0,1].')
        p = self.G.shape[1]
        scales = np.broadcast_to(np.asarray(scales, dtype=float), (p,))
        mean = np.broadcast_to(np.asarray(mean, dtype=float), (p,))
        if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(mean).all():
            raise ValueError('Prior means must be finite and scales finite and positive.')
        g, d, _ = self.whiten(alpha_cp)
        q = beta*(g.T @ g)
        q[np.diag_indices(p)] += scales**-2
        if self.smoothing is not None:
            q += self.smoothing.T @ self.smoothing
        r = np.linalg.cholesky(q).T
        return cho_solve((r, False), scales**-2*mean + beta*(g.T @ d)), r

    def gaussian_reference(self, mean=0., scales=.5, alpha_cp=0., *, beta=1.):
        mean, r = self.gaussian_factor(mean, scales, alpha_cp, beta=beta)
        return mean, cho_solve((r, False), np.eye(len(mean)))
