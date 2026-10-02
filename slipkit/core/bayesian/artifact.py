"""Explicit, checksummed inference bundles for Gaussian and native results."""
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import shutil
import tempfile
import numpy as np
import scipy
from .problem import AltarProblem, parameter_layout, file_hash
from .results import AltarPosterior, AltarSlipDistribution


@dataclass
class InferenceArtifact:
    problem: AltarProblem
    posterior: AltarPosterior
    faults: list
    datasets: list
    metadata: dict

    @property
    def result(self):
        return AltarSlipDistribution(self.posterior, self.faults, self.datasets)


def save_inference(path, problem, posterior, *, prior, alpha_cp=0., seed=None,
                   faults=(), datasets=(), length_unit='km', inference_settings=None):
    """Create a new bundle; refuse overwrites. Geometry is optional for matrix callers."""
    if length_unit not in ('km', 'm') or not np.isfinite(alpha_cp) or alpha_cp < 0:
        raise ValueError('Use mesh units km/m and finite nonnegative fixed Cp.')
    if problem.layout != posterior.layout or problem.G.shape[1] != posterior.samples.shape[1]:
        raise ValueError('Problem and posterior layout/width differ.')
    if faults and parameter_layout(faults, datasets) != problem.layout:
        raise ValueError('Saved geometry does not match problem layout.')
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(G=problem.G, data=problem.data, samples=posterior.samples, **problem.covariance_arrays)
    for name, value in dict(smoothing=problem.smoothing, exact_mean=posterior.exact_mean,
                             precision_factor=posterior.precision_factor).items():
        if value is not None:
            arrays[name] = value
    for name, value in posterior.probabilities.items():
        if name not in ('prior', 'likelihood', 'posterior'):
            raise ValueError('Unknown probability field.')
        arrays['probability_'+name] = value
    geometry = []
    for i, fault in enumerate(faults):
        from slipkit.core.fault import TriangularFaultMesh
        if not isinstance(fault, TriangularFaultMesh):
            raise TypeError('Durable geometry currently supports TriangularFaultMesh.')
        arrays[f'vertices_{i}'], arrays[f'faces_{i}'] = fault.get_mesh_geometry()
        geometry.append(dict(components=[c.value for c in fault.active_components()],
            strike_slip_type=fault.strike_slip_type.value, dip_slip_type=fault.dip_slip_type.value))
    observations = []
    for i, dataset in enumerate(datasets):
        for name in ('coords', 'data', 'sigma', 'unit_vecs'):
            arrays[f'dataset_{i}_{name}'] = getattr(dataset, name)
        ramp = dataset.ramp
        observations.append(dict(name=dataset.name, ramp=None if ramp is None else
            dict(degree=ramp.degree, center=ramp.center.tolist(), scale=ramp.scale)))
    # JSON round-trip enforces portable, explicit prior metadata before creating output.
    prior = json.loads(json.dumps(prior, allow_nan=False, default=lambda v: np.asarray(v).tolist()))
    meta = dict(format='slipkit-inference-v1', layout=problem.layout, prior=prior,
        alpha_cp=float(alpha_cp), cp_policy='fixed', seed=seed, length_unit=length_unit,
        final_beta=posterior.final_beta, diagnostics=posterior.diagnostics, inference_settings=inference_settings,
        faults=geometry, datasets=observations, covariance_keys=list(problem.covariance_arrays),
        beta_statistics=json.loads(posterior.beta_statistics.to_json(orient='records')),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.glob('*.py')})
    temporary = Path(tempfile.mkdtemp(prefix=path.name+'-', dir=path.parent))
    try:
        payload = temporary/'arrays.npz'
        np.savez(payload, **arrays)
        meta['arrays_sha256'] = file_hash(payload)
        (temporary/'manifest.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return str(path)


def load_inference(path):
    """Load numeric arrays without pickle, verify their hash and semantic geometry."""
    import pandas as pd
    from slipkit.core.fault import TriangularFaultMesh, SlipComponent, StrikeSlipType, DipSlipType
    from slipkit.core.data import GeodeticDataSet, Ramp
    path = Path(path).resolve()
    meta = json.loads((path/'manifest.json').read_text())
    payload = path/'arrays.npz'
    if meta.get('format') != 'slipkit-inference-v1':
        raise ValueError('Unsupported inference artifact format.')
    if file_hash(payload) != meta['arrays_sha256']:
        raise ValueError('Inference array checksum mismatch.')
    with np.load(payload, allow_pickle=False) as data:
        keys = meta['covariance_keys']
        covariance = data[keys[0]] if keys == ['covariance'] else tuple(data[k] for k in keys)
        problem = AltarProblem(data['G'], data['data'], covariance, meta['layout'],
                               data['smoothing'] if 'smoothing' in data else None)
        p = problem.G.shape[1]
        mean = data['exact_mean'] if 'exact_mean' in data else None
        factor = data['precision_factor'] if 'precision_factor' in data else None
        if (mean is None) != (factor is None):
            raise ValueError('Exact inference requires both mean and precision factor.')
        if mean is not None and (mean.shape != (p,) or factor.shape != (p, p) or
                not np.isfinite(mean).all() or not np.isfinite(factor).all() or
                np.any(np.diag(factor) <= 0) or np.any(np.tril(factor, -1) != 0)):
            raise ValueError('Invalid exact mean/precision factor.')
        if not np.isfinite(meta['final_beta']) or not 0 <= meta['final_beta'] <= 1:
            raise ValueError('Invalid saved beta.')
        probabilities = {k.removeprefix('probability_'): data[k] for k in data.files if k.startswith('probability_')}
        samples = data['samples']
        if any(v.shape != (len(samples),) or not np.isfinite(v).all() for v in probabilities.values()):
            raise ValueError('Invalid saved probabilities.')
        posterior = AltarPosterior(samples, meta['final_beta'], pd.DataFrame(meta['beta_statistics']),
            probabilities, meta['layout'], str(path), diagnostics=meta['diagnostics'],
            exact_mean=mean, precision_factor=factor)
        faults = [TriangularFaultMesh((data[f'vertices_{i}'], data[f'faces_{i}']),
            StrikeSlipType(f['strike_slip_type']), DipSlipType(f['dip_slip_type']),
            [SlipComponent(c) for c in f['components']]) for i, f in enumerate(meta['faults'])]
        datasets = [GeodeticDataSet(**{n: data[f'dataset_{i}_{n}'] for n in ('coords', 'data', 'sigma', 'unit_vecs')},
            name=d['name'], ramp=None if d['ramp'] is None else Ramp(**d['ramp']))
            for i, d in enumerate(meta['datasets'])]
    if faults and parameter_layout(faults, datasets) != problem.layout:
        raise ValueError('Saved geometry/ramp identity differs from layout.')
    return InferenceArtifact(problem, posterior, faults, datasets, meta)
