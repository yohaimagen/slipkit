"""Strict final-state import without inventing geometry or parameter order."""
import os
import hashlib
import json
import warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from .results import AltarPosterior
from .problem import file_hash


class AltarResultImporter:
    def load(self, results_dir, n_parameters=None, *, manifest=None, allow_incomplete=False, stage_file=None):
        run_path = os.path.dirname(os.path.abspath(results_dir))
        if manifest is None:
            for parent in Path(results_dir).resolve().parents:
                path = parent/'manifest.json'
                if path.is_file():
                    manifest = path
                    break
        if isinstance(manifest, (str, os.PathLike)):
            run_path = str(Path(manifest).resolve().parent)
            with open(manifest) as stream:
                manifest = json.load(stream)
        manifest = manifest or {}
        if manifest.get('output_dir'):
            parts = Path(manifest['output_dir']).parts
            resolved = Path(results_dir).resolve()
            if not Path(manifest['output_dir']).is_absolute() and tuple(resolved.parts[-len(parts):]) == parts:
                run_path = str(resolved.parents[len(parts)-1])
        if manifest.get('status') == 'failed' and not allow_incomplete:
            raise ValueError('A failed run manifest cannot expose a successful posterior.')
        expected = manifest.get('n_parameters', n_parameters)
        if expected is None or expected <= 0:
            raise ValueError('Expected parameter count or a run manifest is required.')
        if n_parameters is not None and n_parameters != expected:
            raise ValueError('Manifest and requested parameter counts differ.')
        final = os.path.join(results_dir, 'step_final.h5')
        diagnostic_only = manifest.get('status') == 'failed' or stage_file is not None
        if stage_file is not None:
            if not allow_incomplete or Path(stage_file).name != str(stage_file) or not str(stage_file).startswith('step_') or not str(stage_file).endswith('.h5'):
                raise ValueError('Explicit stage imports require a contained archive and allow_incomplete=True.')
            final = os.path.join(results_dir, stage_file)

        if not os.path.isfile(final) and allow_incomplete and stage_file is None:
            stages = [name for name in os.listdir(results_dir)
                      if name.startswith('step_') and name.endswith('.h5') and name[5:-3].isdigit()]
            if stages:
                final = os.path.join(results_dir, max(stages, key=lambda name: int(name[5:-3])))
                diagnostic_only = True
        with h5py.File(final, 'r') as archive:
            beta = float(np.asarray(archive['Annealer/beta']).item())
            if not np.isfinite(beta) or beta < 0 or beta > 1:
                raise ValueError('Invalid annealing beta.')
            if abs(beta-1.) > 1e-8 and not allow_incomplete:
                raise ValueError('Incomplete annealing output is not a posterior (beta != 1).')
            group = archive['ParameterSets']
            if set(group.keys()) == {'theta'}:
                samples = np.asarray(group['theta'], dtype=float)
            else:
                sets = manifest.get('parameter_sets')
                if not sets or len({s['name'] for s in sets}) != len(sets) or set(group.keys()) != {s['name'] for s in sets}:
                    raise ValueError('Named parameter sets require exact manifest order and widths.')
                blocks = []
                for entry in sets:
                    block = np.asarray(group[entry['name']], dtype=float)
                    if block.ndim != 2 or block.shape[1] != entry['width']:
                        raise ValueError('Named parameter set width mismatch.')
                    blocks.append(block)
                if len({b.shape[0] for b in blocks}) != 1:
                    raise ValueError('Named parameter sets have inconsistent particle counts.')
                samples = np.hstack(blocks)
            if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != expected or not np.isfinite(samples).all():
                raise ValueError('Final samples must be finite with shape (at least 2 particles, N_param).')
            if 'chains' in manifest and len(samples) != manifest['chains']:
                raise ValueError('Final particle count does not match the serial run manifest.')
            probabilities = {}
            if 'Bayesian' in archive:
                for name in ('prior', 'likelihood', 'posterior'):
                    if name in archive['Bayesian']:
                        values = np.asarray(archive['Bayesian'][name], dtype=float).reshape(-1)
                        if values.shape != (len(samples),) or not np.isfinite(values).all():
                            raise ValueError(f'Invalid {name} probability array.')
                        probabilities[name] = values
        transform = manifest.get('transform')
        if transform:
            if 'file' in transform:
                path = Path(run_path)/transform['file']
                if path.resolve().parent != Path(run_path).resolve():
                    raise ValueError('Transform file must be inside the isolated run.')
                if file_hash(path) != transform['sha256']:
                    raise ValueError('Prior transform checksum mismatch.')
                with np.load(path, allow_pickle=False) as arrays:
                    offset, matrix = arrays['offset'], arrays['matrix']
            else:
                offset = np.asarray(transform['offset'])
                matrix = np.asarray(transform.get('scales', transform.get('matrix')))
            if offset.shape != (expected,) or matrix.shape not in ((expected,), (expected, expected)):
                raise ValueError('Invalid saved prior transform dimensions.')
            samples = (samples*matrix if matrix.ndim == 1 else samples @ matrix.T) + offset
            if not np.isfinite(samples).all():
                raise ValueError('Nonfinite physical samples after transform.')
            if manifest.get('prior') == 'uniform':
                upper = offset + (matrix if matrix.ndim == 1 else np.diag(matrix))
                if np.any(samples < offset) or np.any(samples > upper):
                    raise ValueError('Uniform posterior samples violate physical bounds.')
        likelihood_offset = manifest.get('likelihood_offset', 0.)
        prior_offset = manifest.get('prior_offset', 0.)
        if not np.isfinite([likelihood_offset, prior_offset]).all():
            raise ValueError('Probability normalization offsets must be finite.')
        for name, adjustment in dict(likelihood=likelihood_offset, prior=prior_offset,
                                     posterior=prior_offset+beta*likelihood_offset).items():
            if name in probabilities:
                probabilities[name] += adjustment
        record = AltarPosterior(samples, beta, self.load_beta_statistics(results_dir), probabilities,
                                manifest.get('layout', []), run_path,
                                [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir))
                                 if f.startswith('step_') and f.endswith('.h5')])
        record.diagnostics['diagnostic_only'] = diagnostic_only or not record.annealing_complete
        if record.diagnostics['diagnostic_only']:
            record.diagnostics['sampling_adequacy'] = 'incomplete: diagnostic output only'
        return record

    def load_beta_statistics(self, results_dir: str) -> pd.DataFrame:
        """Parses ``BetaStatistics.txt`` into a tidy DataFrame.

        Args:
            results_dir: Path to the ALTar results directory.

        Returns:
            A DataFrame with columns
            ``[iteration, beta, scaling, accepted, invalid, rejected]``.
            Returns an empty DataFrame if the file is absent.
        """
        path = os.path.join(results_dir, "BetaStatistics.txt")
        if not os.path.isfile(path):
            for parent in Path(results_dir).resolve().parents:
                progress = parent/'progress.json'
                if progress.is_file() and (parent/'manifest.json').is_file():
                    stages = json.loads(progress.read_text()).get('stages', [])
                    if stages:
                        return pd.DataFrame(stages).rename(columns={'next_scaling': 'scaling'})
                    break
            warnings.warn(
                f"BetaStatistics.txt not found in {results_dir}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.DataFrame(
                columns=["iteration", "beta", "scaling", "accepted", "invalid", "rejected"]
            )

        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("iteration"):
                    continue
                try:
                    parts = line.replace("(", "").replace(")", "").split(",")
                    records.append(
                        {
                            "iteration": int(parts[0]),
                            "beta": float(parts[1]),
                            "scaling": float(parts[2]),
                            "accepted": int(parts[3]),
                            "invalid": int(parts[4]),
                            "rejected": int(parts[5]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(records)
