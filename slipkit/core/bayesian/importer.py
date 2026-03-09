"""
ALTar result importer.

Reads the HDF5 files and ``BetaStatistics.txt`` that ALTar writes after a
static slip inversion and constructs an ``AltarSlipDistribution``.
"""

import os
import warnings
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd


class AltarResultImporter:
    """Reads ALTar's output directory and builds posterior sample arrays.

    ALTar writes one HDF5 file per beta step (``step_000.h5``, …) plus a
    final ``step_final.h5`` containing the samples at ``beta = 1``. A plain-
    text file ``BetaStatistics.txt`` records the annealing trajectory.

    Typical HDF5 layout::

        step_final.h5
        ├── Annealer/
        │   ├── beta          (scalar)
        │   └── covariance    (N_param × N_param)
        ├── Bayesian/
        │   ├── prior         (N_chains,)
        │   ├── likelihood    (N_chains,)
        │   └── posterior     (N_chains,)
        └── ParameterSets/
            ├── strikeslip    (N_chains × M_patches)
            └── dipslip       (N_chains × M_patches)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        results_dir: str,
        n_patches: int,
        convergence_tolerance: float = 1e-3,
    ) -> "AltarSlipDistribution":  # noqa: F821
        """Loads the final posterior samples from an ALTar results directory.

        Args:
            results_dir: Path to the directory containing ALTar output files.
            n_patches: Number of fault patches (M).
            convergence_tolerance: Tolerance for checking ``beta == 1``.

        Returns:
            An ``AltarSlipDistribution`` populated with posterior samples,
            beta statistics, and convergence metadata.

        Raises:
            FileNotFoundError: If ``step_final.h5`` is not found.
        """
        from slipkit.core.bayesian.results import AltarSlipDistribution

        final_h5 = os.path.join(results_dir, "step_final.h5")
        if not os.path.isfile(final_h5):
            raise FileNotFoundError(
                f"ALTar final output not found: {final_h5}\n"
                "Ensure the ALTar simulation completed successfully."
            )

        ss_samples, ds_samples, final_beta = self._read_final_step(
            final_h5, n_patches
        )
        beta_stats = self.load_beta_statistics(results_dir)
        step_files = self._list_step_files(results_dir)

        if abs(final_beta - 1.0) > convergence_tolerance:
            warnings.warn(
                f"ALTar simulation may not have converged: "
                f"final beta = {final_beta:.4f} (expected 1.0). "
                "Consider increasing `chains` or `steps`.",
                RuntimeWarning,
                stacklevel=2,
            )

        mean_slip = np.concatenate(
            [ss_samples.mean(axis=0), ds_samples.mean(axis=0)]
        )

        return AltarSlipDistribution(
            slip_vector=mean_slip,
            faults=[],
            ss_samples=ss_samples,
            ds_samples=ds_samples,
            beta_statistics=beta_stats,
            final_beta=final_beta,
            step_files=step_files,
        )

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_final_step(
        self, h5_path: str, n_patches: int
    ) -> tuple:
        """Reads posterior samples and beta from ``step_final.h5``.

        Args:
            h5_path: Path to the HDF5 file.
            n_patches: Number of fault patches.

        Returns:
            Tuple ``(ss_samples, ds_samples, final_beta)`` where
            ``ss_samples`` and ``ds_samples`` each have shape
            ``(N_chains, M_patches)``.

        Raises:
            KeyError: If expected datasets are missing from the HDF5 file.
        """
        with h5py.File(h5_path, "r") as f:
            final_beta = float(np.array(f["Annealer/beta"]))

            psets = f["ParameterSets"]
            # ALTar stores parameter sets under their psets_list names.
            # We support both "strikeslip"/"dipslip" and "strike_slip"/"dip_slip".
            ss_key = self._find_key(psets, ["strikeslip", "strike_slip"])
            ds_key = self._find_key(psets, ["dipslip", "dip_slip"])

            ss_samples = np.array(psets[ss_key])  # (N_chains, M)
            ds_samples = np.array(psets[ds_key])  # (N_chains, M)

        if ss_samples.shape[1] != n_patches or ds_samples.shape[1] != n_patches:
            raise ValueError(
                f"Expected {n_patches} patches in posterior samples, "
                f"got strikeslip={ss_samples.shape[1]}, "
                f"dipslip={ds_samples.shape[1]}."
            )

        return ss_samples, ds_samples, final_beta

    @staticmethod
    def _find_key(group: h5py.Group, candidates: List[str]) -> str:
        """Returns the first candidate key present in an HDF5 group.

        Args:
            group: The HDF5 group to search.
            candidates: Ordered list of key names to try.

        Returns:
            The first matching key name.

        Raises:
            KeyError: If none of the candidates exist in the group.
        """
        for key in candidates:
            if key in group:
                return key
        raise KeyError(
            f"None of {candidates} found in HDF5 group '{group.name}'. "
            f"Available keys: {list(group.keys())}"
        )

    @staticmethod
    def _list_step_files(results_dir: str) -> List[str]:
        """Returns sorted paths to all ``step_*.h5`` files in the results dir.

        Args:
            results_dir: Path to the ALTar results directory.

        Returns:
            Sorted list of absolute paths to HDF5 step files.
        """
        files = [
            os.path.join(results_dir, f)
            for f in sorted(os.listdir(results_dir))
            if f.startswith("step_") and f.endswith(".h5")
        ]
        return files
