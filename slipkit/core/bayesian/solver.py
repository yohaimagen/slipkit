"""
ALTar Bayesian solver strategy.

Orchestrates data export, configuration generation, ALTar subprocess execution,
and result import for a full static Bayesian slip inversion via the CATMIP
algorithm.
"""

import os
import subprocess
import warnings
from typing import Optional, Tuple

import numpy as np

from slipkit.core.solvers import SolverStrategy
from slipkit.core.bayesian.config import AltarConfigBuilder
from slipkit.core.bayesian.exporter import AltarDataExporter
from slipkit.core.bayesian.importer import AltarResultImporter
from slipkit.core.bayesian.results import AltarSlipDistribution


class AltarBayesianSolver(SolverStrategy):
    """Bayesian slip inversion solver backed by ALTar's CATMIP algorithm.

    This solver receives the assembled Green's function matrix and data vector
    from :class:`~slipkit.core.bayesian.assembler.AltarAssembler`, exports
    them to HDF5 files, generates an ALTar ``.pfg`` configuration file, runs
    ALTar as a subprocess, and imports the posterior samples into an
    :class:`~slipkit.core.bayesian.results.AltarSlipDistribution`.

    **Expected input layout from** ``AltarAssembler``::

        A — shape (N_obs, 2 * M_patches) : [G_ss | G_ds]
        b — shape (2 * N_obs,)           : [d_obs | sigma]

    The ``bounds`` argument accepted by :meth:`solve` is ignored; slip bounds
    are controlled by the prior distributions in the ``.pfg`` configuration.

    Attributes:
        mw_mean: Mean moment magnitude used for Dirichlet seeding.
        mw_sigma: Standard deviation of moment magnitude for seeding.
        areas_m2: Patch areas in m² ``(M,)``.
        work_dir: Working directory for all ALTar I/O.
        alpha_cp: Fractional model error for static Cp (0 = disabled).
        ss_prior_sigma: Sigma of Gaussian prior on strike-slip.
        ds_prior_support: ``(lower, upper)`` uniform prior bounds on dip-slip.
        n_ramp_params: Number of InSAR ramp parameters (0 = disabled).
        chains: Markov chains per ALTar task.
        steps: Metropolis burn-in steps per beta step.
        tasks: MPI tasks per host.
        hosts: Number of MPI hosts.
        gpus: GPUs per task.
        gpu_precision: ``"float32"`` or ``"float64"``.
        use_gpu: Whether to use the CUDA model and sampler.
        output_freq: Save results every this many beta steps.
        keep_work_dir: Retain the working directory after the run.
        altar_cmd: Shell command used to invoke ALTar (default ``"slipmodel"``).
        last_result: The most recent ``AltarSlipDistribution`` from :meth:`solve`.
    """

    def __init__(
        self,
        mw_mean: float,
        mw_sigma: float,
        areas_m2: np.ndarray,
        work_dir: str = "./altar_run",
        alpha_cp: float = 0.0,
        ss_prior_sigma: float = 0.5,
        ds_prior_support: Tuple[float, float] = (-0.5, 20.0),
        n_ramp_params: int = 0,
        chains: int = 2**10,
        steps: int = 1000,
        tasks: int = 1,
        hosts: int = 1,
        gpus: int = 1,
        gpu_precision: str = "float32",
        use_gpu: bool = False,
        output_freq: int = 1,
        keep_work_dir: bool = True,
        altar_cmd: str = "slipmodel",
    ) -> None:
        self.mw_mean = mw_mean
        self.mw_sigma = mw_sigma
        self.areas_m2 = areas_m2
        self.work_dir = os.path.abspath(work_dir)
        self.alpha_cp = alpha_cp
        self.ss_prior_sigma = ss_prior_sigma
        self.ds_prior_support = ds_prior_support
        self.n_ramp_params = n_ramp_params
        self.chains = chains
        self.steps = steps
        self.tasks = tasks
        self.hosts = hosts
        self.gpus = gpus
        self.gpu_precision = gpu_precision
        self.use_gpu = use_gpu
        self.output_freq = output_freq
        self.keep_work_dir = keep_work_dir
        self.altar_cmd = altar_cmd
        self.last_result: Optional[AltarSlipDistribution] = None

    # ------------------------------------------------------------------
    # SolverStrategy interface
    # ------------------------------------------------------------------

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Runs the full ALTar Bayesian inversion pipeline.

        Args:
            A: Assembled Green's function matrix ``[G_ss | G_ds]``,
               shape ``(N_obs, 2 * M_patches)``.
            b: Assembled data vector ``[d_obs | sigma]``,
               shape ``(2 * N_obs,)``.
            bounds: Ignored; present for interface compatibility.

        Returns:
            Posterior mean slip vector of shape ``(2 * M_patches,)``
            ordered as ``[mean_ss | mean_ds]``.

        Raises:
            ValueError: If array dimensions are inconsistent.
            RuntimeError: If ALTar exits with a non-zero return code.
        """
        n_obs, n_param = A.shape
        n_patches = n_param // 2

        if n_param % 2 != 0:
            raise ValueError(
                f"Expected A with an even number of columns (2*M_patches), "
                f"got {n_param}."
            )
        if len(b) != 2 * n_obs:
            raise ValueError(
                f"Expected b of length 2*N_obs={2 * n_obs}, got {len(b)}."
            )
        if len(self.areas_m2) != n_patches:
            raise ValueError(
                f"areas_m2 has {len(self.areas_m2)} entries but A implies "
                f"{n_patches} patches."
            )

        G = A
        d_obs = b[:n_obs]
        sigma = b[n_obs:]

        os.makedirs(self.work_dir, exist_ok=True)
        case_dir = os.path.join(self.work_dir, "case")
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(case_dir, exist_ok=True)

        self._export_data(G, d_obs, sigma, case_dir)
        pfg_path = self._write_config(n_patches, n_obs, case_dir, results_dir)
        self._run_altar(pfg_path)
        result = self._import_results(results_dir, n_patches)

        self.last_result = result
        return result.get_mean_slip()

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_last_result(self) -> Optional[AltarSlipDistribution]:
        """Returns the ``AltarSlipDistribution`` from the most recent run.

        Returns:
            The last result object, or ``None`` if :meth:`solve` has not
            been called yet.
        """
        return self.last_result

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _export_data(
        self,
        G: np.ndarray,
        d_obs: np.ndarray,
        sigma: np.ndarray,
        case_dir: str,
    ) -> None:
        """Exports Green's functions, data, covariance, and areas to HDF5.

        Args:
            G: Green's function matrix ``(N_obs, 2 * M)``.
            d_obs: Observations vector ``(N_obs,)``.
            sigma: Diagonal uncertainties ``(N_obs,)``.
            case_dir: Directory to write ALTar input files.
        """
        exporter = AltarDataExporter(case_dir)
        exporter.export_all(G, d_obs, sigma, self.areas_m2, self.alpha_cp)

    def _write_config(
        self,
        n_patches: int,
        n_obs: int,
        case_dir: str,
        results_dir: str,
    ) -> str:
        """Generates and saves the ALTar ``.pfg`` configuration file.

        Args:
            n_patches: Number of fault patches.
            n_obs: Number of observations.
            case_dir: Directory containing ALTar input files.
            results_dir: Directory where ALTar writes output.

        Returns:
            Absolute path to the written ``.pfg`` file.
        """
        areas_km2 = self.areas_m2 / 1e6
        builder = AltarConfigBuilder(
            n_patches=n_patches,
            n_observations=n_obs,
            case_dir=case_dir,
            areas_km2=areas_km2,
            mw_mean=self.mw_mean,
            mw_sigma=self.mw_sigma,
            ss_prior_sigma=self.ss_prior_sigma,
            ds_prior_support=self.ds_prior_support,
            n_ramp_params=self.n_ramp_params,
            use_gpu=self.use_gpu,
            chains=self.chains,
            steps=self.steps,
            tasks=self.tasks,
            hosts=self.hosts,
            gpus=self.gpus,
            gpu_precision=self.gpu_precision,
            output_dir=results_dir,
            output_freq=self.output_freq,
        )
        pfg_path = os.path.join(self.work_dir, "slipmodel.pfg")
        builder.save(pfg_path)
        return pfg_path

    def _run_altar(self, pfg_path: str) -> None:
        """Invokes ALTar as a subprocess and monitors completion.

        Args:
            pfg_path: Absolute path to the ``.pfg`` configuration file.

        Raises:
            RuntimeError: If the ``slipmodel`` command exits with a non-zero
                return code or is not found on ``PATH``.
        """
        cmd = [self.altar_cmd, f"--config={pfg_path}"]
        print(f"Running ALTar: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=False,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"ALTar command '{self.altar_cmd}' not found on PATH.\n"
                "Install ALTar and ensure the 'slipmodel' executable is "
                "accessible. See https://altar.readthedocs.io for installation."
            ) from None

        if result.returncode != 0:
            raise RuntimeError(
                f"ALTar exited with code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                "Check the terminal output above for ALTar error messages."
            )

    def _import_results(
        self, results_dir: str, n_patches: int
    ) -> AltarSlipDistribution:
        """Reads ALTar's HDF5 output and returns an AltarSlipDistribution.

        Args:
            results_dir: Directory containing ``step_final.h5`` and
                ``BetaStatistics.txt``.
            n_patches: Number of fault patches expected in the output.

        Returns:
            Fully populated ``AltarSlipDistribution``.
        """
        importer = AltarResultImporter()
        return importer.load(results_dir, n_patches)
