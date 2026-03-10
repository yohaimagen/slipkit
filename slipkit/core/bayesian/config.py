"""
ALTar configuration file builder.

Generates ALTar ``.pfg`` configuration files from Python parameters, covering
the static slip inversion model with optional GPU, MPI, and InSAR ramp support.
"""

import os
from typing import Optional, Tuple

import numpy as np


class AltarConfigBuilder:
    """Builds ALTar ``.pfg`` configuration files for static slip inversion.

    The generated configuration targets
    ``altar.models.seismic.cuda.static`` (GPU) or
    ``altar.models.seismic.static`` (CPU) and assembles ``strikeslip``,
    ``dipslip``, and optionally ``ramp`` parameter sets.

    Attributes:
        app_name: Root key in the ``.pfg`` file (ALTar application name).
        n_patches: Number of fault patches (= M).
        n_observations: Total number of data observations.
        case_dir: Directory containing ALTar input files.
        green_file: File name of the Green's function HDF5.
        data_file: File name of the observation vector HDF5.
        cd_file: File name of the covariance matrix HDF5.
        areas_km2: Patch areas in km² used by the Moment distribution.
        mw_mean: Mean moment magnitude for Dirichlet seeding.
        mw_sigma: Standard deviation of moment magnitude for seeding.
        rigidity_gpa: Shear modulus in GPa for Moment distribution.
        ss_prior_sigma: Sigma of the Gaussian prior on strike-slip.
        ds_prior_support: ``(lower, upper)`` bounds of the Uniform prior on
            dip-slip in metres.
        use_moment_prep: Whether to seed dip-slip with the Moment distribution.
        n_ramp_params: Number of InSAR ramp parameters (0 = disabled).
        ramp_prior_support: ``(lower, upper)`` bounds for ramp parameters.
        use_gpu: If True, use CUDA model and sampler.
        chains: Number of Markov chains per task.
        steps: Metropolis burn-in steps per beta step.
        tasks: MPI tasks per host.
        hosts: Number of MPI hosts.
        gpus: GPUs per task (ignored when ``use_gpu=False``).
        gpu_precision: ``"float32"`` or ``"float64"``.
        output_dir: Directory where ALTar writes result HDF5 files.
        output_freq: Save results every this many beta steps.
    """

    def __init__(
        self,
        n_patches: int,
        n_observations: int,
        case_dir: str,
        areas_km2: np.ndarray,
        mw_mean: float,
        mw_sigma: float = 0.2,
        rigidity_gpa: float = 30.0,
        app_name: str = "altar",
        green_file: str = "green.h5",
        data_file: str = "data.h5",
        cd_file: str = "cd.h5",
        ss_prior_sigma: float = 0.5,
        ds_prior_support: Tuple[float, float] = (-0.5, 20.0),
        use_moment_prep: bool = True,
        n_ramp_params: int = 0,
        ramp_prior_support: Tuple[float, float] = (-1.0, 1.0),
        use_gpu: bool = False,
        chains: int = 2**10,
        steps: int = 1000,
        tasks: int = 1,
        hosts: int = 1,
        gpus: int = 1,
        gpu_precision: str = "float32",
        output_dir: str = "results",
        output_freq: int = 1,
    ) -> None:
        self.app_name = app_name
        self.n_patches = n_patches
        self.n_observations = n_observations
        self.case_dir = os.path.abspath(case_dir)
        self.areas_km2 = areas_km2
        self.mw_mean = mw_mean
        self.mw_sigma = mw_sigma
        self.rigidity_gpa = rigidity_gpa
        self.green_file = green_file
        self.data_file = data_file
        self.cd_file = cd_file
        self.ss_prior_sigma = ss_prior_sigma
        self.ds_prior_support = ds_prior_support
        self.use_moment_prep = use_moment_prep
        self.n_ramp_params = n_ramp_params
        self.ramp_prior_support = ramp_prior_support
        self.use_gpu = use_gpu
        self.chains = chains
        self.steps = steps
        self.tasks = tasks
        self.hosts = hosts
        self.gpus = gpus
        self.gpu_precision = gpu_precision
        self.output_dir = output_dir
        self.output_freq = output_freq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Renders the complete ``.pfg`` configuration as a string.

        Returns:
            A multi-line string containing a valid ALTar ``.pfg`` file.
        """
        model_key = (
            "altar.models.seismic.cuda.static"
            if self.use_gpu
            else "altar.models.seismic.static"
        )
        pset_key = (
            "altar.cuda.models.parameterset"
            if self.use_gpu
            else "contiguous"
        )
        dist_prefix = "altar.cuda.distributions" if self.use_gpu else "altar.distributions"
        sampler = (
            "altar.cuda.bayesian.metropolis"
            if self.use_gpu
            else "altar.bayesian.metropolis"
        )

        psets_list = ["strikeslip", "dipslip"]
        if self.n_ramp_params > 0:
            psets_list.append("ramp")

        areas_str = self._format_areas()

        lines = [
            f"; ALTar static slip inversion — auto-generated by SlipKit",
            f"",
            f"{self.app_name}:",
            f"",
            f"    model = {model_key}",
            f"    model:",
            f"        case = {self.case_dir}",
            f"        patches = {self.n_patches}",
            f"        green = {self.green_file}",
            f"",
            f"        dataobs = {'altar.cuda.data.datal2' if self.use_gpu else 'altar.data.datal2'}",
            f"        dataobs:",
            f"            observations = {self.n_observations}",
            f"            data_file = {self.data_file}",
            f"            cd_file = {self.cd_file}",
            f"",
            f"        psets_list = [{', '.join(psets_list)}]",
            f"",
            f"        psets:",
            f"",
            f"            strikeslip = {pset_key}",
            f"            strikeslip:",
            f"                count = {self.n_patches}",
            f"                prior = {dist_prefix}.gaussian",
            f"                prior.mean = 0",
            f"                prior.sigma = {self.ss_prior_sigma}",
            f"",
            f"            dipslip = {pset_key}",
            f"            dipslip:",
            f"                count = {self.n_patches}",
        ]

        if self.use_moment_prep:
            moment_model = (
                "altar.models.seismic.cuda.moment"
                if self.use_gpu
                else "altar.models.seismic.moment"
            )
            lines += [
                f"                prep = {moment_model}",
                f"                prep:",
                f"                    Mw_mean = {self.mw_mean}",
                f"                    Mw_sigma = {self.mw_sigma}",
                f"                    Mu = [{self.rigidity_gpa}]",
                f"                    area = [{areas_str}]",
            ]

        lines += [
            f"                prior = {dist_prefix}.uniform",
            f"                prior.support = {self.ds_prior_support}",
        ]

        if self.n_ramp_params > 0:
            lines += [
                f"",
                f"            ramp = {pset_key}",
                f"            ramp:",
                f"                count = {self.n_ramp_params}",
                f"                prior = {dist_prefix}.uniform",
                f"                prior.support = {self.ramp_prior_support}",
            ]

        lines += [
            f"",
            f"    controller:",
            f"        sampler = {sampler}",
            f"        archiver:",
            f"            output_dir = {self.output_dir}",
            f"            output_freq = {self.output_freq}",
            f"",
            f"    job:",
        ]
        if self.hosts > 1 or self.tasks > 1:
            lines.append(f"        hosts = {self.hosts}")
            lines.append(f"        tasks = {self.tasks}")
        if self.use_gpu:
            lines.append(f"        gpus = {self.gpus}")
            lines.append(f"        gpuprecision = {self.gpu_precision}")
        lines += [
            f"        chains = {self.chains}",
            f"        steps  = {self.steps}",
            f"",
        ]

        return "\n".join(lines)

    def save(self, filepath: str) -> str:
        """Writes the ``.pfg`` configuration to disk.

        Args:
            filepath: Absolute or relative path for the output file.

        Returns:
            Absolute path of the written file.
        """
        filepath = os.path.abspath(filepath)
        with open(filepath, "w") as fh:
            fh.write(self.build())
        return filepath

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_areas(self) -> str:
        """Formats the areas array as a compact comma-separated string.

        Returns:
            Comma-separated list of km² area values suitable for embedding
            in the ``.pfg`` ``area = [...]`` field.
        """
        return ", ".join(f"{a:.6g}" for a in self.areas_km2)
