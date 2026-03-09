"""
ALTar data exporter.

Converts SlipKit data structures into the HDF5 files that ALTar's static slip
inversion model expects as inputs.
"""

import os
from typing import Dict

import h5py
import numpy as np


class AltarDataExporter:
    """Exports SlipKit arrays to ALTar-compatible HDF5 input files.

    ALTar's static model expects three mandatory input files:

    * ``green.h5`` — Green's function matrix ``(N_obs, N_param)``.
    * ``data.h5``  — Observed displacement vector ``(N_obs,)``.
    * ``cd.h5``    — Data covariance matrix ``(N_obs, N_obs)``.

    Optionally, a ``areas.h5`` file stores patch areas for the Moment
    distribution seeding.

    Attributes:
        output_dir: Directory where all exported files are written.
    """

    def __init__(self, output_dir: str) -> None:
        """Initialises the exporter.

        Args:
            output_dir: Path to the directory for ALTar input files. Created
                if it does not already exist.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual export methods
    # ------------------------------------------------------------------

    def export_greens_function(
        self, G: np.ndarray, filename: str = "green.h5"
    ) -> str:
        """Writes the Green's function matrix to an HDF5 file.

        ALTar expects the matrix stored with shape ``(N_obs, N_param)`` and
        dataset name ``"green"``.

        Args:
            G: Green's function matrix of shape ``(N_obs, N_param)`` where
               ``N_param = 2 * M_patches``.
            filename: Output file name relative to ``output_dir``.

        Returns:
            Absolute path of the written file.
        """
        path = os.path.join(self.output_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("green", data=G.astype(np.float64))
        return path

    def export_data(
        self, d_obs: np.ndarray, filename: str = "data.h5"
    ) -> str:
        """Writes the observed displacement vector to an HDF5 file.

        Args:
            d_obs: Observed displacements of shape ``(N_obs,)``.
            filename: Output file name relative to ``output_dir``.

        Returns:
            Absolute path of the written file.
        """
        path = os.path.join(self.output_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=d_obs.astype(np.float64))
        return path

    def export_covariance(
        self,
        sigma: np.ndarray,
        d_obs: np.ndarray,
        alpha_cp: float = 0.0,
        filename: str = "cd.h5",
    ) -> str:
        """Writes the data covariance matrix (plus optional static Cp) to HDF5.

        Constructs ``Cd = diag(sigma²)``. When ``alpha_cp > 0``, the static
        prediction covariance ``Cp = diag((alpha_cp * d_obs)²)`` is added to
        form the combined misfit covariance ``Cx = Cd + Cp`` following the
        Minson et al. (2013) fractional error model.

        Args:
            sigma: Diagonal data uncertainties of shape ``(N_obs,)``.
            d_obs: Observed displacements of shape ``(N_obs,)`` used to
                compute ``Cp`` when ``alpha_cp > 0``.
            alpha_cp: Fractional model error coefficient. Set to ``0`` to
                use ``Cd`` only.
            filename: Output file name relative to ``output_dir``.

        Returns:
            Absolute path of the written file.
        """
        cd_diag = sigma ** 2
        if alpha_cp > 0.0:
            cp_diag = (alpha_cp * d_obs) ** 2
            cx_diag = cd_diag + cp_diag
        else:
            cx_diag = cd_diag

        cx = np.diag(cx_diag)
        path = os.path.join(self.output_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("cd", data=cx.astype(np.float64))
        return path

    def export_areas(
        self, areas_m2: np.ndarray, filename: str = "areas.h5"
    ) -> str:
        """Writes patch areas (in km²) to an HDF5 file.

        ALTar's Moment distribution expects areas in km². This method converts
        from m² (SlipKit internal units) to km² before writing.

        Args:
            areas_m2: Patch areas in square metres, shape ``(M,)``.
            filename: Output file name relative to ``output_dir``.

        Returns:
            Absolute path of the written file.
        """
        areas_km2 = areas_m2 / 1e6
        path = os.path.join(self.output_dir, filename)
        with h5py.File(path, "w") as f:
            f.create_dataset("areas", data=areas_km2.astype(np.float64))
        return path

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def export_all(
        self,
        G: np.ndarray,
        d_obs: np.ndarray,
        sigma: np.ndarray,
        areas_m2: np.ndarray,
        alpha_cp: float = 0.0,
    ) -> Dict[str, str]:
        """Exports all required ALTar input files in a single call.

        Args:
            G: Green's function matrix ``(N_obs, 2 * M_patches)``.
            d_obs: Observed displacements ``(N_obs,)``.
            sigma: Diagonal uncertainties ``(N_obs,)``.
            areas_m2: Patch areas in m² ``(M_patches,)``.
            alpha_cp: Fractional model error coefficient for static Cp.

        Returns:
            A dict mapping logical names to absolute file paths::

                {
                    "green": "/path/green.h5",
                    "data":  "/path/data.h5",
                    "cd":    "/path/cd.h5",
                    "areas": "/path/areas.h5",
                }
        """
        return {
            "green": self.export_greens_function(G),
            "data": self.export_data(d_obs),
            "cd": self.export_covariance(sigma, d_obs, alpha_cp),
            "areas": self.export_areas(areas_m2),
        }
