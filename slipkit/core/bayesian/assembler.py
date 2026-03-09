"""
ALTar Assembler for Bayesian static slip inversion.

Prepares raw Strike-Slip and Dip-Slip Green's function matrices without rake
rotation and without Tikhonov regularization. Rake constraints are expressed
as prior distributions in the ALTar configuration rather than as kernel
transformations.
"""

from typing import List, Optional, Tuple

import numpy as np

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.inversion import AbstractAssembler
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.regularization import RegularizationManager


class AltarAssembler(AbstractAssembler):
    """Assembles raw SS/DS Green's function matrices for the ALTar backend.

    Returns the unrotated Green's function kernel and raw observations without
    regularization. The ``regularization_manager`` and ``lambda_spatial``
    arguments are accepted for API compatibility but are intentionally ignored;
    regularization is encoded as prior distributions in the ALTar configuration.

    The assembled system has the layout::

        A = [G_ss | G_ds]   shape (N_obs, 2 * M_patches)
        b = [d_obs | sigma]  shape (2 * N_obs,)

    where ``G_ss`` and ``G_ds`` are strike-slip and dip-slip columns
    consistent with ALTar's ``psets_list = [strikeslip, dipslip]`` convention.

    Attributes:
        _G_raw_cache: Cached Green's function matrix ``(N_obs, 2 * M)``.
        _d_obs_cache: Cached stacked observations vector ``(N_obs,)``.
        _sigma_cache: Cached stacked uncertainties ``(N_obs,)``.
    """

    def __init__(self) -> None:
        self._G_raw_cache: Optional[np.ndarray] = None
        self._d_obs_cache: Optional[np.ndarray] = None
        self._sigma_cache: Optional[np.ndarray] = None

    def assemble(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
        regularization_manager: RegularizationManager,
        lambda_spatial: float,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assembles the raw Green's function system for ALTar.

        Args:
            faults: List of fault models contributing to the inversion.
            datasets: List of geodetic observation datasets.
            engine: Physics engine used to compute Green's functions.
            regularization_manager: Ignored; present for API compatibility.
            lambda_spatial: Ignored; present for API compatibility.
            force_recompute: If True, clears the kernel cache before computing.

        Returns:
            A tuple ``(A, b)`` where:
            - ``A`` is the stacked Green's function matrix
              ``[G_ss | G_ds]`` of shape ``(N_obs, 2 * M_patches)``.
            - ``b`` is the concatenated vector
              ``[d_obs | sigma]`` of shape ``(2 * N_obs,)``.
        """
        if self._G_raw_cache is None or force_recompute:
            self._compute_kernels(faults, datasets, engine)

        A = self._G_raw_cache
        b = np.concatenate([self._d_obs_cache, self._sigma_cache])
        return A, b

    def _compute_kernels(
        self,
        faults: List[AbstractFaultModel],
        datasets: List[GeodeticDataSet],
        engine: GreenFunctionBuilder,
    ) -> None:
        """Computes and caches the raw kernel, observations, and uncertainties.

        Args:
            faults: List of fault models.
            datasets: List of geodetic datasets.
            engine: Green's function engine.
        """
        g_blocks: List[np.ndarray] = []
        d_blocks: List[np.ndarray] = []
        sigma_blocks: List[np.ndarray] = []

        for dataset in datasets:
            fault_g_parts: List[np.ndarray] = []
            for fault in faults:
                fault_g_parts.append(engine.build_kernel(fault, dataset))
            g_blocks.append(np.concatenate(fault_g_parts, axis=1))
            d_blocks.append(dataset.data)
            sigma_blocks.append(dataset.sigma)

        self._G_raw_cache = np.vstack(g_blocks)
        self._d_obs_cache = np.concatenate(d_blocks)
        self._sigma_cache = np.concatenate(sigma_blocks)

    def get_areas(self, faults: List[AbstractFaultModel]) -> np.ndarray:
        """Returns concatenated patch areas across all faults in square metres.

        Args:
            faults: List of fault models implementing ``get_areas()``.

        Returns:
            A ``(M_total,)`` array of patch areas in m².
        """
        return np.concatenate([f.get_areas() for f in faults])

    def clear_cache(self) -> None:
        """Clears all cached arrays, forcing recomputation on next call."""
        self._G_raw_cache = None
        self._d_obs_cache = None
        self._sigma_cache = None
