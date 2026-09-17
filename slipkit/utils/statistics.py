"""
Goodness-of-fit statistics for a slip inversion.

This module computes per-dataset and whole-inversion fit metrics (RMS,
variance reduction, chi-square, etc.) that complement the residual maps produced
by :class:`~slipkit.utils.visualizers.sar_data_fit.SarDataFitVisualizer`.

The metrics operate purely on observed-vs-predicted displacement, so they are
independent of the inversion scheme: pass predicted arrays you already have, or
let :meth:`FitStatistics.compute` build them from the fault(s), slip and engine
(``predicted = G @ slip``), exactly as the residual-plot loop does.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Union

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.physics import GreenFunctionBuilder


# Column order for the summary table.
_METRIC_COLUMNS = [
    "n",
    "rms",
    "wrms",
    "variance_reduction_pct",
    "weighted_vr_pct",
    "chi2",
    "reduced_chi2",
    "mean_residual",
    "std_residual",
    "max_abs_residual",
    "data_norm",
    "residual_norm",
    "correlation",
]


def _nuisance_prediction(slip: object, dataset: GeodeticDataSet) -> np.ndarray:
    """Returns the fitted ramp for a dataset, or zeros if there is none."""
    predictor = getattr(slip, "nuisance_prediction", None)
    if predictor is None:
        return np.zeros(len(dataset))
    return predictor(dataset)


class FitStatistics:
    """Computes goodness-of-fit statistics for a slip inversion."""

    @staticmethod
    def _as_fault_list(
        faults: Union[AbstractFaultModel, Sequence[AbstractFaultModel]],
    ) -> List[AbstractFaultModel]:
        """Normalises a fault or sequence of faults to a list."""
        if isinstance(faults, AbstractFaultModel):
            return [faults]
        return list(faults)

    @staticmethod
    def predict(
        datasets: Sequence[GeodeticDataSet],
        faults: Union[AbstractFaultModel, Sequence[AbstractFaultModel]],
        slip: Union[np.ndarray, "object"],
        engine: GreenFunctionBuilder,
    ) -> List[np.ndarray]:
        """
        Computes ``predicted = G @ slip`` for each dataset.

        Green's-function column blocks are laid out fault by fault, matching the
        slip-vector layout of :class:`~slipkit.core.inversion.SlipDistribution`.
        If ``slip`` is a ``SlipDistribution`` carrying fitted nuisance ramps,
        each dataset's ramp is added to its prediction, so the residuals match
        what the inversion actually minimized.

        Args:
            datasets: The observed datasets.
            faults: The fault model(s) used in the inversion.
            slip: The inverted slip, either a raw vector or a ``SlipDistribution``.
            engine: The Green's-function engine.

        Returns:
            A list of ``(N_i,)`` predicted-displacement arrays, aligned with
            ``datasets``.
        """
        fault_list = FitStatistics._as_fault_list(faults)
        slip_vector = np.asarray(getattr(slip, "slip_vector", slip)).ravel()

        expected = sum(f.num_components() * f.num_patches() for f in fault_list)
        if slip_vector.shape[0] != expected:
            raise ValueError(
                f"slip length ({slip_vector.shape[0]}) does not match the total "
                f"number of unknowns across faults ({expected})."
            )

        predicted = []
        for dataset in datasets:
            pred = np.zeros(len(dataset))
            offset = 0
            for fault in fault_list:
                width = fault.num_components() * fault.num_patches()
                # Matrix-free forward model (no dense kernel materialised).
                pred += engine.predict(fault, dataset, slip_vector[offset:offset + width])
                offset += width
            pred += _nuisance_prediction(slip, dataset)
            predicted.append(pred)
        return predicted

    @staticmethod
    def dataset_metrics(
        observed: np.ndarray,
        predicted: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        n_params: int = 0,
    ) -> Dict[str, float]:
        """
        Computes goodness-of-fit metrics for a single observed/predicted pair.

        Let ``r = observed - predicted`` be the residual and ``w = 1/sigma`` the
        per-point weights (``w = 1`` if ``sigma`` is None). The metrics are:

        * ``rms``: root-mean-square misfit, ``sqrt(mean(r**2))``.
        * ``wrms``: weighted RMS, ``sqrt(sum((w r)**2) / sum(w**2))``.
        * ``variance_reduction_pct``: ``(1 - sum(r**2) / sum(observed**2)) * 100``.
        * ``weighted_vr_pct``: as above but with ``w``-weighted sums.
        * ``chi2``: ``sum((r / sigma)**2)``.
        * ``reduced_chi2``: ``chi2 / (n - n_params)`` (dof floored at 1).
        * ``mean_residual`` / ``std_residual``: bias and scatter of ``r``.
        * ``max_abs_residual``: worst single-point misfit.
        * ``data_norm`` / ``residual_norm``: ``||observed||`` and ``||r||``.
        * ``correlation``: Pearson correlation of observed vs predicted.

        Args:
            observed: ``(N,)`` observed displacements.
            predicted: ``(N,)`` predicted displacements.
            sigma: ``(N,)`` per-point uncertainties for the weighted metrics.
            n_params: Number of free model parameters, used only for the
                reduced-chi-square degrees of freedom.

        Returns:
            A dict of metric name -> value (all in the data's units, except the
            dimensionless percentages, chi-square and correlation).
        """
        d = np.asarray(observed, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        if d.shape != p.shape:
            raise ValueError(
                f"observed and predicted must have the same shape; "
                f"got {d.shape} and {p.shape}."
            )
        n = d.size
        if n == 0:
            raise ValueError("Cannot compute metrics on an empty dataset.")

        r = d - p
        ss_res = float(np.sum(r ** 2))
        ss_dat = float(np.sum(d ** 2))

        w = np.ones_like(d) if sigma is None else 1.0 / np.asarray(sigma, dtype=float).ravel()
        wr2 = float(np.sum((w * r) ** 2))
        wd2 = float(np.sum((w * d) ** 2))
        w2 = float(np.sum(w ** 2))

        dof = max(1, n - int(n_params))

        # Correlation is undefined if either series is constant.
        if n > 1 and np.std(d) > 0 and np.std(p) > 0:
            correlation = float(np.corrcoef(d, p)[0, 1])
        else:
            correlation = float("nan")

        return {
            "n": int(n),
            "rms": float(np.sqrt(ss_res / n)),
            "wrms": float(np.sqrt(wr2 / w2)) if w2 > 0 else float("nan"),
            "variance_reduction_pct": (1.0 - ss_res / ss_dat) * 100.0 if ss_dat > 0 else float("nan"),
            "weighted_vr_pct": (1.0 - wr2 / wd2) * 100.0 if wd2 > 0 else float("nan"),
            "chi2": wr2,
            "reduced_chi2": wr2 / dof,
            "mean_residual": float(np.mean(r)),
            "std_residual": float(np.std(r)),
            "max_abs_residual": float(np.max(np.abs(r))),
            "data_norm": float(np.sqrt(ss_dat)),
            "residual_norm": float(np.sqrt(ss_res)),
            "correlation": correlation,
        }

    @staticmethod
    def compute(
        datasets: Sequence[GeodeticDataSet],
        faults: Optional[Union[AbstractFaultModel, Sequence[AbstractFaultModel]]] = None,
        slip: Optional[Union[np.ndarray, "object"]] = None,
        engine: Optional[GreenFunctionBuilder] = None,
        predicted: Optional[Sequence[np.ndarray]] = None,
        n_params: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Computes per-dataset and whole-inversion fit statistics.

        Supply the predictions one of two ways:

        * pass ``predicted`` (a list aligned with ``datasets``) if you already
          computed ``G @ slip`` (e.g. in the residual-plot loop), or
        * pass ``faults``, ``slip`` and ``engine`` to let this method build them.

        Args:
            datasets: The observed datasets.
            faults: Fault model(s); required unless ``predicted`` is given. Also
                used to size the reduced-chi-square dof of the "Overall" row when
                ``n_params`` is not supplied.
            slip: Inverted slip (vector or ``SlipDistribution``); required unless
                ``predicted`` is given.
            engine: Green's-function engine; required unless ``predicted`` is given.
            predicted: Precomputed predicted arrays, one per dataset.
            n_params: Free-parameter count for the "Overall" reduced chi-square.
                Defaults to the total number of slip unknowns (from ``faults``)
                when available, else 0. Per-dataset reduced chi-square always uses
                the full point count as its dof (``n_params = 0``).

        Returns:
            A :class:`pandas.DataFrame` with one row per dataset (indexed by
            ``dataset.name``) plus a final ``"Overall"`` row aggregating every
            point, and one column per metric (see :meth:`dataset_metrics`).
        """
        if predicted is None:
            if faults is None or slip is None or engine is None:
                raise ValueError(
                    "Provide either predicted=..., or all of faults, slip and engine."
                )
            predicted = FitStatistics.predict(datasets, faults, slip, engine)

        if len(predicted) != len(datasets):
            raise ValueError(
                f"predicted has {len(predicted)} entries but there are "
                f"{len(datasets)} datasets."
            )

        # Total free parameters for the overall reduced chi-square.
        if n_params is None:
            if faults is not None:
                fault_list = FitStatistics._as_fault_list(faults)
                n_params = sum(f.num_components() * f.num_patches() for f in fault_list)
            else:
                n_params = 0
            # Ramp coefficients are free parameters too.
            n_params += sum(
                ramp.num_params for ramp in getattr(slip, "nuisance", {}).values()
            )

        rows: Dict[str, Dict[str, float]] = {}
        obs_all, pred_all, sig_all = [], [], []

        for dataset, pred in zip(datasets, predicted):
            obs = np.asarray(dataset.data, dtype=float).ravel()
            pred = np.asarray(pred, dtype=float).ravel()
            sig = (
                np.ones_like(obs)
                if dataset.sigma is None
                else np.asarray(dataset.sigma, dtype=float).ravel()
            )

            # Per-dataset dof uses the full point count (n_params = 0); attributing
            # the whole model's parameters to one dataset is not meaningful.
            rows[dataset.name] = FitStatistics.dataset_metrics(obs, pred, sig, n_params=0)

            obs_all.append(obs)
            pred_all.append(pred)
            sig_all.append(sig)

        rows["Overall"] = FitStatistics.dataset_metrics(
            np.concatenate(obs_all),
            np.concatenate(pred_all),
            np.concatenate(sig_all),
            n_params=int(n_params),
        )

        df = pd.DataFrame.from_dict(rows, orient="index")[_METRIC_COLUMNS]
        df["n"] = df["n"].astype(int)
        return df
