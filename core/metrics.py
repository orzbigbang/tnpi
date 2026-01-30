import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from enum_types import SignalType
from state import CompareConfig
from core.parse import parse_odg_time_series_csv, parse_plant_samples_csv

EPS = 1e-12
OFFSET_SEARCH_MS = 12000
OFFSET_GRID_MS = 50
OFFSET_MIN_POINTS = 30


def linear_interp_truth(
    t_odg: np.ndarray,
    x_odg: np.ndarray,
    t_plant: np.ndarray,
    out_of_range_policy: str,
) -> np.ndarray:
    if out_of_range_policy == "clip":
        t_plant2 = np.clip(t_plant, t_odg[0], t_odg[-1])
        return np.interp(t_plant2, t_odg, x_odg)
    x_hat = np.interp(np.clip(t_plant, t_odg[0], t_odg[-1]), t_odg, x_odg)
    mask = (t_plant < t_odg[0]) | (t_plant > t_odg[-1])
    x_hat = x_hat.astype(float)
    x_hat[mask] = np.nan
    return x_hat


def _to_datetime_ns(t_vals: np.ndarray) -> pd.Series:
    t_num = pd.to_numeric(t_vals, errors="coerce")
    t_int = pd.Series(t_num, copy=False).astype("Int64")
    return pd.to_datetime(t_int, unit="ns", errors="coerce")


def fast_estimate_offset_ms_resample(
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    grid_ms: int,
    search_ms: int,
) -> float:
    t1_min, t1_max = t1["ts"].min(), t1["ts"].max()
    t2_min, t2_max = t2["ts"].min(), t2["ts"].max()

    start = max(t1_min, t2_min - pd.to_timedelta(search_ms, unit="ms"))
    end = min(t1_max, t2_max + pd.to_timedelta(search_ms, unit="ms"))
    if start >= end:
        return float("nan")

    grid = pd.date_range(start=start, end=end, freq=f"{grid_ms}ms")
    s1 = (
        t1.set_index("ts")["v1"]
        .sort_index()
        .resample(f"{grid_ms}ms")
        .median()
        .reindex(grid)
    )
    s2 = (
        t2.set_index("ts")["v2"]
        .sort_index()
        .reindex(grid)
        .interpolate(method="time", limit_direction="both")
    )

    a = s1.to_numpy(dtype="float64")
    b = s2.to_numpy(dtype="float64")

    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < OFFSET_MIN_POINTS:
        return float("nan")

    a = a[valid]
    b = b[valid]

    def z(x: np.ndarray) -> np.ndarray:
        m = np.mean(x)
        sd = np.std(x)
        if not np.isfinite(sd) or sd < 1e-12:
            return x - m
        return (x - m) / sd

    a = z(a)
    b = z(b)

    max_shift = int(search_ms // grid_ms)
    best_shift = 0
    best_rmse = float("inf")

    for sh in range(-max_shift, max_shift + 1):
        if sh < 0:
            aa = a[-sh:]
            bb = b[:len(aa)]
        elif sh > 0:
            bb = b[sh:]
            aa = a[:len(bb)]
        else:
            aa = a
            bb = b

        if len(aa) < OFFSET_MIN_POINTS:
            continue

        diff = aa - bb
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = sh

    return float(best_shift * grid_ms)


def estimate_offset_ms(
    t1_ns: np.ndarray,
    v1: np.ndarray,
    t2_ns: np.ndarray,
    v2: np.ndarray,
    grid_ms: int = OFFSET_GRID_MS,
    search_ms: int = OFFSET_SEARCH_MS,
) -> float:
    t1_dt = _to_datetime_ns(t1_ns)
    t2_dt = _to_datetime_ns(t2_ns)
    mask1 = t1_dt.notna().to_numpy() & np.isfinite(v1)
    mask2 = t2_dt.notna().to_numpy() & np.isfinite(v2)
    if mask1.sum() < OFFSET_MIN_POINTS or mask2.sum() < OFFSET_MIN_POINTS:
        return float("nan")
    t1 = pd.DataFrame({"ts": t1_dt[mask1].to_numpy(), "v1": v1[mask1]})
    t2 = pd.DataFrame({"ts": t2_dt[mask2].to_numpy(), "v2": v2[mask2]})
    if t1.empty or t2.empty:
        return float("nan")
    return fast_estimate_offset_ms_resample(t1, t2, grid_ms=grid_ms, search_ms=search_ms)


def compute_compare_metrics(
    id_to_signal: Dict[str, Union[str, Tuple[str, SignalType]]],
    plant_data_path: Union[str, os.PathLike],
    odg_path: Union[str, os.PathLike],
    cfg: CompareConfig,
    time_range: Optional[Tuple[float, float]] = None,
    plant_cache: Optional[Tuple[pd.Series, pd.DataFrame]] = None,
    *,
    plant_encoding: Optional[str] = None,
    odg_encoding: Optional[str] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if plant_cache is not None:
        t_l, y_l = plant_cache
    else:
        t_l, y_l = parse_plant_samples_csv(plant_data_path, id_to_signal, encoding=plant_encoding)
    t_h, x_h = parse_odg_time_series_csv(odg_path, encoding=odg_encoding)

    t_plant = t_l.to_numpy(dtype=float)
    t_odg = t_h.to_numpy(dtype=float)

    plant_order = np.argsort(t_plant)
    odg_order = np.argsort(t_odg)
    t_plant = t_plant[plant_order]
    t_odg = t_odg[odg_order]

    y_l = y_l.iloc[plant_order].reset_index(drop=True)
    x_h = x_h.iloc[odg_order].reset_index(drop=True)

    if time_range is not None:
        start, end = time_range
        if start > end:
            raise ValueError("Time range start must be <= end.")
        plant_mask = (t_plant >= start) & (t_plant <= end)
        odg_mask = (t_odg >= start) & (t_odg <= end)
        t_plant = t_plant[plant_mask]
        t_odg = t_odg[odg_mask]
        y_l = y_l.iloc[plant_mask].reset_index(drop=True)
        x_h = x_h.iloc[odg_mask].reset_index(drop=True)

    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between plant and odg CSV.")

    denom = {}
    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=float)
        if cfg.accuracy_denominator == "range":
            d = np.nanmax(xs) - np.nanmin(xs)
        elif cfg.accuracy_denominator == "std":
            d = np.nanstd(xs)
        else:
            d = None
        denom[c] = d

    rows = []
    acc_values = []
    rmse_values = []
    corr_values = []

    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=float)
        ys = y_l[c].to_numpy(dtype=float)

        valid = np.isfinite(t_odg) & np.isfinite(xs)
        th = t_odg[valid]
        xh = xs[valid]
        if len(th) < 2:
            continue
        df_tmp = pd.DataFrame({"t": th, "x": xh}).drop_duplicates(subset=["t"]).sort_values("t")
        th = df_tmp["t"].to_numpy(dtype=float)
        xh = df_tmp["x"].to_numpy(dtype=float)
        if len(th) < 2:
            continue

        x_hat = linear_interp_truth(th, xh, t_plant, cfg.out_of_range_policy)

        ok = np.isfinite(ys) & np.isfinite(x_hat)
        if ok.sum() == 0:
            continue

        diff = ys - x_hat
        abs_diff = np.abs(diff)
        rmse = float(np.sqrt(np.nanmean(diff[ok] ** 2)))
        corr = float(np.corrcoef(ys[ok], x_hat[ok])[0, 1]) if ok.sum() > 1 else float("nan")

        if cfg.accuracy_denominator == "abs_truth":
            s = np.abs(x_hat) + EPS
            r = abs_diff / s
        else:
            d = denom[c]
            d = float(d) if (d is not None and d > EPS) else 1.0
            r = abs_diff / (d + EPS)

        point_acc = 1.0 - np.clip(r, 0.0, 1.0)
        sig_acc = np.nanmean(point_acc[ok]) if cfg.aggregate_policy == "mean" else np.nanmin(point_acc[ok])

        rows.append(
            {
                "signal": c,
                "n_points_used": int(ok.sum()),
                "matching_score": float(sig_acc),
                "rmse": rmse,
                "correlation": corr,
                "mean_abs_error": float(np.nanmean(abs_diff[ok])),
            }
        )
        acc_values.append(sig_acc)
        rmse_values.append(rmse)
        if np.isfinite(corr):
            corr_values.append(corr)

    detail = pd.DataFrame(rows).sort_values("matching_score", ascending=True)
    matching_score = float(np.nanmean(acc_values)) if acc_values else float("nan")
    rmse_avg = float(np.nanmean(rmse_values)) if rmse_values else float("nan")
    corr_avg = float(np.nanmean(corr_values)) if corr_values else float("nan")

    x_common = x_h[common_cols].to_numpy(dtype=float)
    y_common = y_l[common_cols].to_numpy(dtype=float)
    x_mean = np.nanmean(x_common, axis=1)
    y_mean = np.nanmean(y_common, axis=1)
    offset_ms = estimate_offset_ms(t_odg, x_mean, t_plant, y_mean)
    if not detail.empty:
        detail = detail.copy()
        detail["offset_ms"] = float(offset_ms)

    metrics = {
        "matching_score": matching_score,
        "rmse": rmse_avg,
        "correlation": corr_avg,
        "offset_ms": float(offset_ms),
    }
    return metrics, detail
