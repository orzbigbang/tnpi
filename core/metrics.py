import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time

from enum_types import SignalType
from core.models import CompareConfig
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
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    t0 = time.perf_counter()
    profile: Dict[str, float] = {}
    if plant_cache is not None:
        t_l, y_l = plant_cache
    else:
        t_l, y_l = parse_plant_samples_csv(plant_data_path, id_to_signal, encoding=plant_encoding)
    profile["parse_plant_ms"] = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    t_h, x_h = parse_odg_time_series_csv(odg_path, encoding=odg_encoding)
    profile["parse_odg_ms"] = (time.perf_counter() - t1) * 1000.0

    t_plant = t_l.to_numpy(dtype=float)
    t_odg = t_h.to_numpy(dtype=float)

    t2 = time.perf_counter()
    plant_order = np.argsort(t_plant)
    odg_order = np.argsort(t_odg)
    t_plant = t_plant[plant_order]
    t_odg = t_odg[odg_order]

    y_l = y_l.iloc[plant_order].reset_index(drop=True)
    x_h = x_h.iloc[odg_order].reset_index(drop=True)
    profile["sort_align_ms"] = (time.perf_counter() - t2) * 1000.0

    if time_range is not None:
        t3 = time.perf_counter()
        start, end = time_range
        if start > end:
            raise ValueError("Time range start must be <= end.")
        plant_mask = (t_plant >= start) & (t_plant <= end)
        odg_mask = (t_odg >= start) & (t_odg <= end)
        t_plant = t_plant[plant_mask]
        t_odg = t_odg[odg_mask]
        y_l = y_l.iloc[plant_mask].reset_index(drop=True)
        x_h = x_h.iloc[odg_mask].reset_index(drop=True)
        profile["range_filter_ms"] = (time.perf_counter() - t3) * 1000.0

    t4 = time.perf_counter()
    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between plant and odg CSV.")

    x_common = x_h[common_cols].to_numpy(dtype=float)
    y_common = y_l[common_cols].to_numpy(dtype=float)

    if cfg.accuracy_denominator == "range":
        denom_vals = np.nanmax(x_common, axis=0) - np.nanmin(x_common, axis=0)
    elif cfg.accuracy_denominator == "std":
        denom_vals = np.nanstd(x_common, axis=0)
    else:
        denom_vals = np.full(x_common.shape[1], np.nan, dtype=float)

    df_x = pd.DataFrame(x_common, columns=common_cols)
    df_x["t"] = t_odg
    df_x = df_x.sort_values("t")
    df_x = df_x.groupby("t", sort=True).first()
    t_unique = df_x.index.to_numpy(dtype=float)
    x_unique = df_x.to_numpy(dtype=float)

    x_hat_mat = np.full((t_plant.shape[0], x_unique.shape[1]), np.nan, dtype=float)
    interp_ok = np.zeros(x_unique.shape[1], dtype=bool)
    if t_unique.size >= 2:
        if cfg.out_of_range_policy == "clip":
            t_plant_clip = np.clip(t_plant, t_unique[0], t_unique[-1])
        else:
            t_plant_clip = np.clip(t_plant, t_unique[0], t_unique[-1])
            out_mask = (t_plant < t_unique[0]) | (t_plant > t_unique[-1])

        for i in range(x_unique.shape[1]):
            xu = x_unique[:, i]
            valid = np.isfinite(t_unique) & np.isfinite(xu)
            if valid.sum() < 2:
                continue
            th = t_unique[valid]
            xh = xu[valid]
            if len(th) < 2:
                continue
            x_hat = np.interp(t_plant_clip, th, xh)
            if cfg.out_of_range_policy != "clip":
                x_hat = x_hat.astype(float, copy=False)
                x_hat[out_mask] = np.nan
            x_hat_mat[:, i] = x_hat
            interp_ok[i] = True

    ok = np.isfinite(y_common) & np.isfinite(x_hat_mat)
    counts = ok.sum(axis=0)
    use_cols = interp_ok & (counts > 0)

    diff = y_common - x_hat_mat
    diff[~ok] = np.nan
    abs_diff = np.abs(diff)
    rmse_vals = np.sqrt(np.nanmean(diff ** 2, axis=0))
    mean_abs_err_vals = np.nanmean(abs_diff, axis=0)

    if cfg.accuracy_denominator == "abs_truth":
        s = np.abs(x_hat_mat) + EPS
        r = abs_diff / s
    else:
        d = np.where(np.isfinite(denom_vals) & (denom_vals > EPS), denom_vals, 1.0)
        r = abs_diff / (d + EPS)
    point_acc = 1.0 - np.clip(r, 0.0, 1.0)
    if cfg.aggregate_policy == "mean":
        sig_acc_vals = np.nanmean(point_acc, axis=0)
    else:
        sig_acc_vals = np.nanmin(point_acc, axis=0)

    y_masked = np.where(ok, y_common, np.nan)
    x_masked = np.where(ok, x_hat_mat, np.nan)
    mean_y = np.nanmean(y_masked, axis=0)
    mean_x = np.nanmean(x_masked, axis=0)
    yc = y_masked - mean_y
    xc = x_masked - mean_x
    denom_corr = np.maximum(counts - 1, 1)
    cov = np.nansum(yc * xc, axis=0) / denom_corr
    std_y = np.sqrt(np.nansum(yc * yc, axis=0) / denom_corr)
    std_x = np.sqrt(np.nansum(xc * xc, axis=0) / denom_corr)
    corr_vals = cov / (std_y * std_x)
    corr_vals[counts < 2] = np.nan

    rows = []
    acc_values = []
    rmse_values = []
    corr_values = []
    for i, c in enumerate(common_cols):
        if not use_cols[i]:
            continue
        rows.append(
            {
                "signal": c,
                "n_points_used": int(counts[i]),
                "matching_score": float(sig_acc_vals[i]),
                "rmse": float(rmse_vals[i]),
                "correlation": float(corr_vals[i]),
                "mean_abs_error": float(mean_abs_err_vals[i]),
            }
        )
        acc_values.append(float(sig_acc_vals[i]))
        rmse_values.append(float(rmse_vals[i]))
        if np.isfinite(corr_vals[i]):
            corr_values.append(float(corr_vals[i]))
    profile["per_signal_ms"] = (time.perf_counter() - t4) * 1000.0

    t5 = time.perf_counter()
    detail = pd.DataFrame(rows).sort_values("matching_score", ascending=True)
    matching_score = float(np.nanmean(acc_values)) if acc_values else float("nan")
    rmse_avg = float(np.nanmean(rmse_values)) if rmse_values else float("nan")
    corr_avg = float(np.nanmean(corr_values)) if corr_values else float("nan")

    x_mean = np.nanmean(x_common, axis=1)
    y_mean = np.nanmean(y_common, axis=1)
    offset_ms = estimate_offset_ms(t_odg, x_mean, t_plant, y_mean)
    if not detail.empty:
        detail = detail.copy()
        detail["offset_ms"] = float(offset_ms)
    profile["offset_estimate_ms"] = (time.perf_counter() - t5) * 1000.0
    profile["total_ms"] = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "matching_score": matching_score,
        "rmse": rmse_avg,
        "correlation": corr_avg,
        "offset_ms": float(offset_ms),
    }
    return metrics, detail, profile
