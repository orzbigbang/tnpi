import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time

from enum_types import SignalType
from core.models import CompareConfig
from core.parse import parse_odg_time_series_csv, parse_plant_samples_csv

EPS = 1e-12
OFFSET_SEARCH_MS = 120000
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


def _as_int64_ns(t_vals: np.ndarray) -> np.ndarray:
    t = np.asarray(t_vals)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype("int64")

    arr = np.asarray(t_vals, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.asarray([], dtype="int64")

    med = float(np.median(np.abs(finite)))
    if med < 1e11:
        scale = 1e9
    elif med < 1e15:
        scale = 1e6
    else:
        scale = 1.0

    return (arr * scale).round().astype("int64")


def _prepare_series(t_ns: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_ns, dtype="int64")
    x = np.asarray(v, dtype="float64")

    m = np.isfinite(x) & np.isfinite(t.astype("float64"))
    t = t[m]
    x = x[m]
    if t.size == 0:
        return t, x

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    t_rev = t[::-1]
    x_rev = x[::-1]
    _, idx = np.unique(t_rev, return_index=True)
    keep_rev = np.sort(idx)
    t_u = t_rev[keep_rev][::-1]
    x_u = x_rev[keep_rev][::-1]
    return t_u, x_u


def _interp_to_grid(t_ns: np.ndarray, v: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    t_u, x_u = _prepare_series(t_ns, v)
    if t_u.size < 2:
        return np.full(grid_ns.shape, np.nan, dtype="float64")

    left = t_u[0]
    right = t_u[-1]
    g_clip = np.clip(grid_ns, left, right)
    y = np.interp(g_clip.astype("float64"), t_u.astype("float64"), x_u.astype("float64")).astype("float64")

    out = (grid_ns < left) | (grid_ns > right)
    y[out] = np.nan
    return y


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fft_xcorr_best_lag(a: np.ndarray, b: np.ndarray, max_shift: int, min_points: int) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.size != b.size or a.size == 0:
        return float("nan")

    ma = np.isfinite(a)
    mb = np.isfinite(b)
    if int(ma.sum()) < min_points or int(mb.sum()) < min_points:
        return float("nan")

    def z(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        xv = x[m]
        mu = float(np.mean(xv)) if xv.size else 0.0
        sd = float(np.std(xv)) if xv.size else 0.0
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        y = (x - mu) / sd
        y[~m] = 0.0
        return y

    a0 = z(a, ma)
    b0 = z(b, mb)

    am = ma.astype("float64")
    bm = mb.astype("float64")

    n = a0.size
    size = _next_pow2(2 * n - 1)

    A = np.fft.rfft(a0, size)
    B = np.fft.rfft(b0, size)
    AM = np.fft.rfft(am, size)
    BM = np.fft.rfft(bm, size)

    corr = np.fft.irfft(A * np.conj(B), size)[: 2 * n - 1]
    overlap = np.fft.irfft(AM * np.conj(BM), size)[: 2 * n - 1]

    lags = np.arange(-(n - 1), n, dtype="int64")
    w = (lags >= -max_shift) & (lags <= max_shift)
    if not np.any(w):
        return float("nan")

    corr_w = corr[w]
    ov_w = overlap[w]
    lags_w = lags[w]

    ok = ov_w >= float(min_points)
    if not np.any(ok):
        return float("nan")

    score = corr_w[ok] / (ov_w[ok] + EPS)
    best = int(np.argmax(score))
    return float(lags_w[ok][best])


def estimate_offset_ms(
    t1_ns: np.ndarray,
    v1: np.ndarray,
    t2_ns: np.ndarray,
    v2: np.ndarray,
    grid_ms: int = OFFSET_GRID_MS,
    search_ms: int = OFFSET_SEARCH_MS,
) -> float:
    t1i = _as_int64_ns(t1_ns)
    t2i = _as_int64_ns(t2_ns)

    v1 = np.asarray(v1, dtype="float64")
    v2 = np.asarray(v2, dtype="float64")

    t1u, v1u = _prepare_series(t1i, v1)
    t2u, v2u = _prepare_series(t2i, v2)
    if t1u.size < 2 or t2u.size < 2:
        return float("nan")

    step_ns = int(grid_ms) * 1_000_000
    search_ns = int(search_ms) * 1_000_000

    t1_min, t1_max = int(t1u[0]), int(t1u[-1])
    t2_min, t2_max = int(t2u[0]), int(t2u[-1])

    start = max(t1_min, t2_min - search_ns)
    end = min(t1_max, t2_max + search_ns)
    if start >= end:
        return float("nan")

    n_steps = int((end - start) // step_ns) + 1
    if n_steps < OFFSET_MIN_POINTS:
        return float("nan")

    grid = start + np.arange(n_steps, dtype="int64") * step_ns

    a = _interp_to_grid(t1u, v1u, grid)
    b = _interp_to_grid(t2u, v2u, grid)

    max_shift = int(search_ms // grid_ms)
    best_lag = _fft_xcorr_best_lag(a, b, max_shift=max_shift, min_points=OFFSET_MIN_POINTS)
    if not np.isfinite(best_lag):
        return float("nan")

    return float(best_lag * grid_ms)


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

    t_plant_ns = _as_int64_ns(t_plant)
    t_odg_ns = _as_int64_ns(t_odg)
    t_plant_f = t_plant_ns.astype(float)
    t_odg_f = t_odg_ns.astype(float)

    if time_range is not None:
        t3 = time.perf_counter()
        start, end = time_range
        if start > end:
            raise ValueError("Time range start must be <= end.")
        start_ns = _as_int64_ns(np.asarray([start], dtype="float64"))[0]
        end_ns = _as_int64_ns(np.asarray([end], dtype="float64"))[0]
        plant_mask = (t_plant_ns >= start_ns) & (t_plant_ns <= end_ns)
        odg_mask = (t_odg_ns >= start_ns) & (t_odg_ns <= end_ns)
        t_plant_ns = t_plant_ns[plant_mask]
        t_odg_ns = t_odg_ns[odg_mask]
        t_plant_f = t_plant_ns.astype(float)
        t_odg_f = t_odg_ns.astype(float)
        y_l = y_l.iloc[plant_mask].reset_index(drop=True)
        x_h = x_h.iloc[odg_mask].reset_index(drop=True)
        profile["range_filter_ms"] = (time.perf_counter() - t3) * 1000.0

    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between plant and odg CSV.")

    denom: Dict[str, Optional[float]] = {}
    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=float)
        if cfg.accuracy_denominator == "range":
            d = np.nanmax(xs) - np.nanmin(xs)
        elif cfg.accuracy_denominator == "std":
            d = np.nanstd(xs)
        else:
            d = None
        denom[c] = d

    t4 = time.perf_counter()
    x_common = x_h[common_cols].to_numpy(dtype=float)
    y_common = y_l[common_cols].to_numpy(dtype=float)
    x_mean = np.nanmean(x_common, axis=1)
    y_mean = np.nanmean(y_common, axis=1)
    offset_ms = estimate_offset_ms(t_odg_ns, x_mean, t_plant_ns, y_mean)

    t_plant_aligned_f = t_plant_f.copy()
    if np.isfinite(offset_ms):
        t_plant_aligned_f = t_plant_f - float(offset_ms) * 1_000_000.0
    profile["offset_estimate_ms"] = (time.perf_counter() - t4) * 1000.0

    t5 = time.perf_counter()
    rows = []
    acc_values = []
    rmse_values = []
    corr_values = []

    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=float)
        ys = y_l[c].to_numpy(dtype=float)

        valid = np.isfinite(t_odg_f) & np.isfinite(xs)
        th = t_odg_f[valid]
        xh = xs[valid]
        if th.size < 2:
            continue

        df_tmp = pd.DataFrame({"t": th, "x": xh}).drop_duplicates(subset=["t"]).sort_values("t")
        th = df_tmp["t"].to_numpy(dtype=float)
        xh = df_tmp["x"].to_numpy(dtype=float)
        if th.size < 2:
            continue

        x_hat = linear_interp_truth(th, xh, t_plant_aligned_f, cfg.out_of_range_policy)

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
                "offset_ms": float(offset_ms),
            }
        )
        acc_values.append(float(sig_acc))
        rmse_values.append(float(rmse))
        if np.isfinite(corr):
            corr_values.append(float(corr))

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values("matching_score", ascending=True)

    matching_score = float(np.nanmean(acc_values)) if acc_values else float("nan")
    rmse_avg = float(np.nanmean(rmse_values)) if rmse_values else float("nan")
    corr_avg = float(np.nanmean(corr_values)) if corr_values else float("nan")
    profile["per_signal_ms"] = (time.perf_counter() - t5) * 1000.0
    profile["total_ms"] = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "matching_score": matching_score,
        "rmse": rmse_avg,
        "correlation": corr_avg,
        "offset_ms": float(offset_ms),
    }
    return metrics, detail, profile
