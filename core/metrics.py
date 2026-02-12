import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time

from enum_types import SignalType
from core.models import CompareConfig
from core.parse import parse_odg_time_series_csv, parse_plant_samples_csv, parse_plant_samples_csv_chunked

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
    *,
    window_ms: int = 30_000,          # 30s 一窗（可调：10_000~60_000）
    min_windows: int = 1,             # 至少要有多少个窗算出结果
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
    window_ns = int(window_ms) * 1_000_000

    t1_min, t1_max = int(t1u[0]), int(t1u[-1])
    t2_min, t2_max = int(t2u[0]), int(t2u[-1])

    start = max(t1_min, t2_min - search_ns)
    end = min(t1_max, t2_max + search_ns)
    if start >= end:
        return float("nan")

    # 单窗至少要能形成足够点
    min_len_ns = (OFFSET_MIN_POINTS - 1) * step_ns
    if window_ns < min_len_ns:
        window_ns = min_len_ns

    max_shift = int(search_ms // grid_ms)

    lags = []
    lags_ms: list[float] = []
    cur = start

    while cur < end:
        MAX_STEPS = 2_000_000  # 或 500_000

        w_end = min(cur + window_ns, end)

        # 反推 eff_step_ns，保证 n_eff <= MAX_STEPS
        span = int(w_end - cur)  # ns
        eff_step_ns = max(step_ns, (span + (MAX_STEPS - 2)) // (MAX_STEPS - 1))
        # 上面等价于 ceil(span / (MAX_STEPS-1))，但用整数避免 float

        n_eff = span // eff_step_ns + 1
        if n_eff > MAX_STEPS:
            # 极端情况下再兜底一次（理论上不会进来）
            eff_step_ns = (span + (MAX_STEPS - 2)) // (MAX_STEPS - 1)
            n_eff = span // eff_step_ns + 1

        if n_eff > MAX_STEPS:
            raise RuntimeError(f"n_eff too large: {n_eff} > {MAX_STEPS}")

        if n_eff < OFFSET_MIN_POINTS:
            cur = w_end
            continue

        eff_grid_ms = eff_step_ns / 1_000_000.0

        grid = cur + np.arange(n_eff, dtype="int64") * eff_step_ns

        a = _interp_to_grid(t1u, v1u, grid)
        b = _interp_to_grid(t2u, v2u, grid)

        max_shift = int(search_ms // eff_grid_ms)
        lag = _fft_xcorr_best_lag(a, b, max_shift=max_shift, min_points=OFFSET_MIN_POINTS)
        if np.isfinite(lag):
            lags_ms.append(float(lag * eff_grid_ms))

        del grid, a, b
        cur = w_end

    if len(lags) < min_windows:
        return float("nan")

    # 多窗取中位数更稳
    best_lag = float(np.median(np.asarray(lags, dtype="float64")))
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

    # ---------- 1) 先读 ODG（通常更小），并决定默认 time_range ----------
    t1 = time.perf_counter()
    t_h, x_h = parse_odg_time_series_csv(odg_path, encoding=odg_encoding)
    profile["parse_odg_ms"] = (time.perf_counter() - t1) * 1000.0

    if time_range is None:
        if len(t_h) >= 2:
            time_range = (float(t_h.iloc[0]), float(t_h.iloc[-1]))

    # ---------- 2) 读 PlantDB（必须 chunk + 读取阶段 time_range 过滤） ----------
    t2 = time.perf_counter()
    if plant_cache is not None:
        t_l, y_l = plant_cache
    else:
        t_l, y_l = parse_plant_samples_csv_chunked(
            plant_data_path,
            id_to_signal,
            encoding=plant_encoding,
            time_range=time_range,
            chunksize=2_000_000,
        )
    profile["parse_plant_ms"] = (time.perf_counter() - t2) * 1000.0

    # 过滤后没有数据
    if len(t_l) < 2 or y_l.empty or len(t_h) < 2 or x_h.empty:
        metrics = {
            "matching_score": float("nan"),
            "rmse": float("nan"),
            "correlation": float("nan"),
            "offset_ms": float("nan"),
        }
        profile["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return metrics, pd.DataFrame(), profile

    # ---------- 3) 排序对齐 ----------
    t_plant = t_l.to_numpy(dtype=float)
    t_odg = t_h.to_numpy(dtype=float)

    t3 = time.perf_counter()
    plant_order = np.argsort(t_plant)
    odg_order = np.argsort(t_odg)
    t_plant = t_plant[plant_order]
    t_odg = t_odg[odg_order]
    y_l = y_l.iloc[plant_order].reset_index(drop=True)
    x_h = x_h.iloc[odg_order].reset_index(drop=True)
    profile["sort_align_ms"] = (time.perf_counter() - t3) * 1000.0

    # ---------- 4) ns(int64) + float（给插值/比较用） ----------
    t_plant_ns = _as_int64_ns(t_plant)
    t_odg_ns = _as_int64_ns(t_odg)
    t_plant_f = t_plant_ns.astype(float)
    t_odg_f = t_odg_ns.astype(float)

    # ---------- 5) 列交集 ----------
    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between plant and odg CSV.")

    # ---------- 6) 预先算 denom（逐列，不建大矩阵） ----------
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

    # ---------- 7) 估 offset（抽样少量列，避免全列 to_numpy 巨矩阵） ----------
    t4 = time.perf_counter()

    K = 64
    step = max(1, len(common_cols) // K)
    cols_for_offset = common_cols[::step]

    x_mean = x_h[cols_for_offset].astype("float32").mean(axis=1, skipna=True).to_numpy("float64")
    y_mean = y_l[cols_for_offset].astype("float32").mean(axis=1, skipna=True).to_numpy("float64")

    offset_ms = estimate_offset_ms(
        t_odg_ns,
        x_mean,
        t_plant_ns,
        y_mean,
        grid_ms=200,
        search_ms=OFFSET_SEARCH_MS,
        window_ms=30_000,
    )
    profile["offset_estimate_ms"] = (time.perf_counter() - t4) * 1000.0

    t_plant_aligned_f = t_plant_f
    if np.isfinite(offset_ms):
        t_plant_aligned_f = t_plant_f - float(offset_ms) * 1_000_000.0  # ms -> ns

    # ---------- 8) 逐信号算指标 ----------
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
        if int(ok.sum()) == 0:
            continue

        diff = ys - x_hat
        abs_diff = np.abs(diff)

        rmse = float(np.sqrt(np.nanmean(diff[ok] ** 2)))
        corr = float(np.corrcoef(ys[ok], x_hat[ok])[0, 1]) if int(ok.sum()) > 1 else float("nan")

        if cfg.accuracy_denominator == "abs_truth":
            s = np.abs(x_hat) + EPS
            r = abs_diff / s
        else:
            d = denom[c]
            d = float(d) if (d is not None and d > EPS) else 1.0
            r = abs_diff / (d + EPS)

        point_acc = 1.0 - np.clip(r, 0.0, 1.0)
        sig_acc = float(np.nanmean(point_acc[ok])) if cfg.aggregate_policy == "mean" else float(np.nanmin(point_acc[ok]))

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
