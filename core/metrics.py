# === 完整替换块：metrics.py（自适应权重 + corr只统计非平坦信号 + 输出nonflat汇总）===

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time

from numpy.fft import rfft, irfft


def next_fast_len(n):
    """返回下一个 2 的幂次方，用于 FFT 优化"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


from enum_types import SignalType
from core.models import CompareConfig
from core.parse import parse_odg_time_series_csv, parse_plant_samples_csv, parse_plant_samples_csv_chunked


# ---------------- dtype knobs ----------------
F = np.float32
TF = np.float64
TINT = np.int64

EPS = 1e-12
OFFSET_SEARCH_MS = 120000
OFFSET_GRID_MS = 50
OFFSET_MIN_POINTS = 30

# 融合权重（非平坦信号）
W_TREND = 0.7
W_ERROR = 0.3

# 常数/近常数信号判定阈值（绝对std阈值）
FLAT_STD_TH = 1e-6


def linear_interp_truth(
    t_odg: np.ndarray,
    x_odg: np.ndarray,
    t_plant: np.ndarray,
    out_of_range_policy: str,
) -> np.ndarray:
    t_odg = np.asarray(t_odg, dtype=TF)
    x_odg = np.asarray(x_odg, dtype=F)
    t_plant = np.asarray(t_plant, dtype=TF)

    if t_odg.size == 0:
        return np.asarray([], dtype=F)

    if out_of_range_policy == "clip":
        t_plant2 = np.clip(t_plant, t_odg[0], t_odg[-1])
        return np.interp(t_plant2, t_odg, x_odg).astype(F)

    x_hat = np.interp(np.clip(t_plant, t_odg[0], t_odg[-1]), t_odg, x_odg).astype(F)
    mask = (t_plant < t_odg[0]) | (t_plant > t_odg[-1])
    x_hat[mask] = np.nan
    return x_hat


def _as_int64_ns(t_vals: np.ndarray) -> np.ndarray:
    t = np.asarray(t_vals)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype(TINT)

    arr = np.asarray(t_vals, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.asarray([], dtype=TINT)

    med = float(np.median(np.abs(finite)))
    if med < 1e11:
        scale = 1e9
    elif med < 1e15:
        scale = 1e6
    else:
        scale = 1.0

    return (arr * scale).round().astype(TINT)


def _prepare_series(t_ns: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_ns, dtype=TINT)
    x = np.asarray(v, dtype=F)

    m = np.isfinite(x)
    t = t[m]
    x = x[m]
    if t.size == 0:
        return t, x

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # 去重：保留最后一次出现
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
        return np.full(grid_ns.shape, np.nan, dtype=F)

    left = t_u[0]
    right = t_u[-1]
    g_clip = np.clip(grid_ns, left, right)

    y = np.interp(
        g_clip.astype(TF),
        t_u.astype(TF),
        x_u.astype(TF),
    ).astype(F)

    out = (grid_ns < left) | (grid_ns > right)
    y[out] = np.nan
    return y


def _fft_xcorr_best_lag(a: np.ndarray, b: np.ndarray, max_shift: int, min_points: int) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != b.size or a.size == 0:
        return float("nan")

    ma = np.isfinite(a)
    mb = np.isfinite(b)
    if int(ma.sum()) < min_points or int(mb.sum()) < min_points:
        return float("nan")

    def z(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        xv = x[m]
        mu = np.float32(xv.mean()) if xv.size else np.float32(0.0)
        sd = np.float32(xv.std()) if xv.size else np.float32(0.0)
        if not np.isfinite(sd) or sd < np.float32(1e-6):
            sd = np.float32(1.0)
        y = (x - mu) / sd
        y = y.astype(np.float32, copy=False)
        y[~m] = np.float32(0.0)
        return y

    a0 = z(a, ma)
    b0 = z(b, mb)
    am = ma.astype(np.float32)
    bm = mb.astype(np.float32)

    n = a0.size
    size = next_fast_len(2 * n - 1)

    a0_contig = np.ascontiguousarray(a0, dtype=np.float32)
    b0_contig = np.ascontiguousarray(b0, dtype=np.float32)
    am_contig = np.ascontiguousarray(am, dtype=np.float32)
    bm_contig = np.ascontiguousarray(bm, dtype=np.float32)

    A = np.fft.rfft(a0_contig, n=size)
    B = np.fft.rfft(b0_contig, n=size)
    AM = np.fft.rfft(am_contig, n=size)
    BM = np.fft.rfft(bm_contig, n=size)

    prod1 = (A * np.conj(B)).astype(np.complex64, copy=False)
    prod2 = (AM * np.conj(BM)).astype(np.complex64, copy=False)

    corr = np.fft.irfft(prod1, n=size).astype(np.float32)[: 2 * n - 1]
    overlap = np.fft.irfft(prod2, n=size).astype(np.float32)[: 2 * n - 1]

    lags = np.arange(-(n - 1), n, dtype=np.int64)
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
    window_ms: int = 5_000,
    min_windows: int = 3,
) -> float:
    t1i = _as_int64_ns(t1_ns)
    t2i = _as_int64_ns(t2_ns)

    v1 = np.asarray(v1, dtype=F)
    v2 = np.asarray(v2, dtype=F)

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

    min_len_ns = (OFFSET_MIN_POINTS - 1) * step_ns
    if window_ns < min_len_ns:
        window_ns = min_len_ns

    lags_ms: list[float] = []
    skipped_windows = 0
    cur = start

    while cur < end:
        MAX_STEPS = 100_000
        w_end = min(cur + window_ns, end)

        span = int(w_end - cur)
        eff_step_ns = max(step_ns, (span + (MAX_STEPS - 2)) // (MAX_STEPS - 1))
        n_eff = span // eff_step_ns + 1

        if n_eff > MAX_STEPS:
            eff_step_ns = (span + (MAX_STEPS - 2)) // (MAX_STEPS - 1)
            n_eff = span // eff_step_ns + 1

        if n_eff > MAX_STEPS:
            skipped_windows += 1
            cur = w_end
            continue

        if n_eff < OFFSET_MIN_POINTS:
            cur = w_end
            continue

        eff_grid_ms = eff_step_ns / 1_000_000.0
        grid = cur + np.arange(n_eff, dtype=TINT) * eff_step_ns

        a = _interp_to_grid(t1u, v1u, grid)
        b = _interp_to_grid(t2u, v2u, grid)

        max_shift = int(search_ms // eff_grid_ms)
        lag = _fft_xcorr_best_lag(a, b, max_shift=max_shift, min_points=OFFSET_MIN_POINTS)
        if np.isfinite(lag):
            lags_ms.append(float(lag * eff_grid_ms))

        del grid, a, b
        cur = w_end

    if skipped_windows > 0:
        print(f"⚠️  Warning: Skipped {skipped_windows} oversized windows (>{MAX_STEPS} points)")

    if len(lags_ms) < min_windows:
        return float("nan")

    return float(np.median(np.asarray(lags_ms, dtype=np.float32)))


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

    # ---------- 1) 先读 ODG ----------
    t1 = time.perf_counter()
    t_h, x_h = parse_odg_time_series_csv(odg_path, encoding=odg_encoding)
    profile["parse_odg_ms"] = (time.perf_counter() - t1) * 1000.0

    if time_range is None:
        if len(t_h) >= 2:
            time_range = (float(t_h.iloc[0]), float(t_h.iloc[-1]))

    # ---------- 2) 读 PlantDB ----------
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

    if len(t_l) < 2 or y_l.empty or len(t_h) < 2 or x_h.empty:
        metrics = {
            "matching_score": float("nan"),
            "rmse": float("nan"),
            "correlation": float("nan"),               # 非平坦corr均值
            "correlation_all": float("nan"),           # 全信号corr均值（可选）
            "matching_score_nonflat": float("nan"),    # 非平坦matching均值（可选）
            "flat_ratio": float("nan"),                # 平坦信号占比（可选）
            "offset_ms": float("nan"),
        }
        profile["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return metrics, pd.DataFrame(), profile

    # ---------- 3) 排序对齐 ----------
    t_plant = t_l.to_numpy(dtype=np.float64)
    t_odg = t_h.to_numpy(dtype=np.float64)

    t3 = time.perf_counter()
    plant_order = np.argsort(t_plant)
    odg_order = np.argsort(t_odg)
    t_plant = t_plant[plant_order]
    t_odg = t_odg[odg_order]
    y_l = y_l.iloc[plant_order].reset_index(drop=True)
    x_h = x_h.iloc[odg_order].reset_index(drop=True)
    profile["sort_align_ms"] = (time.perf_counter() - t3) * 1000.0

    # ---------- 4) ns(int64) ----------
    t_plant_ns = _as_int64_ns(t_plant)
    t_odg_ns = _as_int64_ns(t_odg)

    t_plant_f = t_plant_ns.astype(np.float64)
    t_odg_f = t_odg_ns.astype(np.float64)

    # ---------- 5) 列交集 ----------
    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between plant and odg CSV.")

    # ---------- 6) denom ----------
    denom: Dict[str, Optional[float]] = {}
    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=np.float32)
        if cfg.accuracy_denominator == "range":
            d = float(np.nanmax(xs) - np.nanmin(xs))
        elif cfg.accuracy_denominator == "std":
            d = float(np.nanstd(xs))
        else:
            d = None
        denom[c] = d

    # ---------- 7) 估 offset ----------
    t4 = time.perf_counter()
    K = 64
    step = max(1, len(common_cols) // K)
    cols_for_offset = common_cols[::step]

    x_mean = x_h[cols_for_offset].astype("float32").mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    y_mean = y_l[cols_for_offset].astype("float32").mean(axis=1, skipna=True).to_numpy(dtype=np.float32)

    offset_ms = estimate_offset_ms(
        t_odg_ns,
        x_mean,
        t_plant_ns,
        y_mean,
        grid_ms=200,
        search_ms=OFFSET_SEARCH_MS,
        window_ms=5_000,
    )
    profile["offset_estimate_ms"] = (time.perf_counter() - t4) * 1000.0

    t_plant_aligned_f = t_plant_f
    if np.isfinite(offset_ms):
        t_plant_aligned_f = t_plant_f - float(offset_ms) * 1_000_000.0  # ms -> ns

    # ---------- 8) 逐信号算指标 ----------
    t5 = time.perf_counter()
    rows = []

    # 全体（用于matching_score / rmse）
    score_values_all = []
    rmse_values_all = []

    # corr两套：all / nonflat
    corr_values_all = []
    corr_values_nonflat = []

    # 非平坦matching（可选输出）
    score_values_nonflat = []

    n_flat = 0
    n_used = 0

    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=np.float32)
        ys = y_l[c].to_numpy(dtype=np.float32)

        valid = np.isfinite(t_odg_f) & np.isfinite(xs.astype(np.float64))
        th = t_odg_f[valid]
        xh = xs[valid]
        if th.size < 2:
            continue

        df_tmp = pd.DataFrame({"t": th, "x": xh}).drop_duplicates(subset=["t"]).sort_values("t")
        th = df_tmp["t"].to_numpy(dtype=np.float64)
        xh = df_tmp["x"].to_numpy(dtype=np.float32)
        if th.size < 2:
            continue

        x_hat = linear_interp_truth(th, xh, t_plant_aligned_f, cfg.out_of_range_policy)

        ok = np.isfinite(ys) & np.isfinite(x_hat)
        n_ok = int(ok.sum())
        if n_ok == 0:
            continue

        n_used += 1

        diff = (ys - x_hat).astype(np.float32, copy=False)
        abs_diff = np.abs(diff).astype(np.float32, copy=False)

        rmse = float(np.sqrt(np.nanmean((diff[ok] * diff[ok]).astype(np.float32))))
        corr = (
            float(np.corrcoef(ys[ok].astype(np.float64), x_hat[ok].astype(np.float64))[0, 1])
            if n_ok > 1
            else float("nan")
        )

        # 误差分（0~1）
        if cfg.accuracy_denominator == "abs_truth":
            s = np.abs(x_hat).astype(np.float32, copy=False) + np.float32(EPS)
            r = abs_diff / s
        else:
            d = denom[c]
            d = float(d) if (d is not None and d > EPS) else 1.0
            r = abs_diff / (np.float32(d) + np.float32(EPS))

        point_acc = 1.0 - np.clip(r, 0.0, 1.0)
        sig_acc = (
            float(np.nanmean(point_acc[ok])) if cfg.aggregate_policy == "mean"
            else float(np.nanmin(point_acc[ok]))
        )
        error_score = float(sig_acc)

        # 平坦判定
        std_truth = float(np.nanstd(x_hat[ok].astype(np.float64)))
        std_plant = float(np.nanstd(ys[ok].astype(np.float64)))
        flat = (std_truth < FLAT_STD_TH) or (std_plant < FLAT_STD_TH)
        if flat:
            n_flat += 1

        # 趋势分/融合分
        if flat:
            trend_score = float("nan")
            sig_score = error_score
        else:
            if np.isfinite(corr):
                trend_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
                sig_score = W_TREND * trend_score + W_ERROR * error_score
            else:
                trend_score = float("nan")
                sig_score = error_score

        # detail输出
        rows.append(
            {
                "signal": c,
                "n_points_used": n_ok,
                "matching_score": float(sig_score),
                "trend_score": float(trend_score),
                "error_score": float(error_score),
                "rmse": rmse,
                "correlation": corr,
                "mean_abs_error": float(np.nanmean(abs_diff[ok].astype(np.float32))),
                "offset_ms": float(offset_ms),
                "is_flat": bool(flat),
                "std_truth": float(std_truth),
                "std_plant": float(std_plant),
            }
        )

        # 汇总
        score_values_all.append(float(sig_score))
        rmse_values_all.append(float(rmse))

        if np.isfinite(corr):
            corr_values_all.append(float(corr))
            if not flat:
                corr_values_nonflat.append(float(corr))

        if not flat:
            score_values_nonflat.append(float(sig_score))

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values("matching_score", ascending=True)

    matching_score = float(np.nanmean(score_values_all)) if score_values_all else float("nan")
    rmse_avg = float(np.nanmean(rmse_values_all)) if rmse_values_all else float("nan")

    # ✅ 核心：UI建议展示这个（非平坦信号的corr均值）
    corr_avg_nonflat = float(np.nanmean(corr_values_nonflat)) if corr_values_nonflat else float("nan")

    # 可选：保留一个“全体corr均值”用于调试
    corr_avg_all = float(np.nanmean(corr_values_all)) if corr_values_all else float("nan")

    # 可选：非平坦matching均值（更直观）
    matching_score_nonflat = float(np.nanmean(score_values_nonflat)) if score_values_nonflat else float("nan")

    flat_ratio = (float(n_flat) / float(n_used)) if n_used > 0 else float("nan")

    profile["per_signal_ms"] = (time.perf_counter() - t5) * 1000.0
    profile["total_ms"] = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "matching_score": matching_score,
        "rmse": rmse_avg,
        "correlation": corr_avg_nonflat,          # ✅ 建议UI显示这个
        "correlation_all": corr_avg_all,          # 可选
        "matching_score_nonflat": matching_score_nonflat,  # 可选
        "flat_ratio": flat_ratio,                 # 可选：解释“为什么corr低”
        "offset_ms": float(offset_ms),
    }
    return metrics, detail, profile