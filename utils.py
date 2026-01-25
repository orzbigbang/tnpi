import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from state import CompareConfig


DEFAULT_TIME_COL_CANDIDATES = ["timestamp", "time", "ts", "date_time", "datetime", "t"]
DATE_TIME_COL_CANDIDATES = {
    "date": ["date"],
    "time": ["time"],
    "ms": ["milli sec", "millisec", "millisecond", "msec", "ms"],
}
EPS = 1e-12
OFFSET_SEARCH_MS = 12000
OFFSET_GRID_MS = 50
OFFSET_MIN_POINTS = 30


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DEFAULT_TIME_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def parse_time_values(series: pd.Series) -> pd.Series:
    t_num = pd.to_numeric(series, errors="coerce")
    if t_num.notna().all():
        return t_num.astype(float)
    t_dt = pd.to_datetime(series, errors="coerce")
    if t_dt.notna().all():
        return t_dt.view("int64").astype(float)  # ns
    raise ValueError("Time column is neither numeric nor datetime-parsable.")


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", errors="replace")


def read_csv_head(file_bytes: bytes, nrows: int = 1) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, nrows=nrows)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", errors="replace", nrows=nrows)


def normalize_id_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    num = pd.to_numeric(series, errors="coerce")
    mask = num.notna()
    if mask.any():
        int_vals = num[mask].round(0).astype("Int64").astype(str)
        s = s.copy()
        s.loc[mask] = int_vals.values
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def parse_time_series_csv(file_bytes: bytes) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_bytes(file_bytes)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["date"] if c in cols_lower), None)
    time_col2 = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["time"] if c in cols_lower), None)
    ms_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["ms"] if c in cols_lower), None)
    if date_col and time_col2:
        dt = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col2].astype(str),
            errors="coerce",
        )
        if ms_col:
            ms = pd.to_numeric(df[ms_col], errors="coerce")
            dt = dt + pd.to_timedelta(ms.fillna(0), unit="ms")
        if dt.notna().all():
            t_vals = dt.view("int64").astype(float)  # ns
            drop_cols = [date_col, time_col2] + ([ms_col] if ms_col else [])
            signals = df.drop(columns=drop_cols)
            signals = signals.apply(pd.to_numeric, errors="coerce")
            return pd.Series(t_vals, name="timestamp"), signals

    time_col = detect_time_col(df)
    if time_col is not None:
        t_vals = parse_time_values(df[time_col])
        signals = df.drop(columns=[time_col])
        signals = signals.apply(pd.to_numeric, errors="coerce")
        return pd.Series(t_vals, name=time_col), signals

    raise ValueError(
        f"Cannot detect time column. Please include one of {DEFAULT_TIME_COL_CANDIDATES} "
        "or date+time columns (optionally with millisecond)."
    )


def parse_high_time_series_csv(file_bytes: bytes) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_bytes(file_bytes)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["date"] if c in cols_lower), None)
    time_col2 = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["time"] if c in cols_lower), None)
    ms_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["ms"] if c in cols_lower), None)
    if not (date_col and time_col2):
        raise ValueError("ODG CSV must include Date and Time columns.")

    dt = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col2].astype(str),
        errors="coerce",
    )
    if ms_col:
        ms = pd.to_numeric(df[ms_col], errors="coerce")
        dt = dt + pd.to_timedelta(ms.fillna(0), unit="ms")
    if not dt.notna().all():
        raise ValueError("ODG Date/Time columns are not datetime-parsable.")

    t_vals = dt.view("int64").astype(float)  # ns
    drop_cols = [date_col, time_col2] + ([ms_col] if ms_col else [])
    signals = df.drop(columns=drop_cols)
    signals = signals.apply(pd.to_numeric, errors="coerce")
    return pd.Series(t_vals, name="timestamp"), signals


def _pick_column_by_name(columns: List[str], keys: Tuple[str, ...]) -> Optional[str]:
    for col in columns:
        name = col.strip().lower()
        if any(key in name for key in keys):
            return col
    return None


def parse_low_mapping_csv(map_bytes: bytes) -> Dict[str, str]:
    df = read_csv_bytes(map_bytes)
    if df.shape[1] < 2:
        raise ValueError("PlantDB CSV #1 must have at least two columns: ID and signal name.")
    col_names = [str(c) for c in df.columns]
    id_col = _pick_column_by_name(col_names, ("id", "signalid", "tagid"))
    signal_col = _pick_column_by_name(col_names, ("signal", "tag", "name"))
    if id_col is None or signal_col is None:
        id_col = df.columns[0]
        signal_col = df.columns[1]
    ids = normalize_id_series(df[id_col])
    signals = df[signal_col].astype(str)
    mask = ids.notna() & signals.notna()
    mapping = dict(zip(ids[mask], signals[mask]))
    if not mapping:
        raise ValueError("PlantDB CSV #1 has no valid ID-to-signal mappings.")
    return mapping


def parse_low_samples_csv(
    data_bytes: bytes,
    id_to_signal: Dict[str, str],
) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_bytes(data_bytes)
    if df.shape[1] < 3:
        raise ValueError("PlantDB CSV #2 must have at least three columns: time, ID, value.")
    time_col = df.columns[0]
    id_col = df.columns[1]
    value_col = df.columns[2]

    t_vals = parse_time_values(df[time_col])
    ids = normalize_id_series(df[id_col])
    signals = ids.map(id_to_signal)
    values = pd.to_numeric(df[value_col], errors="coerce")

    tmp = pd.DataFrame({"t": t_vals, "signal": signals, "value": values})
    tmp = tmp.dropna(subset=["t", "signal", "value"])
    if tmp.empty:
        raise ValueError("PlantDB CSV #2 has no usable rows after ID mapping.")

    wide = tmp.pivot_table(index="t", columns="signal", values="value", aggfunc="mean")
    wide = wide.sort_index()
    t_series = pd.Series(wide.index.to_numpy(dtype=float), name="timestamp")
    return t_series, wide.reset_index(drop=True)


def load_high_folder_from_zip(zip_bytes: bytes) -> Dict[str, bytes]:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    out = {}
    for name in z.namelist():
        if name.lower().endswith(".csv") and not name.endswith("/"):
            out[name] = z.read(name)
    if not out:
        raise ValueError("No CSV files found in zip.")
    return out


def load_high_folder_from_dir(dir_path: str) -> Dict[str, bytes]:
    if not os.path.isdir(dir_path):
        raise ValueError("ODG folder does not exist.")
    out = {}
    for name in os.listdir(dir_path):
        if name.lower().endswith(".csv"):
            full_path = os.path.join(dir_path, name)
            if os.path.isfile(full_path):
                with open(full_path, "rb") as f:
                    out[name] = f.read()
    if not out:
        raise ValueError("No CSV files found in folder.")
    return out


def _classify_low_csv_head(head: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip().lower() for c in head.columns]
    if not cols:
        return None
    has_id = any("n_monitoring_id" in c for c in cols)
    has_signal = any("s_signal_name1" in c for c in cols)
    has_time = any("t_sample_time" in c for c in cols)
    has_value = any("n_signal1" in c for c in cols)
    if has_id and has_signal and not (has_time and has_value):
        return "map"
    if has_time and has_id and has_value:
        return "data"
    return None


def load_low_folder_from_dir(dir_path: str) -> Tuple[bytes, bytes, int, int, int]:
    if not os.path.isdir(dir_path):
        raise ValueError("PlantDB folder does not exist.")

    map_files = []
    data_files = []
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".csv"):
            continue
        full_path = os.path.join(dir_path, name)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as f:
            file_bytes = f.read()
        head = read_csv_head(file_bytes, nrows=1)
        kind = _classify_low_csv_head(head)
        if kind == "data":
            data_files.append((name, file_bytes))
        elif kind == "map":
            map_files.append((name, file_bytes))

    if not map_files:
        raise ValueError("No PlantDB mapping CSV found (needs ID + signal name columns).")
    if not data_files:
        raise ValueError("No PlantDB data CSV found (needs time + ID + value columns).")

    merged_mapping: Dict[str, str] = {}
    for name, file_bytes in map_files:
        mapping = parse_low_mapping_csv(file_bytes)
        for key, val in mapping.items():
            if key in merged_mapping and merged_mapping[key] != val:
                raise ValueError(
                    f"Conflicting signal mapping for ID {key}: {merged_mapping[key]} vs {val} in {name}."
                )
            merged_mapping[key] = val

    map_df = pd.DataFrame(
        {"id": list(merged_mapping.keys()), "signal": list(merged_mapping.values())}
    )
    map_buf = io.BytesIO()
    map_df.to_csv(map_buf, index=False)

    data_frames = []
    data_rows = 0
    data_cols = None
    for name, file_bytes in data_files:
        df = read_csv_bytes(file_bytes)
        if df.shape[1] < 3:
            continue
        df = df.iloc[:, :3]
        if data_cols is None:
            data_cols = list(df.columns)
        else:
            df.columns = data_cols
        data_rows += len(df)
        data_frames.append(df)

    if not data_frames:
        raise ValueError("PlantDB data CSVs have no usable rows.")

    data_df = pd.concat(data_frames, ignore_index=True)
    data_buf = io.BytesIO()
    data_df.to_csv(data_buf, index=False)

    return map_buf.getvalue(), data_buf.getvalue(), len(map_files), len(data_files), data_rows


def compute_high_meta(
    high_files: Dict[str, bytes],
) -> Tuple[Optional[Tuple[float, float]], int, int]:
    t_min = None
    t_max = None
    row_count = 0
    high_signals = set()
    for data in high_files.values():
        try:
            t_h, x_h = parse_high_time_series_csv(data)
            t_vals = t_h.to_numpy(dtype=float)
            if t_vals.size:
                cur_min = float(np.nanmin(t_vals))
                cur_max = float(np.nanmax(t_vals))
                if t_min is None or cur_min < t_min:
                    t_min = cur_min
                if t_max is None or cur_max > t_max:
                    t_max = cur_max
            row_count += int(len(t_h))
            high_signals.update(x_h.columns.astype(str))
        except Exception:
            continue
    time_range = None
    if t_min is not None and t_max is not None:
        time_range = (t_min, t_max)
    return time_range, len(high_signals), row_count


def linear_interp_truth(
    t_high: np.ndarray,
    x_high: np.ndarray,
    t_low: np.ndarray,
    out_of_range_policy: str,
) -> np.ndarray:
    if out_of_range_policy == "clip":
        t_low2 = np.clip(t_low, t_high[0], t_high[-1])
        return np.interp(t_low2, t_high, x_high)
    x_hat = np.interp(np.clip(t_low, t_high[0], t_high[-1]), t_high, x_high)
    mask = (t_low < t_high[0]) | (t_low > t_high[-1])
    x_hat = x_hat.astype(float)
    x_hat[mask] = np.nan
    return x_hat


def compute_accuracy_for_pair(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
    cfg: CompareConfig,
    time_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, pd.DataFrame]:
    metrics, detail = compute_compare_metrics(
        low_map_bytes,
        low_data_bytes,
        high_bytes,
        cfg,
        time_range,
    )
    return float(metrics["matching_score"]), detail


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
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
    cfg: CompareConfig,
    time_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    id_to_signal = parse_low_mapping_csv(low_map_bytes)
    t_l, y_l = parse_low_samples_csv(low_data_bytes, id_to_signal)
    t_h, x_h = parse_high_time_series_csv(high_bytes)

    t_low = t_l.to_numpy(dtype=float)
    t_high = t_h.to_numpy(dtype=float)

    low_order = np.argsort(t_low)
    high_order = np.argsort(t_high)
    t_low = t_low[low_order]
    t_high = t_high[high_order]

    y_l = y_l.iloc[low_order].reset_index(drop=True)
    x_h = x_h.iloc[high_order].reset_index(drop=True)

    if time_range is not None:
        start, end = time_range
        if start > end:
            raise ValueError("Time range start must be <= end.")
        low_mask = (t_low >= start) & (t_low <= end)
        high_mask = (t_high >= start) & (t_high <= end)
        t_low = t_low[low_mask]
        t_high = t_high[high_mask]
        y_l = y_l.iloc[low_mask].reset_index(drop=True)
        x_h = x_h.iloc[high_mask].reset_index(drop=True)

    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between low and high CSV.")

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

        valid = np.isfinite(t_high) & np.isfinite(xs)
        th = t_high[valid]
        xh = xs[valid]
        if len(th) < 2:
            continue
        df_tmp = pd.DataFrame({"t": th, "x": xh}).drop_duplicates(subset=["t"]).sort_values("t")
        th = df_tmp["t"].to_numpy(dtype=float)
        xh = df_tmp["x"].to_numpy(dtype=float)
        if len(th) < 2:
            continue

        x_hat = linear_interp_truth(th, xh, t_low, cfg.out_of_range_policy)

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
    offset_ms = estimate_offset_ms(t_high, x_mean, t_low, y_mean)
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


def compute_signal_match_stats(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
) -> Dict[str, object]:
    id_to_signal = parse_low_mapping_csv(low_map_bytes)
    t_l, y_l = parse_low_samples_csv(low_data_bytes, id_to_signal)
    t_h, x_h = parse_high_time_series_csv(high_bytes)

    low_signals = set(y_l.columns.astype(str))
    high_signals = set(x_h.columns.astype(str))
    overlap = low_signals & high_signals

    return {
        "low_signals": len(low_signals),
        "high_signals": len(high_signals),
        "overlap": len(overlap),
        "missing_in_high": len(low_signals - high_signals),
        "missing_in_low": len(high_signals - low_signals),
    }


def compute_signal_match_stats_all(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_files: Dict[str, bytes],
) -> Dict[str, object]:
    id_to_signal = parse_low_mapping_csv(low_map_bytes)
    _, y_l = parse_low_samples_csv(low_data_bytes, id_to_signal)
    low_signals = set(y_l.columns.astype(str))

    high_signals = set()
    for data in high_files.values():
        try:
            _, x_h = parse_high_time_series_csv(data)
            high_signals.update(x_h.columns.astype(str))
        except Exception:
            continue

    overlap = low_signals & high_signals
    return {
        "low_signals": len(low_signals),
        "high_signals": len(high_signals),
        "overlap": len(overlap),
        "missing_in_high": len(low_signals - high_signals),
        "missing_in_low": len(high_signals - low_signals),
    }


def compute_low_time_range_and_signal_count(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
) -> Tuple[Tuple[float, float], int]:
    id_to_signal = parse_low_mapping_csv(low_map_bytes)
    t_l, y_l = parse_low_samples_csv(low_data_bytes, id_to_signal)
    t_vals = t_l.to_numpy(dtype=float)
    if t_vals.size == 0:
        raise ValueError("PlantDB CSVs have no valid time values.")
    return (float(np.nanmin(t_vals)), float(np.nanmax(t_vals))), int(y_l.shape[1])


def compute_high_signal_count(high_files: Dict[str, bytes]) -> int:
    high_signals = set()
    for data in high_files.values():
        try:
            _, x_h = parse_high_time_series_csv(data)
            high_signals.update(x_h.columns.astype(str))
        except Exception:
            continue
    return len(high_signals)


def compute_high_time_range(high_files: Dict[str, bytes]) -> Optional[Tuple[float, float]]:
    t_min = None
    t_max = None
    for name, data in high_files.items():
        try:
            t_h, _ = parse_high_time_series_csv(data)
            t_vals = t_h.to_numpy(dtype=float)
            if t_vals.size == 0:
                continue
            cur_min = float(np.nanmin(t_vals))
            cur_max = float(np.nanmax(t_vals))
        except Exception:
            continue
        if t_min is None or cur_min < t_min:
            t_min = cur_min
        if t_max is None or cur_max > t_max:
            t_max = cur_max
    if t_min is None or t_max is None:
        return None
    return t_min, t_max
