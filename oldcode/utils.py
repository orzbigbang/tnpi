import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from state import CompareConfig


# =============================================================================
# 全局常量/配置
# =============================================================================

DEFAULT_TIME_COL_CANDIDATES = ["timestamp", "time", "ts", "date_time", "datetime", "t"]

DATE_TIME_COL_CANDIDATES = {
    "date": ["date"],
    "time": ["time"],
    "ms": ["milli sec", "millisec", "millisecond", "msec", "ms"],
}

EPS = 1e-12

# 估计时间偏移 offset 的搜索范围（毫秒）
# ⚠️ 你原来是 12000（±12s），偏移稍大就容易 NaN
# 建议至少 60000 或 120000。这里默认改成 120000。
OFFSET_SEARCH_MS = 120000

# offset 搜索使用的网格（毫秒）。你的高采样=50ms，所以默认=50ms合理
OFFSET_GRID_MS = 50

# offset 估计时要求的最少有效点数
OFFSET_MIN_POINTS = 30


# =============================================================================
# 时间列检测与解析
# =============================================================================

def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DEFAULT_TIME_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def parse_time_values(series: pd.Series) -> pd.Series:
    """
    将时间列解析为 float：
    - 若整列可转数值：直接 float（单位由数据本身决定）
    - 若整列可转 datetime：转成 int64 ns 再 float
    """
    t_num = pd.to_numeric(series, errors="coerce")
    if t_num.notna().all():
        return t_num.astype(float)

    t_dt = pd.to_datetime(series, errors="coerce")
    if t_dt.notna().all():
        return t_dt.view("int64").astype(float)  # ns

    raise ValueError("Time column is neither numeric nor datetime-parsable.")


# =============================================================================
# CSV 读取（bytes -> DataFrame），带编码容错
# =============================================================================

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


# =============================================================================
# ID 规范化（PlantDB mapping/data 需要）
# =============================================================================

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


# =============================================================================
# 时间序列 CSV 解析（通用版）
# =============================================================================

def parse_time_series_csv(file_bytes: bytes) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_bytes(file_bytes)

    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["date"] if c in cols_lower), None)
    time_col2 = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["time"] if c in cols_lower), None)
    ms_col = next((cols_lower[c] for c in DATE_TIME_COL_CANDIDATES["ms"] if c in cols_lower), None)

    # 格式 A: date + time (+ ms)
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
            signals = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
            return pd.Series(t_vals, name="timestamp"), signals

    # 格式 B: 单列时间
    time_col = detect_time_col(df)
    if time_col is not None:
        t_vals = parse_time_values(df[time_col])
        signals = df.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
        return pd.Series(t_vals, name=time_col), signals

    raise ValueError(
        f"Cannot detect time column. Please include one of {DEFAULT_TIME_COL_CANDIDATES} "
        "or date+time columns (optionally with millisecond)."
    )


# =============================================================================
# 高采样（ODG）CSV 解析：更严格
# =============================================================================

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
    signals = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
    return pd.Series(t_vals, name="timestamp"), signals


# =============================================================================
# PlantDB mapping/data 解析
# =============================================================================

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
    id_col = _pick_column_by_name(col_names, ("id", "signalid", "tagid", "n_monitoring_id"))
    signal_col = _pick_column_by_name(col_names, ("signal", "tag", "name", "s_signal_name1"))

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

    wide = tmp.pivot_table(index="t", columns="signal", values="value", aggfunc="mean").sort_index()
    t_series = pd.Series(wide.index.to_numpy(dtype=float), name="timestamp")
    return t_series, wide.reset_index(drop=True)


# =============================================================================
# 读取 ODG 文件夹（zip 或目录）
# =============================================================================

def load_high_folder_from_zip(zip_bytes: bytes) -> Dict[str, bytes]:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    out: Dict[str, bytes] = {}
    for name in z.namelist():
        if name.lower().endswith(".csv") and not name.endswith("/"):
            out[name] = z.read(name)
    if not out:
        raise ValueError("No CSV files found in zip.")
    return out


def load_high_folder_from_dir(dir_path: str) -> Dict[str, bytes]:
    if not os.path.isdir(dir_path):
        raise ValueError("ODG folder does not exist.")
    out: Dict[str, bytes] = {}
    for name in os.listdir(dir_path):
        if name.lower().endswith(".csv"):
            full_path = os.path.join(dir_path, name)
            if os.path.isfile(full_path):
                with open(full_path, "rb") as f:
                    out[name] = f.read()
    if not out:
        raise ValueError("No CSV files found in folder.")
    return out


# =============================================================================
# PlantDB 文件夹读取与合并
# =============================================================================

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

    map_df = pd.DataFrame({"id": list(merged_mapping.keys()), "signal": list(merged_mapping.values())})
    map_buf = io.BytesIO()
    map_df.to_csv(map_buf, index=False)

    data_frames = []
    data_rows = 0
    data_cols = None
    for _, file_bytes in data_files:
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


# =============================================================================
# 高采样数据元信息统计
# =============================================================================

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


# =============================================================================
# 插值：用高采样作为真值，在低采样时间点取值
# =============================================================================

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


# =============================================================================
# 对外接口：算某一对 low/high 的 matching score
# =============================================================================

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


# =============================================================================
# ✅ 新版 offset 估计算法：插值到网格 + FFT 互相关（更稳，不容易 NaN）
# =============================================================================

def _as_int64_ns(t_vals: np.ndarray) -> np.ndarray:
    """
    将输入时间数组尽量转换成 int64 的 ns 时间戳。

    支持：
    - datetime64[ns] / pandas datetime -> int64 ns
    - numeric：用数量级推断单位（sec/ms/ns）并转换到 ns

    目的：
    - 避免你原实现“强假设 unit=ns”导致 NaT / NaN
    """
    t = np.asarray(t_vals)

    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype("int64")

    arr = np.asarray(t_vals, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.asarray([], dtype="int64")

    med = float(np.median(np.abs(finite)))
    if med < 1e11:          # seconds epoch ~ 1e9
        scale = 1e9
    elif med < 1e15:        # milliseconds epoch ~ 1e12
        scale = 1e6
    else:                   # nanoseconds epoch ~ 1e18
        scale = 1.0

    return (arr * scale).round().astype("int64")


def _prepare_series(t_ns: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    清洗并排序：
    - 过滤无效点（NaN/inf）
    - 按时间升序
    - 去重时间戳（保留最后一个）
    """
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

    # 去重：保留最后一个
    t_rev = t[::-1]
    x_rev = x[::-1]
    _, idx = np.unique(t_rev, return_index=True)
    keep_rev = np.sort(idx)
    t_u = t_rev[keep_rev][::-1]
    x_u = x_rev[keep_rev][::-1]
    return t_u, x_u


def _interp_to_grid(t_ns: np.ndarray, v: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    """
    线性插值到规则网格。范围外返回 NaN（drop 策略）。
    """
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
    """
    在规则采样序列 a/b 上估计最优 lag（单位：样本点）。
    - 用 mask 处理 NaN：无效点置 0
    - 用 FFT 做互相关
    - 用 overlap（有效点重叠数）归一化，避免缺失导致偏差
    - 取 [-max_shift, +max_shift] 内 score 最大的 lag
    """
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
    """
    新版 offset 估计（替换原来的 resample+枚举 RMSE）。

    符号约定：
    - 返回 offset_ms > 0 表示：t2 相对 t1 更晚（t2 滞后 t1）
    """
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


# =============================================================================
# 核心：比较 low vs high
# =============================================================================

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

    # 原始时间（float）
    t_low = t_l.to_numpy(dtype=float)
    t_high = t_h.to_numpy(dtype=float)

    # 排序并同步 DataFrame
    low_order = np.argsort(t_low)
    high_order = np.argsort(t_high)
    t_low = t_low[low_order]
    t_high = t_high[high_order]
    y_l = y_l.iloc[low_order].reset_index(drop=True)
    x_h = x_h.iloc[high_order].reset_index(drop=True)

    # 统一时间单位到 ns（int64 ns -> float ns）
    t_low_ns = _as_int64_ns(t_low)
    t_high_ns = _as_int64_ns(t_high)
    t_low_f = t_low_ns.astype(float)
    t_high_f = t_high_ns.astype(float)

    # time_range 过滤（这里假设 time_range 与输入时间同一量纲；
    # 你的数据大概率也是 ns。如果你的 time_range 是秒/ms，需要在外层统一。）
    if time_range is not None:
        start, end = time_range
        if start > end:
            raise ValueError("Time range start must be <= end.")

        # 将 time_range 也转换成 ns 来对齐过滤
        start_ns = _as_int64_ns(np.asarray([start], dtype="float64"))[0]
        end_ns = _as_int64_ns(np.asarray([end], dtype="float64"))[0]

        low_mask = (t_low_ns >= start_ns) & (t_low_ns <= end_ns)
        high_mask = (t_high_ns >= start_ns) & (t_high_ns <= end_ns)

        t_low_ns = t_low_ns[low_mask]
        t_high_ns = t_high_ns[high_mask]
        t_low_f = t_low_ns.astype(float)
        t_high_f = t_high_ns.astype(float)

        y_l = y_l.iloc[low_mask].reset_index(drop=True)
        x_h = x_h.iloc[high_mask].reset_index(drop=True)

    common_cols = [c for c in y_l.columns if c in x_h.columns]
    if not common_cols:
        raise ValueError("No overlapping signal columns between low and high CSV.")

    # 计算每列 denominator（基于 high）
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

    # ------------------------------------------------------------
    # ✅ 先估 offset，再对齐时间轴
    # ------------------------------------------------------------
    x_common = x_h[common_cols].to_numpy(dtype=float)
    y_common = y_l[common_cols].to_numpy(dtype=float)
    x_mean = np.nanmean(x_common, axis=1)
    y_mean = np.nanmean(y_common, axis=1)

    # estimate_offset_ms 的符号约定：
    # offset > 0 表示 t2(这里 low) 比 t1(这里 high) 更晚（low 滞后）
    # 要把 low 对齐到 high：t_low_aligned = t_low - offset
    offset_ms = estimate_offset_ms(t_high_ns, x_mean, t_low_ns, y_mean)

    t_low_aligned_f = t_low_f.copy()
    if np.isfinite(offset_ms):
        t_low_aligned_f = t_low_f - float(offset_ms) * 1_000_000.0  # ms -> ns

    # ------------------------------------------------------------
    # 逐信号插值 + 打分（用对齐后的 t_low_aligned_f）
    # ------------------------------------------------------------
    rows = []
    acc_values = []
    rmse_values = []
    corr_values = []

    for c in common_cols:
        xs = x_h[c].to_numpy(dtype=float)
        ys = y_l[c].to_numpy(dtype=float)

        # high 侧：过滤无效点，构建单调且去重的时间序列（ns）
        valid = np.isfinite(t_high_f) & np.isfinite(xs)
        th = t_high_f[valid]
        xh = xs[valid]
        if th.size < 2:
            continue

        # 去重并排序（避免重复时间导致插值异常）
        df_tmp = pd.DataFrame({"t": th, "x": xh}).drop_duplicates(subset=["t"]).sort_values("t")
        th = df_tmp["t"].to_numpy(dtype=float)
        xh = df_tmp["x"].to_numpy(dtype=float)
        if th.size < 2:
            continue

        # 在对齐后的 low 时间点上取 high 的参考值
        x_hat = linear_interp_truth(th, xh, t_low_aligned_f, cfg.out_of_range_policy)

        ok = np.isfinite(ys) & np.isfinite(x_hat)
        if ok.sum() == 0:
            continue

        diff = ys - x_hat
        abs_diff = np.abs(diff)

        rmse = float(np.sqrt(np.nanmean(diff[ok] ** 2)))
        corr = float(np.corrcoef(ys[ok], x_hat[ok])[0, 1]) if ok.sum() > 1 else float("nan")

        # 将误差换算成 0~1 分数
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
        acc_values.append(sig_acc)
        rmse_values.append(rmse)
        if np.isfinite(corr):
            corr_values.append(corr)

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values("matching_score", ascending=True)

    matching_score = float(np.nanmean(acc_values)) if acc_values else float("nan")
    rmse_avg = float(np.nanmean(rmse_values)) if rmse_values else float("nan")
    corr_avg = float(np.nanmean(corr_values)) if corr_values else float("nan")

    metrics = {
        "matching_score": matching_score,
        "rmse": rmse_avg,
        "correlation": corr_avg,
        "offset_ms": float(offset_ms),
    }
    return metrics, detail


# =============================================================================
# 信号集合统计
# =============================================================================

def compute_signal_match_stats(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
) -> Dict[str, object]:
    id_to_signal = parse_low_mapping_csv(low_map_bytes)
    _, y_l = parse_low_samples_csv(low_data_bytes, id_to_signal)
    _, x_h = parse_high_time_series_csv(high_bytes)

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


# =============================================================================
# 辅助统计：low 时间范围/信号数；high 信号数/时间范围
# =============================================================================

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
    for _, data in high_files.items():
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
