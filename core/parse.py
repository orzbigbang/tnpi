import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from enum_types import PlantCsvType, SignalType
from core.io_csv import (
    read_csv_first_n_cols,
    read_csv_first_n_cols_chunks,
    read_csv_head,
    read_csv_path,
)

DATE_TIME_COL_CANDIDATES = {
    "date": ["date"],
    "time": ["time"],
    "ms": ["milli sec", "millisec", "millisecond", "msec", "ms"],
}


def parse_time_values(series: pd.Series) -> pd.Series:
    # Strip simple trailing timezone offsets like +09 or +09:00 before parsing.
    s = series.astype(str).str.strip()
    s = s.str.replace(r"([+-]\d{2})(?::?\d{2})?$", "", regex=True)
    # Remove non-printable characters that can break parsing.
    s = s.str.replace(r"[^\x20-\x7E]", "", regex=True)
    t_dt = pd.to_datetime(s, errors="coerce")
    if t_dt.isna().any():
        mask = t_dt.isna()
        # Retry common no-fraction format for remaining rows.
        t_dt.loc[mask] = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if t_dt.notna().all():
        return t_dt.view("int64").astype(float)  # ns
    # Surface a small sample to help identify bad formats.
    bad_mask = t_dt.isna()
    sample_raw = series.astype(str).head(5).tolist()
    sample_clean = s.head(5).tolist()
    bad_raw = series.astype(str)[bad_mask].head(5).tolist()
    bad_clean = s[bad_mask].head(5).tolist()
    raise ValueError(
        "Time column is neither numeric nor datetime-parsable. "
        f"sample_raw={sample_raw} sample_clean={sample_clean} "
        f"bad_count={int(bad_mask.sum())} bad_raw={bad_raw} bad_clean={bad_clean}"
    )


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


def parse_odg_time_series_csv(
    path: Union[str, os.PathLike],
    encoding: Optional[str] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_path(path, encoding=encoding)
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


def _normalize_id_to_signal(
    mapping: Dict[str, Union[str, Tuple[str, SignalType]]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in mapping.items():
        if isinstance(val, tuple):
            out[key] = str(val[0])
        else:
            out[key] = str(val)
    return out


def parse_plant_samples_csv(
    path: Union[str, os.PathLike],
    id_to_signal: Dict[str, Union[str, Tuple[str, SignalType]]],
    encoding: Optional[str] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_csv_first_n_cols(path, 3, encoding=encoding)
    if df.shape[1] < 3:
        raise ValueError("PlantDB CSV #2 must have at least three columns: time, ID, value.")
    time_col = df.columns[0]
    id_col = df.columns[1]
    value_col = df.columns[2]

    t_vals = parse_time_values(df[time_col])
    ids = normalize_id_series(df[id_col])
    signals = ids.map(_normalize_id_to_signal(id_to_signal))
    values = pd.to_numeric(df[value_col], errors="coerce")

    tmp = pd.DataFrame({"t": t_vals, "signal": signals, "value": values})
    tmp = tmp.dropna(subset=["t", "signal", "value"])
    if tmp.empty:
        raise ValueError("PlantDB CSV #2 has no usable rows after ID mapping.")

    wide = tmp.pivot_table(index="t", columns="signal", values="value", aggfunc="mean")
    wide = wide.sort_index()
    t_series = pd.Series(wide.index.to_numpy(dtype=float), name="timestamp")
    return t_series, wide.reset_index(drop=True)


def load_odg_folder_from_dir(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise ValueError("ODG folder does not exist.")
    out = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".csv"):
            full_path = os.path.join(dir_path, name)
            if os.path.isfile(full_path):
                out.append(full_path)
    if not out:
        raise ValueError("No CSV files found in folder.")
    return sorted(out)


def _classify_plant_csv_head(head: pd.DataFrame) -> Optional[PlantCsvType]:
    cols = [str(c).strip().lower() for c in head.columns]
    if not cols:
        return None
    has_id = any("n_monitoring_id" in c for c in cols)
    has_signal = any("s_signal_name1" in c for c in cols)
    has_time = any("t_sample_time" in c for c in cols)
    has_value = any("n_signal1" in c for c in cols)
    if has_id and has_signal and not (has_time and has_value):
        return PlantCsvType.MAP
    if has_time and has_id and has_value:
        return PlantCsvType.DATA
    return None


def load_plant_folder_from_dir(
    dir_path: str,
    *,
    map_encoding: Optional[str] = None,
    data_encoding: Optional[str] = None,
) -> Tuple[List[str], List[str], int, int, int]:
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
        head = read_csv_head(full_path, encoding=map_encoding or data_encoding)
        kind = _classify_plant_csv_head(head)
        if kind == PlantCsvType.DATA:
            data_files.append(full_path)
        elif kind == PlantCsvType.MAP:
            map_files.append(full_path)

    if not map_files:
        raise ValueError("No PlantDB mapping CSV found (needs ID + signal name columns).")
    if not data_files:
        raise ValueError("No PlantDB data CSV found (needs time + ID + value columns).")

    data_rows = 0
    for path in data_files:
        chunks = read_csv_first_n_cols_chunks(path, 3, encoding=data_encoding)
        has_rows = False
        for chunk in chunks:
            if chunk.shape[1] < 3:
                continue
            data_rows += int(len(chunk))
            has_rows = True
        if not has_rows:
            continue

    if data_rows == 0:
        raise ValueError("PlantDB data CSVs have no usable rows.")

    return map_files, data_files, len(map_files), len(data_files), data_rows


def compute_odg_meta(
    odg_files: List[str],
    *,
    encoding: Optional[str] = None,
) -> Tuple[Optional[Tuple[float, float]], int, int]:
    t_min = None
    t_max = None
    row_count = 0
    odg_signals = set()
    for path in odg_files:
        try:
            head = read_csv_head(path, nrows=0, encoding=encoding)
            if head.shape[1] > 3:
                odg_signals.update([str(c) for c in head.columns[3:]])
            chunks = read_csv_first_n_cols_chunks(path, 3, encoding=encoding)
            for time_df in chunks:
                if time_df.shape[1] < 2:
                    continue
                date_col = time_df.columns[0]
                time_col = time_df.columns[1]
                dt = pd.to_datetime(
                    time_df[date_col].astype(str) + " " + time_df[time_col].astype(str),
                    errors="coerce",
                )
                if time_df.shape[1] >= 3:
                    ms = pd.to_numeric(time_df[time_df.columns[2]], errors="coerce")
                    dt = dt + pd.to_timedelta(ms.fillna(0), unit="ms")
                if dt.notna().any():
                    t_vals = dt.view("int64").astype(float)
                    cur_min = float(np.nanmin(t_vals))
                    cur_max = float(np.nanmax(t_vals))
                    if t_min is None or cur_min < t_min:
                        t_min = cur_min
                    if t_max is None or cur_max > t_max:
                        t_max = cur_max
                row_count += int(len(time_df))
        except Exception:
            continue
    time_range = None
    if t_min is not None and t_max is not None:
        time_range = (t_min, t_max)
    return time_range, len(odg_signals), row_count


def compute_plant_time_range_and_signal_count(
    plant_map_paths: List[str],
    plant_data_paths: List[str],
    *,
    map_encoding: Optional[str] = None,
    data_encoding: Optional[str] = None,
) -> Tuple[Tuple[float, float], int, Dict[str, Tuple[str, SignalType]], int]:
    merged_mapping: Dict[str, Tuple[str, SignalType]] = {}
    skipped = 0
    for path in plant_map_paths:
        df = read_csv_path(path, encoding=map_encoding)
        if df.shape[1] < 2:
            continue
        ids = normalize_id_series(df.iloc[:, 0])
        signals = df.iloc[:, 1].astype(str)
        signal_types_raw = df.iloc[:, -1].astype(str)
        mask = ids.notna() & signals.notna() & signal_types_raw.notna()
        for key, name, st_raw in zip(ids[mask], signals[mask], signal_types_raw[mask]):
            st_str = str(st_raw).strip()
            if not st_str or st_str.lower() == "nan":
                skipped += 1
                continue
            try:
                st = SignalType(st_str)
            except Exception as ex:
                st = SignalType.LOW
                # raise ValueError(f"Invalid signal type '{st_raw}' in {path}.") from ex
            val = (name, st)
            if key in merged_mapping and merged_mapping[key] != val:
                raise ValueError(
                    f"Conflicting signal mapping for ID {key}: {merged_mapping[key]} vs {val} in {path}."
                )
            merged_mapping[key] = val
    if not merged_mapping:
        raise ValueError("PlantDB CSV #1 has no valid ID-to-signal mappings.")

    t_min = None
    t_max = None
    for path in plant_data_paths:
        chunks = read_csv_first_n_cols_chunks(path, 1, encoding=data_encoding)
        for df in chunks:
            if df.shape[1] < 1:
                continue
            t_vals = parse_time_values(df.iloc[:, 0])
            t_vals = pd.to_numeric(t_vals, errors="coerce").to_numpy(dtype=float)
            if t_vals.size == 0:
                continue
            cur_min = float(np.nanmin(t_vals))
            cur_max = float(np.nanmax(t_vals))
            if t_min is None or cur_min < t_min:
                t_min = cur_min
            if t_max is None or cur_max > t_max:
                t_max = cur_max
    if t_min is None or t_max is None:
        raise ValueError("PlantDB CSVs have no valid time values.")
    signal_count = len({val[0] for val in merged_mapping.values()})
    return (t_min, t_max), signal_count, merged_mapping, skipped
