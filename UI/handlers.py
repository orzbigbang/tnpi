from typing import Callable, Optional, Tuple

import pandas as pd

import numpy as np

from multiprocess import compute_accuracy_for_all_high_mp
from state import AppConfig, CompareConfig, RunResult
from utils import (
    compute_compare_metrics,
    compute_high_meta,
    compute_low_time_range_and_signal_count,
    compute_signal_match_stats_all,
    load_high_folder_from_dir,
    load_low_folder_from_dir,
)


def reset_high_state(
    app_cfg: AppConfig,
) -> None:
    ui_state = app_cfg.state
    high_cfg = app_cfg.high
    ui_state.high_confirmed = False
    high_cfg.high_files = None
    high_cfg.high_names = []
    high_cfg.high_time_range = None
    high_cfg.high_signal_count = None
    high_cfg.high_row_count = None


def reset_low_state(
    app_cfg: AppConfig,
) -> None:
    ui_state = app_cfg.state
    low_cfg = app_cfg.low
    ui_state.low_confirmed = False
    low_cfg.low_time_range = None
    low_cfg.low_signal_count = None
    low_cfg.low_map_bytes = None
    low_cfg.low_data_bytes = None
    low_cfg.low_map_count = None
    low_cfg.low_data_count = None
    low_cfg.low_data_rows = None


def confirm_high_folder(app_cfg: AppConfig, folder_path: str) -> Optional[str]:
    ui_state = app_cfg.state
    high_cfg = app_cfg.high
    if not folder_path:
        return "Please enter a folder path before confirming."
    try:
        ui_state.high_folder = folder_path
        high_cfg.high_files = load_high_folder_from_dir(folder_path)
        high_cfg.high_names = sorted(high_cfg.high_files.keys())
        (
            high_cfg.high_time_range,
            high_cfg.high_signal_count,
            high_cfg.high_row_count,
        ) = compute_high_meta(high_cfg.high_files)
        if high_cfg.high_time_range is not None:
            t_min, t_max = high_cfg.high_time_range
            dt_min = pd.to_datetime(int(t_min), unit="ns")
            dt_max = pd.to_datetime(int(t_max), unit="ns")
            ui_state.range_start = dt_min.strftime("%Y-%m-%d-%H-%M")
            ui_state.range_end = dt_max.strftime("%Y-%m-%d-%H-%M")
        ui_state.high_confirmed = True
    except Exception as ex:
        reset_high_state(app_cfg)
        return f"ODG confirm failed: {ex}"
    return None


def confirm_low_folder(app_cfg: AppConfig, folder_path: str) -> Optional[str]:
    ui_state = app_cfg.state
    low_cfg = app_cfg.low
    if not folder_path:
        return "Please enter a folder path before confirming."
    try:
        ui_state.low_folder = folder_path
        (
            low_cfg.low_map_bytes,
            low_cfg.low_data_bytes,
            low_cfg.low_map_count,
            low_cfg.low_data_count,
            low_cfg.low_data_rows,
        ) = load_low_folder_from_dir(folder_path)
        low_cfg.low_time_range, low_cfg.low_signal_count = (
            compute_low_time_range_and_signal_count(
                low_cfg.low_map_bytes,
                low_cfg.low_data_bytes,
            )
        )
        ui_state.low_confirmed = True
    except Exception as ex:
        reset_low_state(app_cfg)
        return f"PlantDB confirm failed: {ex}"
    return None


def confirm_samples(
    app_cfg: AppConfig,
    high_folder: str,
    low_folder: str,
) -> Optional[str]:
    high_path = (high_folder or "").strip()
    low_path = (low_folder or "").strip()
    if not high_path or not low_path:
        return "Please enter both high and low folder paths before confirming."
    high_error = confirm_high_folder(app_cfg, high_path)
    if high_error:
        return high_error
    low_error = confirm_low_folder(app_cfg, low_path)
    if low_error:
        reset_high_state(app_cfg)
        return low_error
    overlap_error = validate_sampling_overlap(app_cfg)
    if overlap_error:
        reset_high_state(app_cfg)
        reset_low_state(app_cfg)
        return overlap_error
    app_cfg.dump_ini()
    return None


def validate_sampling_overlap(app_cfg: AppConfig) -> Optional[str]:
    high_cfg = app_cfg.high
    low_cfg = app_cfg.low
    if high_cfg.high_time_range is None or low_cfg.low_time_range is None:
        return "Unable to read sampling time ranges for overlap check."
    high_start, high_end = high_cfg.high_time_range
    low_start, low_end = low_cfg.low_time_range
    overlap_start = max(high_start, low_start)
    overlap_end = min(high_end, low_end)
    if overlap_start > overlap_end:
        return "ODG and PlantDB time ranges do not overlap."
    return None


def parse_time_range(
    range_mode: str,
    range_start: str,
    range_end: str,
) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    if range_mode != "range":
        return None, None
    start_txt = (range_start or "").strip()
    end_txt = (range_end or "").strip()
    if not start_txt or not end_txt:
        return None, "Please enter both range start and range end."
    try:
        start_dt = pd.to_datetime(start_txt, format="%Y-%m-%d-%H-%M", errors="coerce")
        end_dt = pd.to_datetime(end_txt, format="%Y-%m-%d-%H-%M", errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ValueError
        start_val = float(start_dt.value)
        end_val = float(end_dt.value)
    except ValueError:
        return (
            None,
            "Range values must use format YYYY-mm-dd-hh-mm (e.g., 2024-01-01-00-00).",
        )
    return (start_val, end_val), None


def compute_run_result(
    app_cfg: AppConfig,
    *,
    range_mode: str,
    range_start: str,
    range_end: str,
    run_mode: str,
    progress_cb: Callable[[int, int], None],
) -> RunResult:
    high_cfg = app_cfg.high
    low_cfg = app_cfg.low
    cfg = app_cfg.compare
    time_range, range_error = parse_time_range(range_mode, range_start, range_end)
    if range_error:
        return RunResult(ok=False, error=range_error)
    low_map_bytes = low_cfg.low_map_bytes
    low_data_bytes = low_cfg.low_data_bytes
    detail_all = pd.DataFrame()
    if run_mode == "multiprocess":
        metrics, detail_all = compute_accuracy_for_all_high_mp(
            low_map_bytes,
            low_data_bytes,
            high_cfg.high_files,
            cfg,
            progress_cb=progress_cb,
            time_range=time_range,
        )
    else:
        high_names = high_cfg.high_names
        metrics_rows = []
        detail_rows = []
        for idx, name in enumerate(high_names, start=1):
            try:
                metrics, detail = compute_compare_metrics(
                    low_map_bytes,
                    low_data_bytes,
                    high_cfg.high_files[name],
                    cfg,
                    time_range,
                )
                metrics_rows.append(metrics)
                if not detail.empty:
                    detail_rows.append(detail)
            except Exception:
                metrics_rows.append(
                    {
                        "matching_score": float("nan"),
                        "rmse": float("nan"),
                        "correlation": float("nan"),
                        "offset_ms": float("nan"),
                    }
                )
            progress_cb(idx, len(high_names))
        if metrics_rows:
            metrics = {
                "matching_score": float(np.nanmean([m["matching_score"] for m in metrics_rows])),
                "rmse": float(np.nanmean([m["rmse"] for m in metrics_rows])),
                "correlation": float(np.nanmean([m["correlation"] for m in metrics_rows])),
                "offset_ms": float(np.nanmean([m["offset_ms"] for m in metrics_rows])),
            }
        else:
            metrics = {
                "matching_score": float("nan"),
                "rmse": float("nan"),
                "correlation": float("nan"),
                "offset_ms": float("nan"),
            }
        if detail_rows:
            detail_all = pd.concat(detail_rows, ignore_index=True)
    try:
        match_stats = compute_signal_match_stats_all(
            low_map_bytes,
            low_data_bytes,
            high_cfg.high_files,
        )
    except Exception as ex:
        match_stats = {
            "low_signals": 0,
            "high_signals": 0,
            "overlap": 0,
            "missing_in_high": 0,
            "missing_in_low": 0,
            "error": str(ex),
        }
    return RunResult(
        ok=True,
        summary=metrics["matching_score"],
        matching_score=metrics["matching_score"],
        rmse=metrics["rmse"],
        correlation=metrics["correlation"],
        offset_ms=metrics["offset_ms"],
        match_stats=match_stats,
        detail=detail_all,
    )
