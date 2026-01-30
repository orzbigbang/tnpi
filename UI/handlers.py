import os
from typing import Callable, Optional, Tuple

import pandas as pd

from core.parallel import compute_accuracy_for_all_odg_mp
from state import AppConfig, RunResult
from core.io_csv import detect_csv_encoding
from core.parse import (
    compute_odg_meta,
    compute_plant_time_range_and_signal_count,
    load_odg_folder_from_dir,
    load_plant_folder_from_dir,
)


def confirm_samples(
    app_cfg: AppConfig,
    odg_folder: str,
    plant_folder: str,
) -> Optional[str]:
    odg_path = (odg_folder or "").strip()
    plant_path = (plant_folder or "").strip()
    if not odg_path or not plant_path:
        return "Please enter both odg and plant folder paths before confirming."
    odg_error = confirm_odg_folder(app_cfg, odg_path)
    if odg_error:
        return odg_error
    plant_error = confirm_plant_folder(app_cfg, plant_path)
    if plant_error:
        reset_odg_state(app_cfg)
        return plant_error
    overlap_error = validate_sampling_overlap(app_cfg)
    if overlap_error:
        reset_odg_state(app_cfg)
        reset_plant_state(app_cfg)
        return overlap_error
    app_cfg.dump_ini()
    return None


def confirm_odg_folder(app_cfg: AppConfig, folder_path: str) -> Optional[str]:
    if not folder_path:
        return "Please enter a folder path before confirming."
    try:
        app_cfg.state.odg_folder = folder_path
        app_cfg.odg.odg_files = load_odg_folder_from_dir(folder_path)
        encodings = {detect_csv_encoding(p) for p in app_cfg.odg.odg_files}
        if len(encodings) > 1:
            raise ValueError(f"ODG CSV encodings are inconsistent: {sorted(encodings)}")
        app_cfg.odg.odg_encoding = next(iter(encodings)) if encodings else None
        app_cfg.odg.odg_names = [os.path.basename(p) for p in app_cfg.odg.odg_files]
        (
            app_cfg.odg.odg_time_range,
            app_cfg.odg.odg_signal_count,
            app_cfg.odg.odg_row_count,
        ) = compute_odg_meta(app_cfg.odg.odg_files, encoding=app_cfg.odg.odg_encoding)
        if app_cfg.odg.odg_time_range is not None:
            t_min, t_max = app_cfg.odg.odg_time_range
            dt_min = pd.to_datetime(int(t_min), unit="ns")
            dt_max = pd.to_datetime(int(t_max), unit="ns")
            app_cfg.state.range_start = dt_min.strftime("%Y-%m-%d-%H-%M")
            app_cfg.state.range_end = dt_max.strftime("%Y-%m-%d-%H-%M")
        app_cfg.state.odg_confirmed = True
    except Exception as ex:
        reset_odg_state(app_cfg)
        return f"ODG confirm failed: {ex}"
    return None


def confirm_plant_folder(app_cfg: AppConfig, folder_path: str) -> Optional[str]:
    if not folder_path:
        return "Please enter a folder path before confirming."
    try:
        app_cfg.state.plant_folder = folder_path
        (
            app_cfg.plant.plant_map_files,
            app_cfg.plant.plant_data_files,
            app_cfg.plant.plant_map_count,
            app_cfg.plant.plant_data_count,
            app_cfg.plant.plant_data_rows,
        ) = load_plant_folder_from_dir(folder_path)
        map_encodings = {detect_csv_encoding(p) for p in app_cfg.plant.plant_map_files}
        if len(map_encodings) > 1:
            raise ValueError(f"Plant map CSV encodings are inconsistent: {sorted(map_encodings)}")
        app_cfg.plant.plant_map_encoding = next(iter(map_encodings)) if map_encodings else None
        data_encodings = {detect_csv_encoding(p) for p in app_cfg.plant.plant_data_files}
        if len(data_encodings) > 1:
            raise ValueError(f"Plant data CSV encodings are inconsistent: {sorted(data_encodings)}")
        app_cfg.plant.plant_data_encoding = next(iter(data_encodings)) if data_encodings else None
        (
            app_cfg.plant.plant_time_range,
            app_cfg.plant.plant_signal_count,
            app_cfg.plant.plant_id_signal_map,
        ) = (
            compute_plant_time_range_and_signal_count(
                app_cfg.plant.plant_map_files,
                app_cfg.plant.plant_data_files,
                map_encoding=app_cfg.plant.plant_map_encoding,
                data_encoding=app_cfg.plant.plant_data_encoding,
            )
        )
        app_cfg.state.plant_confirmed = True
    except Exception as ex:
        reset_plant_state(app_cfg)
        return f"PlantDB confirm failed: {ex}"
    return None


def validate_sampling_overlap(app_cfg: AppConfig) -> Optional[str]:
    if app_cfg.odg.odg_time_range is None or app_cfg.plant.plant_time_range is None:
        return "Unable to read sampling time ranges for overlap check."
    odg_start, odg_end = app_cfg.odg.odg_time_range
    plant_start, plant_end = app_cfg.plant.plant_time_range
    overlap_start = max(odg_start, plant_start)
    overlap_end = min(odg_end, plant_end)
    if overlap_start > overlap_end:
        return "ODG and PlantDB time ranges do not overlap."
    return None


def compute_run_result(
    app_cfg: AppConfig,
    *,
    range_mode: str,
    range_start: str,
    range_end: str,
    progress_cb: Callable,
) -> RunResult:
    try:
        time_range = parse_time_range(range_mode, range_start, range_end)
    except ValueError as ex:
        return RunResult(ok=False, error=str(ex))

    metrics, detail_all = compute_accuracy_for_all_odg_mp(
        app_cfg.plant.plant_id_signal_map,
        app_cfg.plant.plant_data_files,
        app_cfg.odg.odg_files,
        app_cfg.compare,
        progress_cb=progress_cb,
        time_range=time_range,
        plant_data_encoding=app_cfg.plant.plant_data_encoding,
        odg_encoding=app_cfg.odg.odg_encoding,
    )
    return RunResult(
        ok=True,
        summary=metrics["matching_score"],
        matching_score=metrics["matching_score"],
        rmse=metrics["rmse"],
        correlation=metrics["correlation"],
        offset_ms=metrics["offset_ms"],
        detail=detail_all,
    )


def parse_time_range(
    range_mode: str,
    range_start: str,
    range_end: str,
) -> Optional[Tuple[float, float]]:
    if range_mode != "range":
        return None
    start_txt = (range_start or "").strip()
    end_txt = (range_end or "").strip()
    if not start_txt or not end_txt:
        raise ValueError("Please enter both range start and range end.")
    try:
        start_dt = pd.to_datetime(start_txt, format="%Y-%m-%d-%H-%M", errors="coerce")
        end_dt = pd.to_datetime(end_txt, format="%Y-%m-%d-%H-%M", errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ValueError
        start_val = float(start_dt.value)
        end_val = float(end_dt.value)
    except ValueError:
        raise ValueError(
            "Range values must use format YYYY-mm-dd-hh-mm (e.g., 2026-01-21-13-45)."
        )
    return start_val, end_val


def reset_odg_state(
    app_cfg: AppConfig,
) -> None:
    ui_state = app_cfg.state
    odg_cfg = app_cfg.odg
    ui_state.odg_confirmed = False
    odg_cfg.odg_files = []
    odg_cfg.odg_names = []
    odg_cfg.odg_time_range = None
    odg_cfg.odg_signal_count = None
    odg_cfg.odg_row_count = None
    odg_cfg.odg_encoding = None


def reset_plant_state(
    app_cfg: AppConfig,
) -> None:
    ui_state = app_cfg.state
    plant_cfg = app_cfg.plant
    ui_state.plant_confirmed = False
    plant_cfg.plant_time_range = None
    plant_cfg.plant_signal_count = None
    plant_cfg.plant_id_signal_map = {}
    plant_cfg.plant_map_files = []
    plant_cfg.plant_data_files = []
    plant_cfg.plant_map_count = None
    plant_cfg.plant_data_count = None
    plant_cfg.plant_data_rows = None
    plant_cfg.plant_map_encoding = None
    plant_cfg.plant_data_encoding = None
