from typing import Callable, Optional, Tuple

import pandas as pd

from core.confirm import confirm_samples as confirm_samples_core
from core.parallel import compute_accuracy_for_all_odg_mp
from core.models import RunResult
from state import AppConfig


def confirm_samples(
    app_cfg: AppConfig,
    odg_folder: str,
    plant_folder: str,
) -> Optional[str]:
    app_cfg.state.confirm_sample_inspector = None
    result = confirm_samples_core(odg_folder, plant_folder)
    if not result.ok:
        reset_odg_state(app_cfg)
        reset_plant_state(app_cfg)
        msg = result.error or "Confirm sampling failed."
        if result.error_code:
            msg = f"{msg} (code: {result.error_code})"
        return msg

    odg_meta = result.odg_meta
    plant_meta = result.plant_meta
    if odg_meta is None or plant_meta is None:
        reset_odg_state(app_cfg)
        reset_plant_state(app_cfg)
        return "Confirm sampling failed: incomplete metadata."

    app_cfg.state.odg_folder = (odg_folder or "").strip()
    app_cfg.state.plant_folder = (plant_folder or "").strip()

    app_cfg.odg.odg_files = odg_meta.files
    app_cfg.odg.odg_names = odg_meta.names
    app_cfg.odg.odg_time_range = odg_meta.time_range
    app_cfg.odg.odg_signal_count = odg_meta.signal_count
    app_cfg.odg.odg_row_count = odg_meta.row_count
    app_cfg.odg.odg_encoding = odg_meta.encoding

    app_cfg.plant.plant_map_files = plant_meta.map_files
    app_cfg.plant.plant_data_files = plant_meta.data_files
    app_cfg.plant.plant_map_count = plant_meta.map_count
    app_cfg.plant.plant_data_count = plant_meta.data_count
    app_cfg.plant.plant_data_rows = plant_meta.data_rows
    app_cfg.plant.plant_skipped_mappings = plant_meta.skipped_mappings
    app_cfg.plant.plant_map_encoding = plant_meta.map_encoding
    app_cfg.plant.plant_data_encoding = plant_meta.data_encoding
    app_cfg.plant.plant_time_range = plant_meta.time_range
    app_cfg.plant.plant_signal_count = plant_meta.signal_count
    app_cfg.plant.plant_id_signal_map = plant_meta.id_signal_map

    if app_cfg.odg.odg_time_range is not None:
        t_min, t_max = app_cfg.odg.odg_time_range
        dt_min = pd.to_datetime(int(t_min), unit="ns")
        dt_max = pd.to_datetime(int(t_max), unit="ns")
        app_cfg.state.range_start = dt_min.strftime("%Y-%m-%d-%H-%M")
        app_cfg.state.range_end = dt_max.strftime("%Y-%m-%d-%H-%M")

    app_cfg.state.odg_confirmed = True
    app_cfg.state.plant_confirmed = True
    app_cfg.state.confirm_sample_inspector = result.inspector
    overlap_error = validate_sampling_overlap(app_cfg)
    if overlap_error:
        reset_odg_state(app_cfg)
        reset_plant_state(app_cfg)
        return overlap_error
    app_cfg.dump_ini()
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
        return RunResult(ok=False, error=str(ex), error_code="INVALID_TIME_RANGE")

    try:
        metrics, detail_all, profile = compute_accuracy_for_all_odg_mp(
            app_cfg.plant.plant_id_signal_map,
            app_cfg.plant.plant_data_files,
            app_cfg.odg.odg_files,
            app_cfg.compare,
            progress_cb=progress_cb,
            time_range=time_range,
            plant_data_encoding=app_cfg.plant.plant_data_encoding,
            odg_encoding=app_cfg.odg.odg_encoding,
        )
    except Exception as ex:
        return RunResult(ok=False, error=str(ex), error_code="RUN_FAILED")

    return RunResult(
        ok=True,
        summary=metrics["matching_score"],
        matching_score=metrics["matching_score"],
        rmse=metrics["rmse"],
        correlation=metrics["correlation"],
        offset_ms=metrics["offset_ms"],
        detail=detail_all,
        compute_inspector=profile,
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
    plant_cfg.plant_skipped_mappings = None
    plant_cfg.plant_map_encoding = None
    plant_cfg.plant_data_encoding = None
