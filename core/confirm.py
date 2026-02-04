import os
import time
from typing import Optional

import pandas as pd

from core.errors import ConfirmError
from core.io_csv import detect_csv_encoding
from core.models import ConfirmResult, OdgMeta, PlantMeta
from core.parse import (
    compute_odg_meta,
    compute_plant_time_range_and_signal_count,
    load_odg_folder_from_dir,
    load_plant_folder_from_dir,
)


def confirm_samples(odg_folder: str, plant_folder: str) -> ConfirmResult:
    t0 = time.perf_counter()
    inspector: dict[str, float] = {}
    odg_path = (odg_folder or "").strip()
    plant_path = (plant_folder or "").strip()
    if not odg_path or not plant_path:
        return ConfirmResult(
            ok=False,
            error="Please enter both odg and plant folder paths before confirming.",
            error_code="CONFIRM_MISSING_PATH",
        )
    odg_meta = None
    plant_meta = None
    try:
        odg_meta = confirm_odg_folder(odg_path, inspector=inspector)
        plant_meta = confirm_plant_folder(plant_path, inspector=inspector)
    except ConfirmError as ex:
        return ConfirmResult(
            ok=False,
            error=str(ex),
            error_code=ex.code,
            odg_meta=odg_meta,
            plant_meta=plant_meta,
            inspector=inspector,
        )

    inspector["confirm_total_ms"] = (time.perf_counter() - t0) * 1000.0
    return ConfirmResult(
        ok=True,
        odg_meta=odg_meta,
        plant_meta=plant_meta,
        inspector=inspector,
    )


def confirm_odg_folder(
    folder_path: str,
    *,
    inspector: Optional[dict[str, float]] = None,
) -> OdgMeta:
    if not folder_path:
        raise ConfirmError("Please enter a folder path before confirming.", "ODG_MISSING_PATH")
    try:
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        odg_files = load_odg_folder_from_dir(folder_path)
        if inspector is not None:
            inspector["odg_load_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        encodings = {detect_csv_encoding(p) for p in odg_files}
        if inspector is not None:
            inspector["odg_detect_encoding_ms"] = (time.perf_counter() - t2) * 1000.0
        if len(encodings) > 1:
            raise ConfirmError(
                f"ODG CSV encodings are inconsistent: {sorted(encodings)}",
                "ODG_ENCODING_MISMATCH",
            )
        odg_encoding = next(iter(encodings)) if encodings else None
        odg_names = [os.path.basename(p) for p in odg_files]

        t3 = time.perf_counter()
        odg_time_range, odg_signal_count, odg_row_count = compute_odg_meta(
            odg_files, encoding=odg_encoding
        )
        if inspector is not None:
            inspector["odg_meta_ms"] = (time.perf_counter() - t3) * 1000.0

        t4 = time.perf_counter()
        if odg_time_range is not None:
            t_min, t_max = odg_time_range
            _ = pd.to_datetime(int(t_min), unit="ns")
            _ = pd.to_datetime(int(t_max), unit="ns")
        if inspector is not None:
            inspector["odg_range_format_ms"] = (time.perf_counter() - t4) * 1000.0
            inspector["odg_total_ms"] = (time.perf_counter() - t0) * 1000.0
    except ConfirmError:
        raise
    except Exception as ex:
        raise ConfirmError(f"ODG confirm failed: {ex}", "ODG_CONFIRM_FAILED") from ex

    return OdgMeta(
        files=odg_files,
        names=odg_names,
        time_range=odg_time_range,
        signal_count=odg_signal_count,
        row_count=odg_row_count,
        encoding=odg_encoding,
    )


def confirm_plant_folder(
    folder_path: str,
    *,
    inspector: Optional[dict[str, float]] = None,
) -> PlantMeta:
    if not folder_path:
        raise ConfirmError("Please enter a folder path before confirming.", "PLANT_MISSING_PATH")
    try:
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        (
            plant_map_files,
            plant_data_files,
            plant_map_count,
            plant_data_count,
            plant_data_rows,
        ) = load_plant_folder_from_dir(folder_path)
        if inspector is not None:
            inspector["plant_load_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        map_encodings = {detect_csv_encoding(p) for p in plant_map_files}
        if inspector is not None:
            inspector["plant_map_encoding_ms"] = (time.perf_counter() - t2) * 1000.0
        if len(map_encodings) > 1:
            raise ConfirmError(
                f"Plant map CSV encodings are inconsistent: {sorted(map_encodings)}",
                "PLANT_MAP_ENCODING_MISMATCH",
            )
        plant_map_encoding = next(iter(map_encodings)) if map_encodings else None

        t3 = time.perf_counter()
        data_encodings = {detect_csv_encoding(p) for p in plant_data_files}
        if inspector is not None:
            inspector["plant_data_encoding_ms"] = (time.perf_counter() - t3) * 1000.0
        if len(data_encodings) > 1:
            raise ConfirmError(
                f"Plant data CSV encodings are inconsistent: {sorted(data_encodings)}",
                "PLANT_DATA_ENCODING_MISMATCH",
            )
        plant_data_encoding = next(iter(data_encodings)) if data_encodings else None

        t4 = time.perf_counter()
        (
            plant_time_range,
            plant_signal_count,
            plant_id_signal_map,
            plant_skipped_mappings,
        ) = compute_plant_time_range_and_signal_count(
            plant_map_files,
            plant_data_files,
            map_encoding=plant_map_encoding,
            data_encoding=plant_data_encoding,
        )
        if inspector is not None:
            inspector["plant_meta_ms"] = (time.perf_counter() - t4) * 1000.0
            inspector["plant_total_ms"] = (time.perf_counter() - t0) * 1000.0
    except ConfirmError:
        raise
    except Exception as ex:
        raise ConfirmError(f"PlantDB confirm failed: {ex}", "PLANT_CONFIRM_FAILED") from ex

    return PlantMeta(
        map_files=plant_map_files,
        data_files=plant_data_files,
        map_count=plant_map_count,
        data_count=plant_data_count,
        data_rows=plant_data_rows,
        skipped_mappings=plant_skipped_mappings,
        map_encoding=plant_map_encoding,
        data_encoding=plant_data_encoding,
        time_range=plant_time_range,
        signal_count=plant_signal_count,
        id_signal_map=plant_id_signal_map,
    )
