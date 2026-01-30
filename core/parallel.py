import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from state import CompareConfig
from core.metrics import compute_compare_metrics
from core.parse import parse_plant_samples_csv

_PLANT_CACHE: Optional[Tuple[pd.Series, pd.DataFrame]] = None


def _init_worker(
    id_to_signal: Dict[str, Union[str, Tuple[str, object]]],
    plant_data_path: str,
    plant_data_encoding: Optional[str],
) -> None:
    global _PLANT_CACHE
    _PLANT_CACHE = parse_plant_samples_csv(
        plant_data_path,
        id_to_signal,
        encoding=plant_data_encoding,
    )


def _compute_one(
    odg_path: str,
    cfg: CompareConfig,
    time_range: Optional[Tuple[float, float]],
    odg_encoding: Optional[str],
) -> Tuple[str, dict, pd.DataFrame, str]:
    try:
        if _PLANT_CACHE is None:
            raise ValueError("Plant data cache is not initialized.")
        metrics, detail = compute_compare_metrics(
            {},
            "",
            odg_path,
            cfg,
            time_range,
            plant_cache=_PLANT_CACHE,
            odg_encoding=odg_encoding,
        )
        return os.path.basename(odg_path), metrics, detail, ""
    except Exception as ex:
        return os.path.basename(odg_path), {}, pd.DataFrame(), str(ex)


def compute_accuracy_for_all_odg_mp(
    id_to_signal: Dict[str, Union[str, Tuple[str, object]]],
    plant_data_files: list[str],
    odg_files: list[str],
    cfg: CompareConfig,
    max_workers: int = 0,
    progress_cb=None,
    time_range: Optional[Tuple[float, float]] = None,
    *,
    plant_data_encoding: Optional[str] = None,
    odg_encoding: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    worker_count = max_workers or (os.cpu_count() or 1)
    if not id_to_signal or not plant_data_files:
        raise ValueError("PlantDB ID-to-signal mapping and data CSV paths are required.")
    plant_data_path = plant_data_files[0]
    summary_rows = []
    detail_rows = []

    _init_worker(id_to_signal, plant_data_path, plant_data_encoding)
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = [
            ex.submit(_compute_one, odg_path, cfg, time_range, odg_encoding)
            for odg_path in odg_files
        ]
        completed = 0
        total = len(futures)
        if progress_cb:
            progress_cb(0, total)
        for fut in as_completed(futures):
            name, metrics, detail, err = fut.result()
            summary_rows.append(
                {
                    "odg_csv": name,
                    "matching_score": float(metrics.get("matching_score", float("nan"))),
                    "rmse": float(metrics.get("rmse", float("nan"))),
                    "correlation": float(metrics.get("correlation", float("nan"))),
                    "offset_ms": float(metrics.get("offset_ms", float("nan"))),
                    "n_signals": int(detail.shape[0]) if not detail.empty else 0,
                    "error": err or "",
                }
            )
            if not detail.empty:
                detail = detail.copy()
                detail.insert(0, "odg_csv", name)
                detail_rows.append(detail)
            completed += 1
            if progress_cb:
                progress_cb(completed, total)

    summary = pd.DataFrame(summary_rows).sort_values("odg_csv")
    avg = {
        "matching_score": float(summary["matching_score"].mean()) if summary["matching_score"].notna().any() else float("nan"),
        "rmse": float(summary["rmse"].mean()) if summary["rmse"].notna().any() else float("nan"),
        "correlation": float(summary["correlation"].mean()) if summary["correlation"].notna().any() else float("nan"),
        "offset_ms": float(summary["offset_ms"].mean()) if summary["offset_ms"].notna().any() else float("nan"),
    }

    if detail_rows:
        detail_all = pd.concat(detail_rows, ignore_index=True)
    else:
        detail_all = pd.DataFrame()

    return avg, detail_all
