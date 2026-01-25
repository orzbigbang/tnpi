import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import pandas as pd

from state import CompareConfig
from utils import compute_compare_metrics, compute_signal_match_stats


def _compute_one(
    name: str,
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
    cfg: CompareConfig,
    time_range,
) -> Tuple[str, dict, pd.DataFrame, str]:
    try:
        metrics, detail = compute_compare_metrics(low_map_bytes, low_data_bytes, high_bytes, cfg, time_range)
        return name, metrics, detail, ""
    except Exception as ex:
        return name, {}, pd.DataFrame(), str(ex)


def compute_accuracy_for_all_high_mp(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_files: Dict[str, bytes],
    cfg: CompareConfig,
    max_workers: int = 0,
    progress_cb=None,
    time_range=None,
) -> Tuple[dict, pd.DataFrame]:
    worker_count = max_workers or (os.cpu_count() or 1)
    summary_rows = []
    detail_rows = []

    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        futures = [
            ex.submit(_compute_one, name, low_map_bytes, low_data_bytes, high_files[name], cfg, time_range)
            for name in high_files
        ]
        completed = 0
        total = len(futures)
        if progress_cb:
            progress_cb(0, total)
        for fut in as_completed(futures):
            name, metrics, detail, err = fut.result()
            summary_rows.append(
                {
                    "high_csv": name,
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
                detail.insert(0, "high_csv", name)
                detail_rows.append(detail)
            completed += 1
            if progress_cb:
                progress_cb(completed, total)

    summary = pd.DataFrame(summary_rows).sort_values("high_csv")
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


def _match_one(
    name: str,
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
) -> Dict[str, object]:
    try:
        stats = compute_signal_match_stats(low_map_bytes, low_data_bytes, high_bytes)
        stats["high_csv"] = name
        return stats
    except Exception as ex:
        return {
            "high_csv": name,
            "low_signals": 0,
            "high_signals": 0,
            "overlap": 0,
            "missing_in_high": 0,
            "missing_in_low": 0,
            "error": str(ex),
        }


def compute_signal_match_all_mp(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_files: Dict[str, bytes],
    max_workers: int = 0,
    progress_cb=None,
) -> pd.DataFrame:
    worker_count = max_workers or (os.cpu_count() or 1)
    rows = []
    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        futures = [
            ex.submit(_match_one, name, low_map_bytes, low_data_bytes, high_files[name])
            for name in high_files
        ]
        completed = 0
        total = len(futures)
        if progress_cb:
            progress_cb(0, total)
        for fut in as_completed(futures):
            rows.append(fut.result())
            completed += 1
            if progress_cb:
                progress_cb(completed, total)
    return pd.DataFrame(rows).sort_values("high_csv")
