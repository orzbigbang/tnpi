import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import pandas as pd

from state import CompareConfig
from utils import compute_compare_metrics, compute_signal_match_stats


# =============================================================================
# 多进程 worker：单个 high CSV 的 compare 计算
# =============================================================================

def _compute_one(
    name: str,
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
    cfg: CompareConfig,
    time_range,
) -> Tuple[str, dict, pd.DataFrame, str]:
    """
    子进程执行：对单个 high CSV 计算 compare 指标。

    设计目的：
    - ProcessPoolExecutor 里 submit 的函数必须是 top-level（可 pickling）
    - 返回值要可 pickling：这里返回 dict + DataFrame + str 都可序列化/传回主进程

    返回：
    - name: high 文件名（用于主进程聚合）
    - metrics: dict（compute_compare_metrics 的 metrics）
    - detail: DataFrame（每信号一行）
    - err: str（失败时错误消息；成功返回 ""）

    注意：
    - 这里 catch Exception，保证单个文件失败不会导致整个批处理失败
    """
    try:
        metrics, detail = compute_compare_metrics(
            low_map_bytes,
            low_data_bytes,
            high_bytes,
            cfg,
            time_range,
        )
        return name, metrics, detail, ""
    except Exception as ex:
        # 失败时返回空 metrics + 空 detail，并携带错误字符串
        return name, {}, pd.DataFrame(), str(ex)


# =============================================================================
# 批量 compare：对所有 high CSV 并行计算 accuracy（matching_score 等）
# =============================================================================

def compute_accuracy_for_all_high_mp(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_files: Dict[str, bytes],
    cfg: CompareConfig,
    max_workers: int = 0,
    progress_cb=None,
    time_range=None,
) -> Tuple[dict, pd.DataFrame]:
    """
    多进程批量计算：对 high_files 中每个 high CSV 执行 compare，并聚合结果。

    输入：
    - low_map_bytes / low_data_bytes：PlantDB 侧合并后的 bytes
    - high_files: dict[name] = csv_bytes（ODG 侧多个文件）
    - cfg: CompareConfig（比较算法参数）
    - max_workers:
        - 0 表示自动 = os.cpu_count()（或 1）
    - progress_cb: 可选回调 progress_cb(done, total)
        - 用于 UI 显示进度条
    - time_range: 可选时间窗，只在该范围内比较

    输出：
    - avg: dict
        - matching_score/rmse/correlation/offset_ms 的“跨文件均值”
        - 注意：这是对每个 high 文件算出来的 summary 再平均，并非按点/按信号加权
    - detail_all: DataFrame
        - 将每个 high 的 detail（每信号一行）合并，并插入 high_csv 列
        - 方便前端做 drill-down（按文件/按信号查看）

    重要行为/坑点：
    - as_completed() 是“哪个先算完先返回”，所以 summary_rows 的追加是乱序的；
      最后用 sort_values("high_csv") 做了排序输出
    - DataFrame 在进程间传输有开销；high_files 很大时序列化成本会明显
      （但你这里的逻辑保持简单可靠）
    """
    # worker 数量：显式指定优先，否则用 CPU 核数
    worker_count = max_workers or (os.cpu_count() or 1)

    # summary_rows：每个 high 文件一行（文件级别汇总）
    summary_rows = []

    # detail_rows：每个 high 文件一个 detail DataFrame（信号级别详情）
    detail_rows = []

    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        # 1) 提交所有任务（每个 high 文件一个 future）
        futures = [
            ex.submit(
                _compute_one,
                name,
                low_map_bytes,
                low_data_bytes,
                high_files[name],
                cfg,
                time_range,
            )
            for name in high_files
        ]

        # 2) 初始化进度
        completed = 0
        total = len(futures)
        if progress_cb:
            progress_cb(0, total)

        # 3) 按完成顺序收集结果（无序）
        for fut in as_completed(futures):
            name, metrics, detail, err = fut.result()

            # 4) 将每个 high 文件的 summary 行加入汇总列表
            #    metrics 可能为空（单文件失败），所以用 get + nan 兜底
            summary_rows.append(
                {
                    "high_csv": name,
                    "matching_score": float(metrics.get("matching_score", float("nan"))),
                    "rmse": float(metrics.get("rmse", float("nan"))),
                    "correlation": float(metrics.get("correlation", float("nan"))),
                    "offset_ms": float(metrics.get("offset_ms", float("nan"))),
                    # n_signals：该文件成功比较的信号数（detail 行数）
                    "n_signals": int(detail.shape[0]) if not detail.empty else 0,
                    # error：成功为空字符串；失败为异常信息
                    "error": err or "",
                }
            )

            # 5) 合并 detail：插入 high_csv 列，方便后续按文件分组
            if not detail.empty:
                detail = detail.copy()
                detail.insert(0, "high_csv", name)
                detail_rows.append(detail)

            # 6) 更新进度
            completed += 1
            if progress_cb:
                progress_cb(completed, total)

    # 7) 构建 summary DataFrame 并排序（按文件名）
    summary = pd.DataFrame(summary_rows).sort_values("high_csv")

    # 8) 计算跨文件均值（忽略 NaN）
    #    注意：这是“文件级均值”。不是按信号数或点数加权。
    avg = {
        "matching_score": float(summary["matching_score"].mean())
        if summary["matching_score"].notna().any()
        else float("nan"),
        "rmse": float(summary["rmse"].mean())
        if summary["rmse"].notna().any()
        else float("nan"),
        "correlation": float(summary["correlation"].mean())
        if summary["correlation"].notna().any()
        else float("nan"),
        "offset_ms": float(summary["offset_ms"].mean())
        if summary["offset_ms"].notna().any()
        else float("nan"),
    }

    # 9) 合并所有文件的 detail（信号级详情）
    if detail_rows:
        detail_all = pd.concat(detail_rows, ignore_index=True)
    else:
        detail_all = pd.DataFrame()

    return avg, detail_all


# =============================================================================
# 多进程 worker：单个 high CSV 的信号集合匹配统计
# =============================================================================

def _match_one(
    name: str,
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_bytes: bytes,
) -> Dict[str, object]:
    """
    子进程执行：对单个 high CSV 统计 low/high 信号集合的重叠情况。

    返回：
    - stats dict（含 low_signals/high_signals/overlap/missing...）
    - 并附加 high_csv=name

    失败：
    - 返回一组全 0，并带 error 字段
    """
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


# =============================================================================
# 批量信号匹配统计：对所有 high CSV 并行统计信号集合重叠
# =============================================================================

def compute_signal_match_all_mp(
    low_map_bytes: bytes,
    low_data_bytes: bytes,
    high_files: Dict[str, bytes],
    max_workers: int = 0,
    progress_cb=None,
) -> pd.DataFrame:
    """
    多进程批量统计：对每个 high CSV 计算信号集合重叠情况。

    输出：
    - DataFrame，每个 high 文件一行，按 high_csv 排序

    用途：
    - 在“正式 compare 前”快速判断：信号名是否对得上
    - 发现某些文件缺少大量信号，或低侧 mapping 不一致等
    """
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

    # rows 是乱序（as_completed），最后排序输出稳定结果
    return pd.DataFrame(rows).sort_values("high_csv")
