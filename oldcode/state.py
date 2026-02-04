from dataclasses import dataclass, field
import configparser
import os
from typing import Any, Dict, List, Optional, Tuple, Literal, TYPE_CHECKING

# TYPE_CHECKING 用于“类型检查阶段导入”，避免运行时导入成本/循环依赖
# 这里 detail 字段类型写成 "pd.DataFrame"，但运行时并不强制依赖 pandas
if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# CompareConfig：比较算法配置（会被 compute_compare_metrics 使用）
# =============================================================================

@dataclass
class CompareConfig:
    # 时间列配置。你这里默认会给 "auto"，但类型标注写的是 str：
    # - "auto"：由解析逻辑自动检测时间列（或用 Date+Time 拼）
    # - 或者用户指定具体列名
    time_col: str

    # accuracy_denominator：误差归一化分母的选择
    # - "range": denom = max(x_high) - min(x_high)（每个信号整体尺度）
    # - "std":   denom = std(x_high)
    # - "abs_truth": denom = |x_hat|（逐点尺度，真值越小越敏感）
    accuracy_denominator: Literal["range", "std", "abs_truth"]

    # out_of_range_policy：当低采样时间点落在高采样范围外时怎么办
    # - "drop": 该点插值结果视为 NaN，后续比较时会被过滤掉
    # - "clip": 将 t_low 裁剪到 [t_high_min, t_high_max] 再插值（范围外也给结果）
    out_of_range_policy: Literal["drop", "clip"]

    # aggregate_policy：将逐点 accuracy 聚合为“该信号的匹配度”
    # - "mean": 平均准确度（整体表现）
    # - "min":  最差点准确度（更严格，反映最坏情况）
    aggregate_policy: Literal["mean", "min"]


# =============================================================================
# RunResult：一次运行的结果对象（用于 UI 展示/导出）
# =============================================================================

@dataclass
class RunResult:
    # ok：这次运行是否成功（UI 用来判断展示 success/error）
    ok: bool

    # summary：你可能用于 UI 顶部展示的“总览得分”（可选）
    summary: Optional[float] = None

    # matching_score：整体匹配度（通常是每信号匹配度的均值）
    matching_score: Optional[float] = None

    # rmse：整体 RMSE（通常是每信号 RMSE 的均值）
    rmse: Optional[float] = None

    # correlation：整体相关系数均值（用于判断趋势是否一致）
    correlation: Optional[float] = None

    # offset_ms：估计出的时间偏移（毫秒）
    offset_ms: Optional[float] = None

    # match_stats：信号集合统计信息（low/high 各多少信号、重叠多少等）
    match_stats: Optional[dict] = None

    # error：失败时的错误字符串（用于 UI 显示）
    error: Optional[str] = None

    # detail：每个信号一行的详细结果（DataFrame）
    # 使用字符串注解避免运行时强依赖 pandas
    detail: Optional["pd.DataFrame"] = None

    def dump_result(self) -> Tuple[str, bytes]:
        """
        将 RunResult 导出为 CSV（bytes），用于“下载结果”。

        返回：
        - (filename, file_bytes)

        设计点：
        - 即使 detail 为空，也输出一个固定表头的空 CSV
        - 强制保证输出列顺序：signal, matching_score, correlation, rmse, offset_ms
        - 将 rmse/offset_ms 重命名成更适合用户看的 "RMSE"/"Offset"
        - 使用 utf-8-sig：Excel 直接打开不乱码（带 BOM）
        """
        import pandas as pd  # 这里局部导入，避免模块加载时就依赖 pandas

        # detail 为空则构造空表，保证导出功能不崩
        if self.detail is None:
            df = pd.DataFrame(
                columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"]
            )
        else:
            df = self.detail.copy()

        # 如果 detail 没带 offset_ms 列，则从 RunResult.offset_ms 补一列
        # 注意：这里会让整列都是同一个 offset_ms（符合 UI 展示“整体 offset”的需求）
        if "offset_ms" not in df.columns:
            df["offset_ms"] = self.offset_ms

        # 固定输出列顺序（避免 detail 里多列导致列顺序乱）
        df = df.reindex(columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"])

        # 更友好的列名（面向用户）
        out = df.rename(
            columns={"rmse": "RMSE", "offset_ms": "Offset"}
        )

        # DataFrame.to_csv() 返回 str，这里再 encode 成 bytes
        buf = out.to_csv(index=False, encoding="utf-8-sig")
        return "run_result.csv", buf.encode("utf-8-sig")


# =============================================================================
# MetadataConfig：应用元信息（展示用途）
# =============================================================================

@dataclass
class MetadataConfig:
    # 项目名、版本号通常用于 UI footer/标题栏/日志
    project_name: str = "TNPI"
    version: str = "0.1.0"


# =============================================================================
# StateConfig：运行状态/会话状态（更像 UI state，而不是算法参数）
# =============================================================================

@dataclass
class StateConfig:
    # run_mode：运行模式，通常用于选择线程/进程/单进程等
    # 默认 "multiprocess"
    run_mode: str = "multiprocess"

    # is_running：UI 用来禁用按钮/显示 spinner
    is_running: bool = False

    # range_start/range_end：用户输入的时间范围（字符串形式，可能来自 UI 文本框）
    # 注意：这里并不是 float/ns，而是“输入态”
    range_start: str = ""
    range_end: str = ""

    # range_mode：时间范围模式
    # - "full"：全量
    # - 或 "custom"：使用 range_start/range_end（具体取决于你 UI 逻辑）
    range_mode: str = "full"

    # high_folder/low_folder：用户选择的目录路径（ODG/PlantDB）
    high_folder: str = ""
    low_folder: str = ""

    # high_confirmed/low_confirmed：UI 确认态（例如“已加载并确认”后置 true）
    high_confirmed: bool = False
    low_confirmed: bool = False

    # last_result：上一次运行结果（用于 UI 直接显示/下载）
    last_result: Optional["RunResult"] = None


# =============================================================================
# HighConfig：高采样数据（ODG）解析后的缓存信息
# =============================================================================

@dataclass
class HighConfig:
    # high_files：文件名 -> bytes（zip 解出来/目录读出来）
    # Optional：没加载时为 None
    high_files: Optional[Dict[str, bytes]] = None

    # high_names：文件名列表（便于 UI 展示）
    high_names: List[str] = field(default_factory=list)

    # high_time_range：整体时间范围 (min, max)，单位跟解析一致（通常 ns float）
    high_time_range: Optional[Tuple[float, float]] = None

    # high_signal_count：高采样总信号数（列去重/合并统计结果）
    high_signal_count: Optional[int] = None

    # high_row_count：累计行数（用于提示数据规模）
    high_row_count: Optional[int] = None


# =============================================================================
# LowConfig：低采样数据（PlantDB）解析后的缓存信息
# =============================================================================

@dataclass
class LowConfig:
    # low_time_range：低采样时间范围
    low_time_range: Optional[Tuple[float, float]] = None

    # low_signal_count：低采样信号数（pivot 后宽表列数）
    low_signal_count: Optional[int] = None

    # low_map_bytes/low_data_bytes：合并后的 mapping/data CSV bytes
    # 这里做 bytes 缓存的好处：
    # - 后面 compute_compare_metrics 接口就是 bytes
    # - 不需要把中间 DataFrame 存进 session_state（更轻、更可序列化）
    low_map_bytes: Optional[bytes] = None
    low_data_bytes: Optional[bytes] = None

    # low_map_count：识别到的 mapping CSV 文件数
    low_map_count: Optional[int] = None

    # low_data_count：识别到的 data CSV 文件数
    low_data_count: Optional[int] = None

    # low_data_rows：合并后 data 行数（用于提示规模）
    low_data_rows: Optional[int] = None


# =============================================================================
# AppConfig：应用的“总配置树”（metadata/state/high/low/compare）
# =============================================================================

@dataclass
class AppConfig:
    # metadata/state/high/low/compare 都用 default_factory
    # 好处：避免 dataclass 默认值共享（尤其是 list/dict）
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    state: StateConfig = field(default_factory=StateConfig)
    high: HighConfig = field(default_factory=HighConfig)
    low: LowConfig = field(default_factory=LowConfig)

    # compare：算法参数默认值
    # 注意：CompareConfig 要求四个字段，这里用 lambda 构造默认实例
    compare: CompareConfig = field(
        default_factory=lambda: CompareConfig(
            time_col="auto",
            accuracy_denominator="range",
            out_of_range_policy="drop",
            aggregate_policy="mean",
        )
    )

    @staticmethod
    def _default_ini_path() -> str:
        """
        返回默认 ini 配置文件路径（Windows / LOCALAPPDATA 优先）。

        例：
        - %LOCALAPPDATA%\\TNPI\\app_config.ini
        - 或 fallback 到 ~\\AppData\\Local\\TNPI\\app_config.ini

        设计意图：
        - 适合 exe（PyInstaller）落地配置
        - 无需管理员权限（写在用户目录）
        """
        base_dir = os.getenv("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base_dir, "TNPI", "app_config.ini")

    def dump_ini(self, path: Optional[str] = None) -> str:
        """
        将当前配置持久化到 ini 文件。

        只写入“需要跨启动保存”的内容：
        - state: run_mode/range_mode/high_folder/low_folder
        - compare: accuracy_denominator/out_of_range_policy/aggregate_policy

        注意：
        - 不持久化 high_files/low_data_bytes 等大对象（不适合写入 ini）
        - 不持久化 is_running/last_result 这种“会话态”
        """
        ini_path = path or self._default_ini_path()
        os.makedirs(os.path.dirname(ini_path), exist_ok=True)

        parser = configparser.ConfigParser()

        # UI 状态里“值得保存的部分”
        parser["state"] = {
            "run_mode": self.state.run_mode,
            "range_mode": self.state.range_mode,
            "high_folder": self.state.high_folder,
            "low_folder": self.state.low_folder,
        }

        # 算法参数
        parser["compare"] = {
            "accuracy_denominator": self.compare.accuracy_denominator,
            "out_of_range_policy": self.compare.out_of_range_policy,
            "aggregate_policy": self.compare.aggregate_policy,
        }

        # ini 是文本文件；写 utf-8
        with open(ini_path, "w", encoding="utf-8") as handle:
            parser.write(handle)

        return ini_path

    @classmethod
    def load_ini(cls, path: Optional[str] = None) -> "AppConfig":
        """
        从 ini 文件加载配置；文件不存在则返回默认 AppConfig()。

        设计点：
        - 先构造默认 config，再用 ini 覆盖（fallback 保留默认值）
        - 只读取 state/compare 两个 section
        - 对 compare 的值读出来是 str，但你的类型是 Literal[...]：
          这里不做强校验，假如 ini 写错了，会把错误值塞进去（潜在风险）
          （要严格的话可以加白名单校验/回退）
        """
        ini_path = path or cls._default_ini_path()
        if not os.path.exists(ini_path):
            return cls()

        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding="utf-8")

        config = cls()

        # 读取 state section
        if parser.has_section("state"):
            config.state.run_mode = parser.get("state", "run_mode", fallback=config.state.run_mode)
            config.state.range_mode = parser.get("state", "range_mode", fallback=config.state.range_mode)
            config.state.high_folder = parser.get("state", "high_folder", fallback=config.state.high_folder)
            config.state.low_folder = parser.get("state", "low_folder", fallback=config.state.low_folder)

        # 读取 compare section
        if parser.has_section("compare"):
            config.compare.accuracy_denominator = parser.get(
                "compare", "accuracy_denominator", fallback=config.compare.accuracy_denominator
            )
            config.compare.out_of_range_policy = parser.get(
                "compare", "out_of_range_policy", fallback=config.compare.out_of_range_policy
            )
            config.compare.aggregate_policy = parser.get(
                "compare", "aggregate_policy", fallback=config.compare.aggregate_policy
            )

        return config


# =============================================================================
# get_config：从 session_state 获取 AppConfig（懒加载 + 容错）
# =============================================================================

def get_config(session_state: Any) -> AppConfig:
    """
    从 session_state（例如 Streamlit st.session_state）获取 AppConfig。

    行为：
    1) 如果 session_state["app_config"] 不存在或类型不对：
       - 尝试从 ini 加载
       - 加载失败则回退到默认 AppConfig()
    2) 返回 session_state["app_config"]

    设计意图：
    - Streamlit 重新运行脚本时 session_state 持久，但首次启动需要加载 ini
    - 防止 ini 损坏/字段异常导致整个 App 崩
    """
    if "app_config" not in session_state or not isinstance(session_state["app_config"], AppConfig):
        try:
            session_state["app_config"] = AppConfig.load_ini()
        except Exception:
            session_state["app_config"] = AppConfig()
    return session_state["app_config"]
