from dataclasses import dataclass, field
import configparser
import os
from typing import Any, Dict, List, Optional, Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class CompareConfig:
    time_col: str
    accuracy_denominator: Literal["range", "std", "abs_truth"]
    out_of_range_policy: Literal["drop", "clip"]
    aggregate_policy: Literal["mean", "min"]


@dataclass
class RunResult:
    ok: bool
    summary: Optional[float] = None
    matching_score: Optional[float] = None
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    offset_ms: Optional[float] = None
    match_stats: Optional[dict] = None
    error: Optional[str] = None
    detail: Optional["pd.DataFrame"] = None

    def dump_result(self) -> Tuple[str, bytes]:
        import pandas as pd

        if self.detail is None:
            df = pd.DataFrame(
                columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"]
            )
        else:
            df = self.detail.copy()
        if "offset_ms" not in df.columns:
            df["offset_ms"] = self.offset_ms

        df = df.reindex(columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"])

        out = df.rename(
            columns={"rmse": "RMSE", "offset_ms": "Offset"}
        )
        buf = out.to_csv(index=False, encoding="utf-8-sig")
        return "run_result.csv", buf.encode("utf-8-sig")


@dataclass
class MetadataConfig:
    project_name: str = "TNPI"
    version: str = "0.1.0"


@dataclass
class StateConfig:
    run_mode: str = "multiprocess"
    is_running: bool = False
    range_start: str = ""
    range_end: str = ""
    range_mode: str = "full"
    high_folder: str = ""
    low_folder: str = ""
    high_confirmed: bool = False
    low_confirmed: bool = False
    last_result: Optional["RunResult"] = None


@dataclass
class HighConfig:
    high_files: Optional[Dict[str, bytes]] = None
    high_names: List[str] = field(default_factory=list)
    high_time_range: Optional[Tuple[float, float]] = None
    high_signal_count: Optional[int] = None
    high_row_count: Optional[int] = None


@dataclass
class LowConfig:
    low_time_range: Optional[Tuple[float, float]] = None
    low_signal_count: Optional[int] = None
    low_map_bytes: Optional[bytes] = None
    low_data_bytes: Optional[bytes] = None
    low_map_count: Optional[int] = None
    low_data_count: Optional[int] = None
    low_data_rows: Optional[int] = None


@dataclass
class AppConfig:
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    state: StateConfig = field(default_factory=StateConfig)
    high: HighConfig = field(default_factory=HighConfig)
    low: LowConfig = field(default_factory=LowConfig)
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
        base_dir = os.getenv("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base_dir, "TNPI", "app_config.ini")

    def dump_ini(self, path: Optional[str] = None) -> str:
        ini_path = path or self._default_ini_path()
        os.makedirs(os.path.dirname(ini_path), exist_ok=True)
        parser = configparser.ConfigParser()
        parser["state"] = {
            "run_mode": self.state.run_mode,
            "range_mode": self.state.range_mode,
            "high_folder": self.state.high_folder,
            "low_folder": self.state.low_folder,
        }
        parser["compare"] = {
            "accuracy_denominator": self.compare.accuracy_denominator,
            "out_of_range_policy": self.compare.out_of_range_policy,
            "aggregate_policy": self.compare.aggregate_policy,
        }
        with open(ini_path, "w", encoding="utf-8") as handle:
            parser.write(handle)
        return ini_path

    @classmethod
    def load_ini(cls, path: Optional[str] = None) -> "AppConfig":
        ini_path = path or cls._default_ini_path()
        if not os.path.exists(ini_path):
            return cls()
        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding="utf-8")
        config = cls()
        if parser.has_section("state"):
            config.state.run_mode = parser.get("state", "run_mode", fallback=config.state.run_mode)
            config.state.range_mode = parser.get("state", "range_mode", fallback=config.state.range_mode)
            config.state.high_folder = parser.get("state", "high_folder", fallback=config.state.high_folder)
            config.state.low_folder = parser.get("state", "low_folder", fallback=config.state.low_folder)
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


def get_config(session_state: Any) -> AppConfig:
    if "app_config" not in session_state or not isinstance(session_state["app_config"], AppConfig):
        try:
            session_state["app_config"] = AppConfig.load_ini()
        except Exception:
            session_state["app_config"] = AppConfig()
    return session_state["app_config"]
