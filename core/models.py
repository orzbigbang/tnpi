from dataclasses import dataclass, field
import configparser
import os
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd

from enum_types import SignalType


@dataclass
class RunResult:
    ok: bool
    summary: Optional[float] = None
    matching_score: Optional[float] = None
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    offset_ms: Optional[float] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    detail: Optional["pd.DataFrame"] = None
    compute_inspector: Optional[Dict[str, float]] = None

    def dump_result(self) -> Tuple[str, bytes]:
        if self.detail is None:
            df = pd.DataFrame(
                columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"]
            )
        else:
            df = self.detail.copy()
        if "offset_ms" not in df.columns:
            df["offset_ms"] = self.offset_ms

        df = df.reindex(columns=["signal", "matching_score", "correlation", "rmse", "offset_ms"])
        out = df.rename(columns={"rmse": "RMSE", "offset_ms": "Offset"})
        buf = out.to_csv(index=False, encoding="utf-8-sig")
        return "run_result.csv", buf.encode("utf-8-sig")


@dataclass
class CompareConfig:
    time_col: str = "auto"
    accuracy_denominator: Literal["range", "std", "abs_truth"] = "range"
    out_of_range_policy: Literal["drop", "clip"] = "drop"
    aggregate_policy: Literal["mean", "min"] = "mean"


@dataclass
class MetadataConfig:
    project_name: str = "TNPI"
    version: str = "0.1.0"


@dataclass
class StateConfig:
    is_confirming: bool = False
    is_running: bool = False
    range_start: str = ""
    range_end: str = ""
    range_mode: str = "full"
    odg_folder: str = ""
    plant_folder: str = ""
    odg_confirmed: bool = False
    plant_confirmed: bool = False
    last_result: Optional["RunResult"] = None
    confirm_sample_inspector: Optional[Dict[str, float]] = None


@dataclass
class OdgConfig:
    odg_files: List[str] = field(default_factory=list)
    odg_names: List[str] = field(default_factory=list)
    odg_time_range: Optional[Tuple[float, float]] = None
    odg_signal_count: Optional[int] = None
    odg_row_count: Optional[int] = None
    odg_encoding: Optional[str] = None


@dataclass
class PlantConfig:
    plant_time_range: Optional[Tuple[float, float]] = None
    plant_signal_count: Optional[int] = None
    plant_map_files: List[str] = field(default_factory=list)
    plant_data_files: List[str] = field(default_factory=list)
    plant_id_signal_map: Dict[str, Tuple[str, "SignalType"]] = field(default_factory=dict)
    plant_map_count: Optional[int] = None
    plant_data_count: Optional[int] = None
    plant_data_rows: Optional[int] = None
    plant_skipped_mappings: Optional[int] = None
    plant_map_encoding: Optional[str] = None
    plant_data_encoding: Optional[str] = None


@dataclass
class AppConfig:
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    state: StateConfig = field(default_factory=StateConfig)
    odg: OdgConfig = field(default_factory=OdgConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)

    @staticmethod
    def _default_ini_path() -> str:
        base_dir = os.getenv("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base_dir, "TNPI", "app_config.ini")

    def dump_ini(self, path: Optional[str] = None) -> str:
        ini_path = path or self._default_ini_path()
        os.makedirs(os.path.dirname(ini_path), exist_ok=True)
        parser = configparser.ConfigParser()
        parser["state"] = {
            "range_mode": self.state.range_mode,
            "odg_folder": self.state.odg_folder,
            "plant_folder": self.state.plant_folder,
        }
        parser["odg"] = {
            "encoding": self.odg.odg_encoding or "",
        }
        parser["plant"] = {
            "map_encoding": self.plant.plant_map_encoding or "",
            "data_encoding": self.plant.plant_data_encoding or "",
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
            config.state.range_mode = parser.get("state", "range_mode", fallback=config.state.range_mode)
            config.state.odg_folder = parser.get(
                "state",
                "odg_folder",
                fallback=parser.get("state", "high_folder", fallback=config.state.odg_folder),
            )
            config.state.plant_folder = parser.get(
                "state",
                "plant_folder",
                fallback=parser.get("state", "low_folder", fallback=config.state.plant_folder),
            )
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
        if parser.has_section("odg"):
            enc = parser.get("odg", "encoding", fallback="").strip()
            config.odg.odg_encoding = enc or None
        if parser.has_section("plant"):
            map_enc = parser.get("plant", "map_encoding", fallback="").strip()
            data_enc = parser.get("plant", "data_encoding", fallback="").strip()
            config.plant.plant_map_encoding = map_enc or None
            config.plant.plant_data_encoding = data_enc or None
        return config


@dataclass
class OdgMeta:
    files: List[str]
    names: List[str]
    time_range: Optional[Tuple[float, float]]
    signal_count: Optional[int]
    row_count: Optional[int]
    encoding: Optional[str]


@dataclass
class PlantMeta:
    map_files: List[str]
    data_files: List[str]
    map_count: Optional[int]
    data_count: Optional[int]
    data_rows: Optional[int]
    skipped_mappings: Optional[int]
    map_encoding: Optional[str]
    data_encoding: Optional[str]
    time_range: Optional[Tuple[float, float]]
    signal_count: Optional[int]
    id_signal_map: Dict[str, Tuple[str, object]]


@dataclass
class ConfirmResult:
    ok: bool
    error: Optional[str] = None
    error_code: Optional[str] = None
    odg_meta: Optional[OdgMeta] = None
    plant_meta: Optional[PlantMeta] = None
    inspector: Optional[Dict[str, float]] = None


def get_config(session_state: Any) -> AppConfig:
    if "app_config" not in session_state or not isinstance(session_state["app_config"], AppConfig):
        try:
            session_state["app_config"] = AppConfig.load_ini()
        except Exception:
            session_state["app_config"] = AppConfig()
    return session_state["app_config"]
