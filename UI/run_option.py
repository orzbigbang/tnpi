import os
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from state import AppConfig


def render_run_option(app_cfg: AppConfig) -> Tuple[str, str, str]:
    st.subheader("Run options")
    if app_cfg.state.range_mode != "range":
        app_cfg.state.range_mode = "range"
        app_cfg.dump_ini()
    selected_key = "selected_quick_range"
    mode_key = "selected_quick_range_mode"
    start_key = "range_start_input"
    end_key = "range_end_input"

    overlap_ranges = _build_overlap_ranges(
        app_cfg.plant.plant_data_spans,
        app_cfg.odg.odg_time_range,
    )
    selected_range: Optional[Tuple[str, float, float]] = None
    if overlap_ranges:
        st.markdown("Select Range")
        st.caption("Quick ranges (Plant data files overlapped with ODG span)")
        raw_selected = st.session_state.get(selected_key)
        for i, (path, start_ns, end_ns) in enumerate(overlap_ranges):
            name = os.path.basename(path)
            label = f"{name}: {_fmt_ns(start_ns)} ~ {_fmt_ns(end_ns)}"
            is_selected = (
                isinstance(raw_selected, tuple)
                and len(raw_selected) == 3
                and raw_selected[0] == path
                and float(raw_selected[1]) == float(start_ns)
                and float(raw_selected[2]) == float(end_ns)
            )
            if st.button(
                label,
                key=f"quick_range_{i}",
                type="primary" if is_selected else "secondary",
            ):
                st.session_state[selected_key] = (path, start_ns, end_ns)
                st.session_state[mode_key] = "full"
                st.session_state[start_key] = _fmt_ns(start_ns)
                st.session_state[end_key] = _fmt_ns(end_ns)
                st.rerun()
        if isinstance(raw_selected, tuple) and len(raw_selected) == 3:
            for path, start_ns, end_ns in overlap_ranges:
                if raw_selected[0] == path and float(raw_selected[1]) == float(start_ns) and float(raw_selected[2]) == float(end_ns):
                    selected_range = (path, float(start_ns), float(end_ns))
                    break
        if selected_range is not None:
            st.caption(
                "Selected: "
                f"{os.path.basename(selected_range[0])} "
                f"({_fmt_ns(selected_range[1])} ~ {_fmt_ns(selected_range[2])})"
            )

    if selected_range is None:
        st.caption("Choose one quick range before running.")
        return "range", "", ""

    selected_start_ns = selected_range[1]
    selected_end_ns = selected_range[2]

    option_mode = st.radio(
        "Range mode",
        options=["full", "input"],
        format_func=lambda x: {"full": "Full range", "input": "Input range"}[x],
        key=mode_key,
        horizontal=True,
    )
    if option_mode == "full":
        range_start = _fmt_ns(selected_start_ns)
        range_end = _fmt_ns(selected_end_ns)
        app_cfg.state.range_start = range_start
        app_cfg.state.range_end = range_end
        st.session_state[start_key] = range_start
        st.session_state[end_key] = range_end
        st.caption("Using full selected quick-range overlap.")
        return "range", range_start, range_end

    raw_start = str(st.session_state.get(start_key, app_cfg.state.range_start or _fmt_ns(selected_start_ns)))
    raw_end = str(st.session_state.get(end_key, app_cfg.state.range_end or _fmt_ns(selected_end_ns)))
    fixed_start, fixed_end, corrected = _clamp_input_to_bounds(
        raw_start,
        raw_end,
        (selected_start_ns, selected_end_ns),
    )

    app_cfg.state.range_start = fixed_start
    app_cfg.state.range_end = fixed_end
    st.session_state[start_key] = fixed_start
    st.session_state[end_key] = fixed_end

    r1, r2 = st.columns(2)
    with r1:
        range_start = st.text_input(
            "Range start (YYYY-mm-dd-hh-mm)",
            value=fixed_start,
            placeholder="e.g. 2026-01-01-01-23",
            key=start_key,
        )
    with r2:
        range_end = st.text_input(
            "Range end (YYYY-mm-dd-hh-mm)",
            value=fixed_end,
            placeholder="e.g. 2026-01-01-02-23",
            key=end_key,
        )
    app_cfg.state.range_start = range_start
    app_cfg.state.range_end = range_end
    st.caption("Format uses 24-hour clock; comparison uses parsed timestamps.")
    text_bounds = _allowed_range_text_bounds((selected_start_ns, selected_end_ns))
    if corrected and text_bounds is not None:
        st.warning(
            "Invalid or out-of-range input was corrected to overlap bounds: "
            f"{text_bounds[0]} ~ {text_bounds[1]}."
        )

    return "range", range_start, range_end


def _fmt_ns(ts_ns: float) -> str:
    return pd.to_datetime(int(ts_ns), unit="ns").strftime("%Y-%m-%d-%H-%M")


def _parse_minute_text(text: str) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime((text or "").strip(), format="%Y-%m-%d-%H-%M", errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _allowed_range_text_bounds(
    bounds_ns: Optional[Tuple[float, float]],
) -> Optional[Tuple[str, str]]:
    if bounds_ns is None:
        return None
    start_ns, end_ns = bounds_ns
    start_dt = pd.to_datetime(int(start_ns), unit="ns").floor("min")
    end_dt = pd.to_datetime(int(end_ns), unit="ns").floor("min")
    if start_dt > end_dt:
        start_dt = end_dt
    return (
        start_dt.strftime("%Y-%m-%d-%H-%M"),
        end_dt.strftime("%Y-%m-%d-%H-%M"),
    )


def _clamp_input_to_bounds(
    start_txt: str,
    end_txt: str,
    bounds_ns: Optional[Tuple[float, float]],
) -> Tuple[str, str, bool]:
    corrected = False
    if bounds_ns is None:
        return start_txt, end_txt, corrected

    text_bounds = _allowed_range_text_bounds(bounds_ns)
    if text_bounds is None:
        return start_txt, end_txt, corrected
    bound_start_txt, bound_end_txt = text_bounds
    bound_start_dt = _parse_minute_text(bound_start_txt)
    bound_end_dt = _parse_minute_text(bound_end_txt)
    if bound_start_dt is None or bound_end_dt is None:
        return start_txt, end_txt, corrected

    start_dt = _parse_minute_text(start_txt)
    end_dt = _parse_minute_text(end_txt)

    if start_dt is None:
        start_txt = bound_start_txt
        start_dt = _parse_minute_text(start_txt)
        corrected = True
    elif start_dt < bound_start_dt:
        start_txt = bound_start_txt
        start_dt = _parse_minute_text(start_txt)
        corrected = True
    elif start_dt > bound_end_dt:
        start_txt = bound_end_txt
        start_dt = _parse_minute_text(start_txt)
        corrected = True

    if end_dt is None:
        end_txt = bound_end_txt
        end_dt = _parse_minute_text(end_txt)
        corrected = True
    elif end_dt > bound_end_dt:
        end_txt = bound_end_txt
        end_dt = _parse_minute_text(end_txt)
        corrected = True
    elif end_dt < bound_start_dt:
        end_txt = bound_start_txt
        end_dt = _parse_minute_text(end_txt)
        corrected = True

    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        end_txt = start_txt
        corrected = True
    return start_txt, end_txt, corrected


def _build_overlap_ranges(
    plant_data_spans: List[Tuple[str, float, float]],
    odg_time_range: Optional[Tuple[float, float]],
) -> List[Tuple[str, float, float]]:
    if odg_time_range is None:
        return []
    odg_start, odg_end = odg_time_range
    rows: List[Tuple[str, float, float]] = []
    for path, start, end in plant_data_spans:
        overlap_start = max(float(start), float(odg_start))
        overlap_end = min(float(end), float(odg_end))
        if overlap_start <= overlap_end:
            rows.append((str(path), overlap_start, overlap_end))
    rows.sort(key=lambda x: x[1])
    return rows
