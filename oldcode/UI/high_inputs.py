import pandas as pd
import streamlit as st

from state import AppConfig

def render_high_inputs(app_cfg: AppConfig) -> str:
    ui_state = app_cfg.state
    high_cfg = app_cfg.high
    st.subheader("ODG inputs")
    st.markdown("ODG folder path")
    folder_cols = st.columns([3, 1])
    with folder_cols[0]:
        high_folder = st.text_input(
            "ODG folder path",
            value=ui_state.high_folder,
            placeholder=r"C:\path\to\folder",
            label_visibility="collapsed",
        )
    with folder_cols[1]:
        st.text("")
    high_meta_placeholder = st.container()
    if ui_state.high_confirmed and high_cfg.high_time_range is not None:
        t_min, t_max = high_cfg.high_time_range
        dt_min = pd.to_datetime(int(t_min), unit="ns")
        dt_max = pd.to_datetime(int(t_max), unit="ns")
        high_meta_placeholder.caption(
            "ODG time range: "
            f"{dt_min.strftime('%Y-%m-%d-%H-%M')} ~ {dt_max.strftime('%Y-%m-%d-%H-%M')}"
        )
        if high_cfg.high_signal_count is not None:
            high_meta_placeholder.caption(
                f"ODG signal count: {high_cfg.high_signal_count}"
            )
        if high_cfg.high_row_count is not None:
            high_meta_placeholder.caption(
                f"ODG files: {len(high_cfg.high_names)} CSVs, "
                f"total rows: {high_cfg.high_row_count}"
            )
    return high_folder
