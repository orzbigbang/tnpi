import pandas as pd
import streamlit as st

from state import AppConfig

def render_low_inputs(app_cfg: AppConfig) -> str:
    ui_state = app_cfg.state
    low_cfg = app_cfg.low
    st.subheader("PlantDB inputs")
    st.markdown("PlantDB folder path")
    folder_cols = st.columns([3, 1])
    with folder_cols[0]:
        low_folder = st.text_input(
            "PlantDB folder path",
            value=ui_state.low_folder,
            placeholder=r"C:\path\to\folder",
            label_visibility="collapsed",
        )
    with folder_cols[1]:
        st.text("")
    st.caption("Folder should include mapping CSV(s) and sampling CSV(s).")
    low_meta_placeholder = st.container()
    if ui_state.low_confirmed and low_cfg.low_time_range is not None:
        t_min, t_max = low_cfg.low_time_range
        dt_min = pd.to_datetime(int(t_min), unit="ns")
        dt_max = pd.to_datetime(int(t_max), unit="ns")
        low_meta_placeholder.caption(
            "PlantDB time range: "
            f"{dt_min.strftime('%Y-%m-%d-%H-%M')} ~ {dt_max.strftime('%Y-%m-%d-%H-%M')}"
        )
        if low_cfg.low_signal_count is not None:
            low_meta_placeholder.caption(
                f"PlantDB signal count: {low_cfg.low_signal_count}"
            )
        if low_cfg.low_map_count is not None and low_cfg.low_data_count is not None:
            low_meta_placeholder.caption(
                f"PlantDB files: mapping CSVs = {low_cfg.low_map_count}, "
                f"sampling CSVs = {low_cfg.low_data_count}"
            )
        if low_cfg.low_data_rows is not None:
            low_meta_placeholder.caption(
                f"PlantDB rows: {low_cfg.low_data_rows}"
            )
    return low_folder
