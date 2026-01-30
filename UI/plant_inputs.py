import pandas as pd
import streamlit as st

from state import AppConfig


def render_plant_inputs(app_cfg: AppConfig) -> str:
    st.subheader("PlantDB inputs")
    st.markdown("PlantDB folder path")
    st.caption("Folder should include mapping CSV(s) and sampling CSV(s).")

    folder_cols = st.columns([3, 1])
    with folder_cols[0]:
        plant_folder = st.text_input(
            "PlantDB folder path",
            value=app_cfg.state.plant_folder,  # load last time used folder
            placeholder=r"C:\path\to\folder",
            label_visibility="collapsed",
        )
    with folder_cols[1]:
        st.text("")

    plant_meta_placeholder = st.container()
    if app_cfg.state.plant_confirmed and app_cfg.plant.plant_time_range is not None:
        t_min, t_max = app_cfg.plant.plant_time_range
        dt_min = pd.to_datetime(int(t_min), unit="ns")
        dt_max = pd.to_datetime(int(t_max), unit="ns")
        plant_meta_placeholder.caption(
            "PlantDB time range: "
            f"{dt_min.strftime('%Y-%m-%d-%H-%M')} ~ {dt_max.strftime('%Y-%m-%d-%H-%M')}"
        )
        if app_cfg.plant.plant_signal_count is not None:
            plant_meta_placeholder.caption(
                f"PlantDB signal count: {app_cfg.plant.plant_signal_count}"
            )
        if app_cfg.plant.plant_map_count is not None and app_cfg.plant.plant_data_count is not None:
            plant_meta_placeholder.caption(
                f"PlantDB files: mapping CSVs = {app_cfg.plant.plant_map_count}, "
                f"sampling CSVs = {app_cfg.plant.plant_data_count}"
            )
        if app_cfg.plant.plant_data_rows is not None:
            plant_meta_placeholder.caption(
                f"PlantDB rows: {app_cfg.plant.plant_data_rows}"
            )
    return plant_folder
