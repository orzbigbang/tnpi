import pandas as pd
import streamlit as st

from state import get_config
from .handlers import confirm_samples
from .odg_inputs import render_odg_inputs
from .plant_inputs import render_plant_inputs
from .run_section import render_run_section
from .sidebar import render_sidebar
from .result import render_run_results
from .inspector import render_compare_samples_inspector, render_confirm_samples_inspector


def run() -> None:
    app_cfg = get_config(st.session_state)
    st.set_page_config(page_title="ODG-vs-PlantDB Accuracy", layout="wide")
    st.title("ODG-vs-PlantDB Accuracy")
    odg_folder = render_odg_inputs(app_cfg)
    plant_folder = render_plant_inputs(app_cfg)

    def start_confirm() -> None:
        app_cfg.state.is_confirming = True

    st.button(
        "Confirm sampling folders",
        type="secondary",
        disabled=app_cfg.state.is_confirming,
        on_click=start_confirm,
    )

    if app_cfg.state.is_confirming:
        with st.spinner("Confirming sampling folders..."):
            error = confirm_samples(app_cfg, odg_folder, plant_folder)
        app_cfg.state.is_confirming = False
        if error:
            st.error(error)
        else:
            st.rerun()

    if app_cfg.state.plant_confirmed and app_cfg.state.odg_confirmed:
        # render_sidebar(app_cfg)
        render_confirm_samples_inspector(app_cfg.state.confirm_sample_inspector)
        render_run_section(app_cfg, render_run_results)
    else:
        st.info("Confirm both ODG and PlantDB inputs to enable run options.")
