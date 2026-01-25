import streamlit as st

from state import get_config
from .handlers import confirm_samples
from .high_inputs import render_high_inputs
from .low_inputs import render_low_inputs
from .run_section import render_run_section
from .sidebar import render_sidebar


def run() -> None:
    app_cfg = get_config(st.session_state)
    st.set_page_config(page_title="ODG-vs-PlantDB Accuracy", layout="wide")
    st.title("ODG-vs-PlantDB Accuracy")
    render_sidebar(app_cfg)
    if "confirm_pending" not in st.session_state:
        st.session_state.confirm_pending = False
    high_folder = render_high_inputs(app_cfg)
    low_folder = render_low_inputs(app_cfg)

    def start_confirm() -> None:
        st.session_state.confirm_pending = True

    st.button(
        "Confirm sampling folders",
        type="secondary",
        disabled=st.session_state.confirm_pending,
        on_click=start_confirm,
    )
    if st.session_state.confirm_pending:
        with st.spinner("Confirming sampling folders..."):
            error = confirm_samples(app_cfg, high_folder, low_folder)
        st.session_state.confirm_pending = False
        if error:
            st.error(error)
        else:
            st.rerun()
    render_run_section(app_cfg)
