from typing import Callable

import pandas as pd
import streamlit as st

from core.models import RunResult
from state import AppConfig
from .handlers import compute_run_result


def render_run_section(
    app_cfg: AppConfig,
    render_run_results: Callable[[RunResult], None],
) -> None:
    odg_names = app_cfg.odg.odg_names
    st.subheader("Time range")
    prev_range_mode = app_cfg.state.range_mode
    range_mode = st.radio(
        "Time range",
        options=["full", "range"],
        format_func=lambda x: {"full": "Full range", "range": "Select range"}[x],
        index=["full", "range"].index(app_cfg.state.range_mode),
        label_visibility="collapsed",
    )
    app_cfg.state.range_mode = range_mode
    if range_mode != prev_range_mode:
        app_cfg.dump_ini()
    range_start = ""
    range_end = ""
    if range_mode == "range":
        if app_cfg.odg.odg_time_range is not None:
            t_min, t_max = app_cfg.odg.odg_time_range
            dt_min = pd.to_datetime(int(t_min), unit="ns")
            dt_max = pd.to_datetime(int(t_max), unit="ns")
            if not app_cfg.state.range_start:
                app_cfg.state.range_start = dt_min.strftime("%Y-%m-%d-%H-%M")
            if not app_cfg.state.range_end:
                app_cfg.state.range_end = dt_max.strftime("%Y-%m-%d-%H-%M")
        r1, r2 = st.columns(2)
        with r1:
            range_start = st.text_input(
                "Range start (YYYY-mm-dd-hh-mm)",
                value=app_cfg.state.range_start,
                placeholder="e.g. 2024-01-01-00-00",
            )
        with r2:
            range_end = st.text_input(
                "Range end (YYYY-mm-dd-hh-mm)",
                value=app_cfg.state.range_end,
                placeholder="e.g. 2024-01-02-00-00",
            )
        app_cfg.state.range_start = range_start
        app_cfg.state.range_end = range_end
        st.caption("Format uses 24-hour clock; comparison uses parsed timestamps.")

    def start_run() -> None:
        app_cfg.state.is_running = True

    st.button(
        "Run",
        type="primary",
        disabled=app_cfg.state.is_running,
        on_click=start_run,
    )
    if app_cfg.state.is_running:
        progress_acc = st.progress(0, text="progerss: starting...")
        if odg_names:
            progress_acc.progress(0, text=f"progerss: 0/{len(odg_names)}")

        def on_progress(done: int, total: int) -> None:
            pct = done / total if total else 1.0
            progress_acc.progress(pct, text=f"progerss: {done}/{total}")

        result = compute_run_result(
            app_cfg,
            range_mode=range_mode,
            range_start=range_start,
            range_end=range_end,
            progress_cb=on_progress,
        )
        if not result.ok:
            err = result.error or "Run failed."
            if result.error_code:
                err = f"{err} (code: {result.error_code})"
            st.error(err)
            app_cfg.state.is_running = False
            st.stop()
        app_cfg.state.last_result = result
        render_run_results(result)
        if odg_names:
            progress_acc.progress(1.0, text="progerss: done")
        app_cfg.state.is_running = False
    elif app_cfg.state.last_result is not None:
        render_run_results(app_cfg.state.last_result)
