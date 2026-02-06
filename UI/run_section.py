from typing import Callable

import streamlit as st

from core.models import RunResult
from state import AppConfig
from .handlers import compute_run_result
from .run_option import render_run_option


def render_run_section(
    app_cfg: AppConfig,
    render_run_results: Callable[[RunResult], None],
) -> None:
    odg_names = app_cfg.odg.odg_names
    range_mode, range_start, range_end = render_run_option(app_cfg)

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
