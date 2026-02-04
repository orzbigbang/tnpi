import pandas as pd
import streamlit as st

from state import AppConfig, RunResult
from .handlers import compute_run_result


def render_run_section(app_cfg: AppConfig) -> None:
    ui_state = app_cfg.state
    high_cfg = app_cfg.high
    if ui_state.low_confirmed and ui_state.high_confirmed:
        try:
            high_names = high_cfg.high_names
            st.caption("Run will compute accuracy for all loaded ODG CSVs.")
            st.subheader("Time range")
            prev_range_mode = ui_state.range_mode
            range_mode = st.radio(
                "Time range",
                options=["full", "range"],
                format_func=lambda x: {"full": "Full range", "range": "Select range"}[x],
                index=["full", "range"].index(ui_state.range_mode),
                label_visibility="collapsed",
            )
            ui_state.range_mode = range_mode
            if range_mode != prev_range_mode:
                app_cfg.dump_ini()
            range_start = ""
            range_end = ""
            if range_mode == "range":
                if high_cfg.high_time_range is not None:
                    t_min, t_max = high_cfg.high_time_range
                    dt_min = pd.to_datetime(int(t_min), unit="ns")
                    dt_max = pd.to_datetime(int(t_max), unit="ns")
                    if not ui_state.range_start:
                        ui_state.range_start = dt_min.strftime("%Y-%m-%d-%H-%M")
                    if not ui_state.range_end:
                        ui_state.range_end = dt_max.strftime("%Y-%m-%d-%H-%M")
                r1, r2 = st.columns(2)
                with r1:
                    range_start = st.text_input(
                        "Range start (YYYY-mm-dd-hh-mm)",
                        value=ui_state.range_start,
                        placeholder="e.g. 2024-01-01-00-00",
                    )
                with r2:
                    range_end = st.text_input(
                        "Range end (YYYY-mm-dd-hh-mm)",
                        value=ui_state.range_end,
                        placeholder="e.g. 2024-01-02-00-00",
                    )
                ui_state.range_start = range_start
                ui_state.range_end = range_end
                st.caption("Format uses 24-hour clock; comparison uses parsed timestamps.")
            st.subheader("Run mode")
            prev_run_mode = ui_state.run_mode
            mode = st.radio(
                "Run mode",
                options=["single", "multiprocess"],
                format_func=lambda x: {"single": "Single process", "multiprocess": "Multiprocess"}[x],
                index=["single", "multiprocess"].index(ui_state.run_mode),
                horizontal=True,
                label_visibility="collapsed",
            )
            ui_state.run_mode = mode
            if mode != prev_run_mode:
                app_cfg.dump_ini()

            def start_run() -> None:
                ui_state.is_running = True

            st.button(
                "Run",
                type="primary",
                disabled=ui_state.is_running,
                on_click=start_run,
            )
            if ui_state.is_running:
                progress_acc = st.progress(0, text="progerss: starting...")
                if high_names:
                    progress_acc.progress(0, text=f"progerss: 0/{len(high_names)}")

                def on_progress(done: int, total: int) -> None:
                    pct = done / total if total else 1.0
                    progress_acc.progress(pct, text=f"progerss: {done}/{total}")

                result = compute_run_result(
                    app_cfg,
                    range_mode=range_mode,
                    range_start=range_start,
                    range_end=range_end,
                    run_mode=mode,
                    progress_cb=on_progress,
                )
                if not result.ok:
                    st.error(result.error or "Run failed.")
                    ui_state.is_running = False
                    st.stop()
                ui_state.last_result = result
                render_run_results(result)
                if high_names:
                    progress_acc.progress(1.0, text="progerss: done")
                ui_state.is_running = False
            elif ui_state.last_result is not None:
                render_run_results(ui_state.last_result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Confirm both ODG and PlantDB inputs to enable run options.")


def render_run_results(result: RunResult) -> None:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        score = result.matching_score if result.matching_score is not None else result.summary
        st.metric(
            "Matching score",
            value=f"{score:.4f}" if pd.notna(score) else "NaN",
        )
    with c2:
        st.metric(
            "Correlation",
            value=f"{result.correlation:.4f}" if pd.notna(result.correlation) else "NaN",
        )
    with c3:
        st.metric(
            "RMSE",
            value=f"{result.rmse:.4f}" if pd.notna(result.rmse) else "NaN",
        )
    with c4:
        st.metric(
            "Offset (ms)",
            value=f"{result.offset_ms:.1f}" if pd.notna(result.offset_ms) else "NaN",
        )
    
    st.subheader("Signal name matching")
    st.caption(
        "PlantDB signals are mapped by ID (CSV1 + CSV2), then compared by signal name. Counts show overlap and missing signals on each side."
    )
    st.dataframe(
        pd.DataFrame([result.match_stats or {}]),
        use_container_width=True,
        height=160,
    )
    try:
        file_name, data = result.dump_result()
        st.download_button(
            "Download signal metrics CSV",
            data=data,
            file_name=file_name,
            mime="text/csv",
        )
    except Exception as ex:
        st.error(f"Prepare download failed: {ex}")
