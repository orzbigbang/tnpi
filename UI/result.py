import pandas as pd
import streamlit as st

from core.models import RunResult
from .inspector import render_compare_samples_inspector


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

    render_compare_samples_inspector(result.compute_inspector)
