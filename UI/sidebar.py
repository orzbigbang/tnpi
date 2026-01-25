import streamlit as st

from state import AppConfig, CompareConfig


def render_sidebar(app_cfg: AppConfig) -> CompareConfig:
    with st.sidebar:
        st.header("Scoring settings")
        st.caption("Hover the info icons for formulas and details.")
        current = app_cfg.compare
        prev = (
            current.accuracy_denominator,
            current.out_of_range_policy,
            current.aggregate_policy,
        )
        accuracy_denominator = st.radio(
            "Accuracy normalization",
            help="range: d = max(high) - min(high); std: d = std(high); abs_truth: r = |y_low - y_hat| / (|truth| + eps), point_acc = 1 - clip(r, 0, 1).",
            options=["range", "std", "abs_truth"],
            format_func=lambda x: {
                "range": "range = max(high) - min(high)",
                "std": "std = std(high)",
                "abs_truth": "abs_truth = |truth(t)| (relative error)",
            }[x],
            index=["range", "std", "abs_truth"].index(current.accuracy_denominator),
        )
        out_of_range_policy = st.radio(
            "If low timestamps are outside high range",
            help="drop: x_hat = NaN for out-of-range points; clip: t_low2 = clip(t_low, t_high[0], t_high[-1]) then interpolate.",
            options=["drop", "clip"],
            format_func=lambda x: {
                "drop": "drop those points (not scored)",
                "clip": "clip to boundary values",
            }[x],
            index=["drop", "clip"].index(current.out_of_range_policy),
        )
        aggregate_policy = st.radio(
            "Aggregate per-signal accuracy",
            help="mean: mean(point_acc) over valid points; min: min(point_acc) for a conservative score.",
            options=["mean", "min"],
            format_func=lambda x: {
                "mean": "mean",
                "min": "min (conservative)",
            }[x],
            index=["mean", "min"].index(current.aggregate_policy),
        )
    app_cfg.compare = CompareConfig(
        time_col=current.time_col,
        accuracy_denominator=accuracy_denominator,
        out_of_range_policy=out_of_range_policy,
        aggregate_policy=aggregate_policy,
    )
    if (
        accuracy_denominator,
        out_of_range_policy,
        aggregate_policy,
    ) != prev:
        app_cfg.dump_ini()
    return app_cfg.compare
