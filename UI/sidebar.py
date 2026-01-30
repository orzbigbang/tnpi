import streamlit as st

from state import AppConfig, CompareConfig


def render_sidebar(app_cfg: AppConfig) -> CompareConfig:
    with st.sidebar:
        st.header("Scoring settings")
        st.caption("Hover the info icons for formulas and details.")
        prev = (
            app_cfg.compare.accuracy_denominator,
            app_cfg.compare.out_of_range_policy,
            app_cfg.compare.aggregate_policy,
        )
        accuracy_denominator = st.radio(
            "Accuracy normalization",
            help="range: d = max(odg) - min(odg); std: d = std(odg); abs_truth: r = |y_plant - y_hat| / (|truth| + eps), point_acc = 1 - clip(r, 0, 1).",
            options=["range", "std", "abs_truth"],
            format_func=lambda x: {
                "range": "range = max(odg) - min(odg)",
                "std": "std = std(odg)",
                "abs_truth": "abs_truth = |truth(t)| (relative error)",
            }[x],
            index=["range", "std", "abs_truth"].index(app_cfg.compare.accuracy_denominator),
        )
        out_of_range_policy = st.radio(
            "If plant timestamps are outside odg range",
            help="drop: x_hat = NaN for out-of-range points; clip: t_plant2 = clip(t_plant, t_odg[0], t_odg[-1]) then interpolate.",
            options=["drop", "clip"],
            format_func=lambda x: {
                "drop": "drop those points (not scored)",
                "clip": "clip to boundary values",
            }[x],
            index=["drop", "clip"].index(app_cfg.compare.out_of_range_policy),
        )
        aggregate_policy = st.radio(
            "Aggregate per-signal accuracy",
            help="mean: mean(point_acc) over valid points; min: min(point_acc) for a conservative score.",
            options=["mean", "min"],
            format_func=lambda x: {
                "mean": "mean",
                "min": "min (conservative)",
            }[x],
            index=["mean", "min"].index(app_cfg.compare.aggregate_policy),
        )
    app_cfg.compare = CompareConfig(
        time_col=app_cfg.compare.time_col,
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
