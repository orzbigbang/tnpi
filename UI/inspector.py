from typing import Mapping, Optional, Sequence, Tuple

import streamlit as st


def _normalize_profile(profile: Mapping[str, float]) -> Sequence[Tuple[str, float]]:
    rows = []
    for key in sorted(profile.keys()):
        val = profile.get(key)
        if val is None:
            continue
        try:
            rows.append((str(key), float(val)))
        except (TypeError, ValueError):
            continue
    return rows


def render_inspector_block(
    title: str,
    profile: Optional[Mapping[str, float]],
    *,
    unit: str = "ms",
    empty_msg: str = "No timing data.",
    key_prefix: str = "profile",
) -> None:
    if not profile:
        return
    rows = _normalize_profile(profile)
    if not rows:
        return
    with st.expander(title):
        for key, val in rows:
            st.write(f"{key}: {val:.2f} {unit}")


def render_confirm_samples_inspector(profile: Optional[Mapping[str, float]]) -> None:
    title = "Confirm sampling Timing"
    if profile:
        render_inspector_block(
            title,
            profile,
            unit="ms",
            key_prefix="confirm_samples",
        )
        return
    with st.expander(title):
        st.write("No timing data yet. Confirm sampling folders to populate.")


def render_compare_samples_inspector(profile: Optional[Mapping[str, float]]) -> None:
    title = "Compute Parameter Timing"
    if profile:
        render_inspector_block(
            title,
            profile,
            unit="ms",
            key_prefix="compare_samples",
        )
        return
    with st.expander(title):
        st.write("No timing data yet. Run comparison to populate.")
