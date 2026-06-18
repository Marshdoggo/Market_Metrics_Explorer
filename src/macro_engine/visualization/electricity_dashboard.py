from __future__ import annotations

import streamlit as st


def render_electricity_dashboard() -> None:
    """Render a safe placeholder when macro dashboard assets are unavailable."""
    st.info(
        "Macro dashboard assets are not available in this deployment yet. "
        "The market metric explorer is still available from the sidebar."
    )
