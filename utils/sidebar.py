"""
Shared sidebar — call render_sidebar() from every page to get consistent
CSS and credit footer across the entire dashboard.
"""

import streamlit as st
from pathlib import Path


from utils.signal_processing import DEFAULT_MAPPING, apply_standard_montage

def render_sidebar():
    """Load custom CSS and global settings. Call this at the top of every page."""

    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    logo_path = "assets/eeg.png"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    
    st.logo(logo_path, size="large")

    # ── Global Montage Configuration ──────────────────────────────────────────
    active_rec = st.session_state.get("active_rec")
    active_rec_name = st.session_state.get("active_rec_name")
    
    if active_rec and active_rec_name:
        st.sidebar.divider()
        st.sidebar.subheader("Global Montage Settings")
        use_montage = st.sidebar.checkbox("Apply 10-20 Montage", value=True, help="Rename channels to standard 10-20 system for all analysis pages.", key="global_use_montage_cb")
        
        mapping = None
        if use_montage:
            with st.sidebar.expander("Edit 10-20 Mapping", expanded=False):
                st.info("Default mapping assumes OpenBCI Cyton+Daisy.")
                mapping = {}
                for ch in active_rec.ch_names:
                    default_name = DEFAULT_MAPPING.get(ch, ch)
                    mapping[ch] = st.text_input(f"{ch} →", value=default_name, key=f"global_map_sb_{ch}")
        
        # Track if montage settings changed
        montage_changed = (
            st.session_state.get("last_use_montage") != use_montage or
            st.session_state.get("last_mapping") != mapping or
            st.session_state.get("last_active_rec_name") != active_rec_name or
            "raw_base" not in st.session_state
        )

        if montage_changed:
            try:
                raw_base = active_rec.build_mne_raw()
                if use_montage:
                    st.session_state["raw_base"] = apply_standard_montage(raw_base, mapping)
                else:
                    st.session_state["raw_base"] = raw_base
                
                st.session_state["last_active_rec_name"] = active_rec_name
                st.session_state["last_use_montage"] = use_montage
                st.session_state["last_mapping"] = mapping
                
                # Clear downstream filtered data if montage changes
                if "raw_filtered" in st.session_state:
                    del st.session_state["raw_filtered"]
                if "raw_asr" in st.session_state:
                    del st.session_state["raw_asr"]
                if "ica_obj" in st.session_state:
                    del st.session_state["ica_obj"]
            except Exception as e:
                st.sidebar.error(f"Montage Error: {e}")


def render_sidebar_footer():
    """Render the credit footer at the bottom of the sidebar."""
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.3rem 0;">
            <div style="font-size: 0.72rem; color: #484f58; margin-bottom: 6px;">
                Developed by
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #c9d1d9;">
                Amir Rafe
            </div>
            <a href="mailto:amir.rafe@txstate.edu"
               style="font-size: 0.72rem; color: #58a6ff; text-decoration: none;">
                amir.rafe@txstate.edu
            </a>
            <div style="margin-top: 6px;">
                <span style="font-size: 0.65rem; color: #30363d;">━━━</span>
            </div>
            <a href="https://pozapas.github.io/" target="_blank"
               style="font-size: 0.7rem; color: #58a6ff; text-decoration: none; margin-top: 4px; display:inline-block;">
                🌐 pozapas.github.io
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
