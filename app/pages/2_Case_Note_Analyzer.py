"""
Page 2: Case Note Analyzer

This module accepts free‑text case notes and applies natural
language processing to extract keywords and summarise the narrative.
Currently the analysis logic is not implemented; the UI acts as a
placeholder for future development.
"""

import streamlit as st

st.title("Case Note Analyzer")

st.markdown("Paste the narrative or alert description below to analyse it.")

note = st.text_area("Case Note", height=200)

if st.button("Analyse"):
    if note.strip():
        st.warning("Case note analysis not implemented yet.")
    else:
        st.info("Please enter a case note to analyse.")