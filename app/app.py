"""
Streamlit entrypoint for the FinCrime Intelligence Copilot.

This script exposes a multi‑page Streamlit application.  Each page lives
in the `app/pages` directory and is automatically discovered and
registered by Streamlit when the app runs.

Usage::

    streamlit run app/app.py

For development you can run with the local Python interpreter:

    python -m streamlit run app/app.py
"""

import streamlit as st

st.set_page_config(page_title="FinCrime Intelligence Copilot", layout="wide")

st.title("FinCrime Intelligence Copilot")

st.markdown(
    """
    Welcome to the prototype FinCrime intelligence copilot. Use the sidebar to
    navigate between modules:

    - **Risk Prediction:** Enter transaction/customer details and view the
      estimated risk score.
    - **Case Note Analyzer:** Paste case narratives to extract key
      information and risk signals.
    - **Similar Case Search:** Explore past alerts similar to your
      current case.
    - **RAG Chatbot:** Ask questions about policies and past cases using
      retrieval‑augmented generation.
    - **Model Performance:** Visualise metrics and compare model
      performance.
    """
)