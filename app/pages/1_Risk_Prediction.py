"""
Page 1: Risk Prediction

This module collects transaction and customer details from the user
and uses a trained machine‑learning model to estimate the risk of
suspicious activity.  At this early stage of the project, the form
fields and model invocation are left as placeholders.
"""

import streamlit as st

st.title("Risk Prediction")

st.markdown("Fill in the transaction and customer details below and click **Predict** to estimate the risk of suspicious activity.")

with st.form(key="prediction_form"):
    st.text_input("Transaction amount (USD)", key="amount")
    st.text_input("Transaction type (e.g. wire, cash)", key="tx_type")
    st.text_input("Country risk score", key="country_risk")
    st.text_input("Customer age", key="age")
    st.text_input("Number of previous alerts", key="alerts")
    submit = st.form_submit_button("Predict")

if submit:
    st.warning("Prediction functionality not implemented yet.")