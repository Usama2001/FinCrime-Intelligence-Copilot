"""
Page 3: Similar Case Search

This module demonstrates how investigators can search for cases similar to
the current alert using embeddings and a vector database.  The search
functionality is not yet implemented.
"""

import streamlit as st

st.title("Similar Case Search")

st.markdown("Enter keywords or paste a case description to find similar past cases in the vector database.")

query = st.text_input("Search query")

if st.button("Search"):
    if query.strip():
        st.warning("Search functionality not implemented yet.")
    else:
        st.info("Please enter a query.")