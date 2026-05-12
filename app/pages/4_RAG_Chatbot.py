"""
Page 4: RAG Chatbot

This module exposes a retrieval‑augmented generation (RAG) interface
powered by large language models.  Users can ask questions about policy
documents and case data, and the chatbot retrieves relevant
information from a knowledge base before generating an answer.

To use this page you will need a configured LLM provider and vector
database.  Placeholder UI is provided here.
"""

import streamlit as st

st.title("RAG Chatbot")

st.markdown("Ask a question about AML policies or past cases.  The chatbot will retrieve relevant documents and generate an answer.")

question = st.text_input("Your question")

if st.button("Ask"):
    if question.strip():
        st.warning("Chatbot functionality not implemented yet.")
    else:
        st.info("Please enter a question.")