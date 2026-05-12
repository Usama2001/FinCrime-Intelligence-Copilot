# FinCrime Intelligence Copilot

## Overview

This project builds a proof‑of‑concept intelligence copilot for financial
crime (FinCrime) investigations.  The aim is to deliver an end‑to‑end
machine‑learning and natural language processing (NLP) solution that
assists analysts in identifying suspicious activity and better
understanding why alerts are triggered.  By combining structured risk
scoring, narrative analysis, semantic search and retrieval–augmented
generation (RAG), the copilot showcases modern ML/NLP/LLM techniques in a
single production‑ready application.

Key features include:

* **ML‑based risk scoring.**  Traditional classifiers (logistic
  regression, random forest, gradient boosting) are trained on synthetic
  transaction and customer data to estimate the likelihood that a case
  represents suspicious activity.
* **NLP case note analysis.**  Narrative fields are cleaned,
  tokenised and embedded.  The copilot summarises case descriptions and
  tags them with potential risk categories.
* **Similar case search.**  Embeddings are stored in a vector
  database (e.g. Qdrant) allowing investigators to retrieve and review
  past cases that resemble the current alert.
* **Retrieval‑augmented Q&A.**  A large language model (via
  LangChain or LlamaIndex) answers investigator questions by retrieving
  relevant policy documents and case notes, citing sources where
  appropriate.
* **Streamlit dashboard.**  A simple multi‑page interface allows
  users to enter customer/transaction details, analyse case notes,
  search similar cases and interact with the RAG assistant.  A
  performance page visualises model metrics.

This repository provides all the code needed to run the copilot
locally or deploy it to a cloud platform (Docker and MLflow support
included).  It is purely educational and does not process real
customer data.

## Repository Structure

```
fincrime-intelligence-copilot/
│
├── app/
│   ├── app.py                # Streamlit entrypoint
│   └── pages/                # Individual Streamlit pages
│       ├── 1_Risk_Prediction.py
│       ├── 2_Case_Note_Analyzer.py
│       ├── 3_Similar_Case_Search.py
│       ├── 4_RAG_Chatbot.py
│       └── 5_Model_Performance.py
│
├── data/
│   ├── raw/                  # Raw synthetic datasets
│   ├── processed/            # Processed datasets ready for modelling
│   └── synthetic/            # Scripts to generate synthetic data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_model_training.ipynb
│   ├── 04_nlp_case_analysis.ipynb
│   └── 05_rag_pipeline.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── nlp_pipeline.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── utils.py
│
├── models/                   # Serialized models
├── vector_db/               # Local vector database files
├── reports/
│   ├── figures/             # Figures and charts for reports
│   └── final_report.md      # Project report
├── tests/                   # Unit tests
│
├── requirements.txt         # Project dependencies
├── Dockerfile               # Dockerfile for containerisation
├── .gitignore               # Files/directories to ignore
├── LICENSE                  # Optional licence file
└── README.md                # Project overview (this file)
```

## Getting Started

To run the project locally:

1. **Clone the repository**

   ```powershell
   git clone https://github.com/<your‑github‑username>/FinCrime-Intelligence-Copilot.git
   cd FinCrime-Intelligence-Copilot
   ```

2. **Create a virtual environment and install dependencies**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```powershell
   python -m streamlit run app/app.py
   ```

This will launch the multi‑page dashboard in your browser.

## Disclaimer

This project uses synthetic or publicly available data to mimic the
behaviour of real financial crime alerts.  It is for educational
purposes only and should not be used in production or considered a
complete solution for compliance obligations.