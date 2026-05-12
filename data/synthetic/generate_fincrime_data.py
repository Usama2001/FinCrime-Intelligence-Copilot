import os
import random
import numpy as np
import pandas as pd


# ---------------------------------------------
# Synthetic FinCrime Dataset Generator
# ---------------------------------------------
# This script creates synthetic customer and transaction-level
# financial crime investigation data.
#
# Target column:
# suspicious_flag = 0 means normal / low-risk case
# suspicious_flag = 1 means suspicious / high-risk case
# ---------------------------------------------


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def create_case_note(row):
    """
    Create a realistic synthetic case note based on risk signals.
    This will later help us in NLP, embeddings, similar case search, and RAG.
    """

    notes = []

    if row["transaction_amount"] > 10000:
        notes.append("high-value transaction")

    if row["high_risk_country_flag"] == 1:
        notes.append("connection with a high-risk country")

    if row["cash_deposit_count_30d"] >= 5:
        notes.append("multiple cash deposits within 30 days")

    if row["international_transfer_count_30d"] >= 4:
        notes.append("frequent international transfers")

    if row["previous_alert_count"] >= 3:
        notes.append("multiple previous AML alerts")

    if row["rapid_movement_flag"] == 1:
        notes.append("rapid movement of funds after credit")

    if row["customer_risk_rating"] == "High":
        notes.append("customer already classified as high risk")

    if row["transaction_type"] == "Wire Transfer":
        notes.append("wire transfer activity")

    if row["transaction_type"] == "Cash Deposit":
        notes.append("cash deposit activity")

    if row["transaction_type"] == "Crypto Purchase":
        notes.append("crypto-related payment activity")

    if len(notes) == 0:
        return (
            "Customer activity appears consistent with normal account behavior. "
            "No major suspicious pattern was identified during the review."
        )

    note_text = ", ".join(notes)

    if row["suspicious_flag"] == 1:
        return (
            f"The alert was generated due to {note_text}. "
            f"The pattern may indicate potential money laundering, layering, or unusual fund movement. "
            f"Further investigation is recommended."
        )

    return (
        f"The transaction shows {note_text}, but the overall pattern is not strong enough "
        f"to confirm suspicious activity. Analyst review may still be required."
    )


def generate_fincrime_dataset(n_rows=3000):
    """
    Generate a synthetic FinCrime dataset with numeric, categorical,
    and text-based features.
    """

    customer_ids = [f"CUST-{100000 + i}" for i in range(n_rows)]
    case_ids = [f"CASE-{200000 + i}" for i in range(n_rows)]

    customer_risk_rating = np.random.choice(
        ["Low", "Medium", "High"],
        size=n_rows,
        p=[0.55, 0.30, 0.15]
    )

    transaction_channel = np.random.choice(
        ["Branch", "ATM", "Online", "Mobile", "Wire"],
        size=n_rows,
        p=[0.20, 0.15, 0.35, 0.20, 0.10]
    )

    transaction_type = np.random.choice(
        ["Cash Deposit", "Cash Withdrawal", "Wire Transfer", "Online Transfer", "Card Payment", "Crypto Purchase"],
        size=n_rows,
        p=[0.18, 0.17, 0.18, 0.25, 0.15, 0.07]
    )

    # Transaction amount: most values are normal, some are very high
    transaction_amount = np.round(
        np.random.lognormal(mean=8.1, sigma=1.0, size=n_rows),
        2
    )

    transaction_count_7d = np.random.poisson(lam=4, size=n_rows)
    avg_transaction_amount_30d = np.round(
        np.random.lognormal(mean=7.8, sigma=0.8, size=n_rows),
        2
    )

    high_risk_country_flag = np.random.choice(
        [0, 1],
        size=n_rows,
        p=[0.82, 0.18]
    )

    cash_deposit_count_30d = np.random.poisson(lam=2, size=n_rows)
    international_transfer_count_30d = np.random.poisson(lam=1.5, size=n_rows)
    previous_alert_count = np.random.poisson(lam=0.8, size=n_rows)
    account_age_days = np.random.randint(30, 3650, size=n_rows)

    rapid_movement_flag = np.random.choice(
        [0, 1],
        size=n_rows,
        p=[0.78, 0.22]
    )

    structuring_flag = np.random.choice(
        [0, 1],
        size=n_rows,
        p=[0.86, 0.14]
    )

    unusual_time_flag = np.random.choice(
        [0, 1],
        size=n_rows,
        p=[0.88, 0.12]
    )

    df = pd.DataFrame({
        "case_id": case_ids,
        "customer_id": customer_ids,
        "transaction_amount": transaction_amount,
        "transaction_count_7d": transaction_count_7d,
        "avg_transaction_amount_30d": avg_transaction_amount_30d,
        "high_risk_country_flag": high_risk_country_flag,
        "cash_deposit_count_30d": cash_deposit_count_30d,
        "international_transfer_count_30d": international_transfer_count_30d,
        "previous_alert_count": previous_alert_count,
        "account_age_days": account_age_days,
        "rapid_movement_flag": rapid_movement_flag,
        "structuring_flag": structuring_flag,
        "unusual_time_flag": unusual_time_flag,
        "customer_risk_rating": customer_risk_rating,
        "transaction_channel": transaction_channel,
        "transaction_type": transaction_type
    })

    # ---------------------------------------------
    # Create suspicious risk score
    # ---------------------------------------------
    # We create rule-based synthetic labels.
    # This makes the dataset realistic and explainable.
    # ---------------------------------------------

    risk_score = (
        (df["transaction_amount"] > 10000).astype(int) * 2.0
        + (df["transaction_amount"] > 25000).astype(int) * 2.0
        + df["high_risk_country_flag"] * 2.5
        + (df["cash_deposit_count_30d"] >= 5).astype(int) * 1.5
        + (df["international_transfer_count_30d"] >= 4).astype(int) * 1.8
        + (df["previous_alert_count"] >= 3).astype(int) * 2.0
        + df["rapid_movement_flag"] * 1.8
        + df["structuring_flag"] * 1.7
        + df["unusual_time_flag"] * 1.0
        + (df["customer_risk_rating"] == "High").astype(int) * 2.0
        + (df["customer_risk_rating"] == "Medium").astype(int) * 0.8
        + (df["transaction_type"] == "Wire Transfer").astype(int) * 1.0
        + (df["transaction_type"] == "Crypto Purchase").astype(int) * 1.4
    )

    # Add random noise so the dataset is not too perfect
    risk_score = risk_score + np.random.normal(0, 1.2, n_rows)

    df["risk_score"] = np.round(risk_score, 2)

    # Convert risk score into binary target
    df["suspicious_flag"] = (df["risk_score"] >= 5.5).astype(int)

    # Create text case note after target is created
    df["case_note"] = df.apply(create_case_note, axis=1)

    # Reorder columns
    df = df[
        [
            "case_id",
            "customer_id",
            "transaction_amount",
            "transaction_count_7d",
            "avg_transaction_amount_30d",
            "high_risk_country_flag",
            "cash_deposit_count_30d",
            "international_transfer_count_30d",
            "previous_alert_count",
            "account_age_days",
            "rapid_movement_flag",
            "structuring_flag",
            "unusual_time_flag",
            "customer_risk_rating",
            "transaction_channel",
            "transaction_type",
            "case_note",
            "risk_score",
            "suspicious_flag",
        ]
    ]

    return df


def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "fincrime_cases.csv")

    df = generate_fincrime_dataset(n_rows=3000)
    df.to_csv(output_path, index=False)

    print("Synthetic FinCrime dataset created successfully.")
    print(f"File saved at: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["suspicious_flag"].value_counts())
    print("\nTarget distribution percentage:")
    print(df["suspicious_flag"].value_counts(normalize=True).round(3) * 100)

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()