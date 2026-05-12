import os
import re
import pandas as pd


RAW_DATA_PATH = "data/raw/fincrime_cases.csv"
PROCESSED_DATA_PATH = "data/processed/fincrime_cases_cleaned.csv"
REPORT_PATH = "reports/data_quality_summary.txt"


def clean_text(text):
    """
    Clean text for NLP usage.

    This function:
    - Converts text to lowercase
    - Removes extra spaces
    - Keeps only letters, numbers, and basic spaces
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset(path=RAW_DATA_PATH):
    """
    Load the raw synthetic FinCrime dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please run data/synthetic/generate_fincrime_data.py first."
        )

    df = pd.read_csv(path)
    return df


def inspect_dataset(df):
    """
    Print important dataset information.
    """
    print("\n========== DATASET OVERVIEW ==========")
    print(f"Dataset shape: {df.shape}")

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["suspicious_flag"].value_counts())
    print("\nTarget distribution percentage:")
    print((df["suspicious_flag"].value_counts(normalize=True) * 100).round(2))

    print("\n========== SAMPLE DATA ==========")
    print(df.head())


def validate_dataset(df):
    """
    Validate important columns and values.
    """
    required_columns = [
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

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    binary_columns = [
        "high_risk_country_flag",
        "rapid_movement_flag",
        "structuring_flag",
        "unusual_time_flag",
        "suspicious_flag",
    ]

    for col in binary_columns:
        unique_values = set(df[col].dropna().unique())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Column {col} contains invalid binary values: {unique_values}")

    if (df["transaction_amount"] < 0).any():
        raise ValueError("transaction_amount contains negative values.")

    if (df["account_age_days"] <= 0).any():
        raise ValueError("account_age_days contains zero or negative values.")

    print("\nDataset validation passed successfully.")


def clean_dataset(df):
    """
    Clean and prepare the dataset for EDA, ML, and NLP.

    We do not encode categorical columns here.
    Encoding will be done later in the ML training step.
    """
    df_cleaned = df.copy()

    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove duplicate case IDs if any
    df_cleaned = df_cleaned.drop_duplicates(subset=["case_id"])

    # Strip extra spaces from categorical columns
    categorical_columns = [
        "customer_risk_rating",
        "transaction_channel",
        "transaction_type",
    ]

    for col in categorical_columns:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

    # Clean case note for NLP usage
    df_cleaned["case_note_cleaned"] = df_cleaned["case_note"].apply(clean_text)

    # Make sure binary columns are integers
    binary_columns = [
        "high_risk_country_flag",
        "rapid_movement_flag",
        "structuring_flag",
        "unusual_time_flag",
        "suspicious_flag",
    ]

    for col in binary_columns:
        df_cleaned[col] = df_cleaned[col].astype(int)

    return df_cleaned


def save_data_quality_report(df_raw, df_cleaned, path=REPORT_PATH):
    """
    Save a simple text report for interview and documentation purposes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("FinCrime Intelligence Copilot - Data Quality Summary\n")
        file.write("=" * 60 + "\n\n")

        file.write("Raw Dataset Shape:\n")
        file.write(f"{df_raw.shape}\n\n")

        file.write("Cleaned Dataset Shape:\n")
        file.write(f"{df_cleaned.shape}\n\n")

        file.write("Missing Values After Cleaning:\n")
        file.write(str(df_cleaned.isnull().sum()))
        file.write("\n\n")

        file.write("Duplicate Rows After Cleaning:\n")
        file.write(str(df_cleaned.duplicated().sum()))
        file.write("\n\n")

        file.write("Target Distribution:\n")
        file.write(str(df_cleaned["suspicious_flag"].value_counts()))
        file.write("\n\n")

        file.write("Target Distribution Percentage:\n")
        file.write(str((df_cleaned["suspicious_flag"].value_counts(normalize=True) * 100).round(2)))
        file.write("\n\n")

        file.write("Column List:\n")
        file.write(str(df_cleaned.columns.tolist()))
        file.write("\n")


def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading raw FinCrime dataset...")
    df_raw = load_dataset()

    inspect_dataset(df_raw)
    validate_dataset(df_raw)

    print("\nCleaning dataset...")
    df_cleaned = clean_dataset(df_raw)

    print("\n========== CLEANED DATASET OVERVIEW ==========")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum())
    print("\nSample cleaned case note:")
    print(df_cleaned[["case_note", "case_note_cleaned"]].head(3))

    df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    save_data_quality_report(df_raw, df_cleaned)

    print("\nProcessed dataset saved successfully.")
    print(f"Saved at: {PROCESSED_DATA_PATH}")
    print(f"Data quality report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    main()