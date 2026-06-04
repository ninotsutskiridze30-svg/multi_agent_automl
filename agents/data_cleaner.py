import pandas as pd

from utils.tools import *
from reports.structured_report import StructuredReport
from utils.llm_client import ask_llm


class DataCleanerAgent:

    def run(self, csv_path):

        df = pd.read_csv(csv_path)

        metadata = inspect_metadata(df)

        prompt = f"""
You are the Data Cleaner.

Dataset metadata:

{metadata}

Decide:
1. Which columns to drop.
2. How to fill missing values.
3. Any datatype fixes.

Return concise reasoning.
"""

        reasoning = ask_llm(prompt)

        log = []

        for col in df.columns:

            missing_ratio = df[col].isnull().mean()

            if missing_ratio > 0.4:
                df = drop_column(df, col)
                log.append(f"Dropped {col}")

            elif missing_ratio > 0:

                if pd.api.types.is_numeric_dtype(df[col]):
                    df = impute_missing(df, col, "median")
                    log.append(f"Median imputed {col}")

                else:
                    df = impute_missing(df, col, "mode")
                    log.append(f"Mode imputed {col}")

        df.to_csv("outputs/clean_data.csv", index=False)

        report = StructuredReport(
            "Data Cleaner",
            "\n".join(log),
            reasoning
        )

        return report