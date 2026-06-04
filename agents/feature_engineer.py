import pandas as pd

from reports.structured_report import StructuredReport
from utils.llm_client import ask_llm
from utils.tools import *


class FeatureEngineerAgent:

    def run(self, report):

        df = pd.read_csv("outputs/clean_data.csv")

        prompt = f"""
You are the Feature Engineer.

Previous report:

{report.to_text()}

Columns:
{list(df.columns)}

Suggest:
1. New features.
2. Encodings.
3. Features to remove.

Return detailed reasoning.
"""

        reasoning = ask_llm(prompt)

        log = []

        for col in df.select_dtypes(include="object").columns:

            df = encode_categorical(df, col)
            log.append(f"Encoded {col}")

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) >= 2:

            c1 = numeric_cols[0]
            c2 = numeric_cols[1]

            df[f"{c1}_{c2}_ratio"] = (
                df[c1] /
                (df[c2] + 1)
            )

            log.append(
                f"Created ratio feature "
                f"{c1}_{c2}_ratio"
            )

        df.to_csv(
            "outputs/engineered_data.csv",
            index=False
        )

        report = StructuredReport(
            "Feature Engineer",
            "\n".join(log),
            reasoning
        )

        return report