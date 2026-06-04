import pandas as pd
import subprocess
import tempfile

from reports.structured_report import StructuredReport
from utils.llm_client import ask_llm


class ModelTrainerAgent:

    def run(self, report, target):

        df = pd.read_csv(
            "outputs/engineered_data.csv"
        )

        metrics_history = []

        best_score = 0

        for iteration in range(3):

            prompt = f"""
You are the Model Trainer.

Dataset columns:
{list(df.columns)}

Target:
{target}

Previous metrics:
{metrics_history}

Generate executable Python code.

Requirements:
- XGBoost classifier
- train_test_split
- Accuracy
- Recall
- F1

Output ONLY code.
"""

            code = ask_llm(prompt)

            with tempfile.NamedTemporaryFile(
                suffix=".py",
                mode="w",
                delete=False
            ) as f:

                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True
            )

            output = result.stdout

            metrics_history.append(output)

            try:
                score = float(
                    output.split("F1:")[-1]
                    .strip()
                )

            except:
                score = 0

            if score > best_score:
                best_score = score

            if score > 0.85:
                break

        report = StructuredReport(
            "Model Trainer",
            "\n".join(metrics_history),
            f"Best F1 = {best_score}"
        )

        return report