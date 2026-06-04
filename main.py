from agents.data_cleaner import DataCleanerAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_trainer import ModelTrainerAgent

import os


def save_log(text):

    with open(
        "outputs/execution_log.txt",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(text)
        f.write("\n\n")


def main():

    os.makedirs("outputs", exist_ok=True)

    cleaner = DataCleanerAgent()
    engineer = FeatureEngineerAgent()
    trainer = ModelTrainerAgent()

    print("Running Data Cleaner...")
    report1 = cleaner.run(
        "data/raw_data.csv"
    )

    save_log(report1.to_text())

    print("Running Feature Engineer...")
    report2 = engineer.run(report1)

    save_log(report2.to_text())

    target = input(
        "Enter target column: "
    )

    print("Running Model Trainer...")
    report3 = trainer.run(
        report2,
        target
    )

    save_log(report3.to_text())

    with open(
        "outputs/final_report.md",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("# Final Report\n\n")

        f.write(
            "## Data Cleaner\n"
        )
        f.write(report1.to_text())

        f.write(
            "\n\n## Feature Engineer\n"
        )
        f.write(report2.to_text())

        f.write(
            "\n\n## Model Trainer\n"
        )
        f.write(report3.to_text())

    print(
        "Finished. Check outputs folder."
    )


if __name__ == "__main__":
    main()