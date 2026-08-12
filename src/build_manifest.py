import csv
import random
from pathlib import Path

from src.build_prompts import build_all_prompts
from src.utils import ROOT, load_yaml


def build_execution_records() -> list[dict]:
    """Build the complete execution manifest without calling any API."""

    config = load_yaml("config/experiment.yaml")
    model_config = load_yaml("config/models.yaml")
    paths = load_yaml("config/paths.yaml")

    seed = config["experiment"]["seed"]
    repetitions = config["design"]["repetitions"]

    enabled_models = [
        model_name
        for model_name, model_data in model_config["models"].items()
        if model_data.get("enabled", False)
    ]

    if not enabled_models:
        raise RuntimeError("No enabled models are configured.")

    prompts = build_all_prompts()

    records = []

    run_number = 1

    for prompt in prompts:
        for model_name in enabled_models:
            for repetition in range(1, repetitions + 1):
                records.append(
                    {
                        "run_id": f"RUN_{run_number:04d}",
                        "case_id": prompt["case_id"],
                        "domain": prompt["domain"],
                        "condition": prompt["condition"],
                        "model_slot": model_name,
                        "repetition": repetition,
                        "prompt_hash": prompt["prompt_hash"],
                        "status": "pending",
                    }
                )

                run_number += 1

    rng = random.Random(seed)
    rng.shuffle(records)

    output_path = ROOT / "experiment_manifest.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "case_id",
                "domain",
                "condition",
                "model_slot",
                "repetition",
                "prompt_hash",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Created manifest: {output_path}")
    print(f"Total execution records: {len(records)}")

    conditions = {}

    for record in records:
        condition = record["condition"]
        conditions[condition] = conditions.get(condition, 0) + 1

    print("Condition distribution:")
    for condition in sorted(conditions):
        print(f"  {condition}: {conditions[condition]}")

    print("Model distribution:")
    for model_name in enabled_models:
        count = sum(
            1
            for record in records
            if record["model_slot"] == model_name
        )
        print(f"  {model_name}: {count}")


if __name__ == "__main__":
    build_execution_records()
