import csv
from pathlib import Path

from src.build_prompts import build_all_prompts
from src.utils import ROOT


def main() -> None:
    """Materialize unique prompts and create a prepared execution manifest."""

    prompt_records = build_all_prompts()

    prompt_dir = ROOT / "runs" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_index = {}

    for record in prompt_records:
        prompt_key = (
            record["case_id"],
            record["condition"],
        )

        prompt_filename = (
            f"{record['case_id']}_{record['condition']}.txt"
        )

        prompt_path = prompt_dir / prompt_filename

        prompt_content = (
            "===== SYSTEM PROMPT =====\n\n"
            f"{record['system_prompt']}\n\n"
            "===== USER PROMPT =====\n\n"
            f"{record['user_prompt']}\n"
        )

        prompt_path.write_text(
            prompt_content,
            encoding="utf-8",
        )

        prompt_index[prompt_key] = {
            "prompt_file": str(
                prompt_path.relative_to(ROOT)
            ),
            "prompt_hash": record["prompt_hash"],
        }

    manifest_path = ROOT / "experiment_manifest.csv"
    prepared_manifest_path = ROOT / "runs" / "prepared_manifest.csv"

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("experiment_manifest.csv is empty.")

    prepared_rows = []

    for row in rows:
        key = (
            row["case_id"],
            row["condition"],
        )

        if key not in prompt_index:
            raise RuntimeError(
                f"No prompt found for {key}."
            )

        prompt_info = prompt_index[key]

        if row["prompt_hash"] != prompt_info["prompt_hash"]:
            raise RuntimeError(
                f"Prompt hash mismatch for {key}."
            )

        prepared_row = dict(row)
        prepared_row["prompt_file"] = prompt_info["prompt_file"]

        prepared_rows.append(prepared_row)

    fieldnames = list(prepared_rows[0].keys())

    with prepared_manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(prepared_rows)

    print(f"Materialized prompts: {len(prompt_records)}")
    print(f"Prepared executions: {len(prepared_rows)}")
    print(f"Prompt directory: {prompt_dir}")
    print(f"Prepared manifest: {prepared_manifest_path}")


if __name__ == "__main__":
    main()
