import csv
import json
from pathlib import Path
from typing import Any

from src.utils import ROOT


FREEZE_DIR = ROOT / "freeze" / "pilot-v1.1"
ANALYSIS_DIR = ROOT / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def build_round1_dataset() -> list[dict[str, Any]]:
    manifest_path = (
        FREEZE_DIR
        / "prepared_manifest.csv"
    )

    parsed_dir = (
        FREEZE_DIR
        / "parsed"
    )

    metadata_dir = (
        FREEZE_DIR
        / "metadata"
    )

    rows = read_csv(
        manifest_path
    )

    if len(rows) != 72:
        raise RuntimeError(
            f"Expected 72 Round 1 runs, "
            f"found {len(rows)}."
        )

    dataset = []

    for row in rows:
        run_id = row["run_id"]

        if row["status"] != "completed":
            raise RuntimeError(
                f"{run_id} is not completed."
            )

        parsed_path = (
            parsed_dir
            / f"{run_id}.json"
        )

        metadata_path = (
            metadata_dir
            / f"{run_id}.json"
        )

        if not parsed_path.exists():
            raise RuntimeError(
                f"Missing parsed output for {run_id}."
            )

        if not metadata_path.exists():
            raise RuntimeError(
                f"Missing metadata for {run_id}."
            )

        parsed = read_json(
            parsed_path
        )

        metadata = read_json(
            metadata_path
        )

        if metadata["run_id"] != run_id:
            raise RuntimeError(
                f"Metadata run_id mismatch for {run_id}."
            )

        record = {
            "run_id": run_id,
            "case_id": row["case_id"],
            "domain": row["domain"],
            "condition": row["condition"],
            "model_slot": row["model_slot"],
            "repetition": int(
                row["repetition"]
            ),
            "prompt_hash": row["prompt_hash"],
            "provider": metadata["provider"],
            "model_id": metadata["model_id"],
            "status": row["status"],
            "attribution": parsed[
                "attribution"
            ],
            "confidence": parsed[
                "confidence"
            ],
            "justification": parsed[
                "justification"
            ],
            "evidence_used": json.dumps(
                parsed["evidence_used"],
                ensure_ascii=False,
            ),
            "limitations": json.dumps(
                parsed["limitations"],
                ensure_ascii=False,
            ),
            "action": parsed["action"],
            "input_tokens": metadata[
                "usage"
            ]["input_tokens"],
            "output_tokens": metadata[
                "usage"
            ]["output_tokens"],
            "total_tokens": metadata[
                "usage"
            ]["total_tokens"],
            "executed_at_utc": metadata[
                "executed_at_utc"
            ],
        }

        dataset.append(
            record
        )

    return dataset

def build_round2_dataset() -> list[dict[str, Any]]:
    manifest_path = (
        FREEZE_DIR
        / "revision_manifest.csv"
    )

    parsed_dir = (
        FREEZE_DIR
        / "revision_parsed"
    )

    metadata_dir = (
        FREEZE_DIR
        / "revision_metadata"
    )

    round1_path = (
        ANALYSIS_DIR
        / "round1_dataset.csv"
    )

    rows = read_csv(
        manifest_path
    )

    if len(rows) != 48:
        raise RuntimeError(
            f"Expected 48 Round 2 runs, "
            f"found {len(rows)}."
        )

    round1_rows = read_csv(
        round1_path
    )

    round1_by_id = {
        row["run_id"]: row
        for row in round1_rows
    }

    if len(round1_by_id) != 72:
        raise RuntimeError(
            "Round 1 analysis dataset must "
            "contain exactly 72 unique runs."
        )

    dataset = []

    for row in rows:
        revision_id = (
            f"{row['initial_run_id']}"
            f"__{row['update_id']}"
        )

        if row["status"] != "completed":
            raise RuntimeError(
                f"{revision_id} is not completed."
            )

        initial_run_id = row[
            "initial_run_id"
        ]

        if initial_run_id not in round1_by_id:
            raise RuntimeError(
                f"Missing Round 1 run "
                f"{initial_run_id} for {revision_id}."
            )

        parsed_path = (
            parsed_dir
            / f"{revision_id}.json"
        )

        metadata_path = (
            metadata_dir
            / f"{revision_id}.json"
        )

        if not parsed_path.exists():
            raise RuntimeError(
                f"Missing parsed output for "
                f"{revision_id}."
            )

        if not metadata_path.exists():
            raise RuntimeError(
                f"Missing metadata for "
                f"{revision_id}."
            )

        parsed = read_json(
            parsed_path
        )

        metadata = read_json(
            metadata_path
        )

        initial = round1_by_id[
            initial_run_id
        ]

        record = {
            "revision_run_id": revision_id,
            "initial_run_id": initial_run_id,
            "case_id": row["case_id"],
            "domain": row["domain"],
            "condition": row["condition"],
            "model_slot": row["model_slot"],
            "repetition": int(
                row["repetition"]
            ),
            "update_id": row["update_id"],
            "update_type": row["update_type"],
            "prompt_hash": row["prompt_hash"],
            "provider": metadata["provider"],
            "model_id": metadata["model_id"],
            "status": row["status"],
            "initial_attribution": initial[
                "attribution"
            ],
            "initial_confidence": int(
                initial["confidence"]
            ),
            "initial_action": initial[
                "action"
            ],
            "revised_attribution": parsed[
                "revised_attribution"
            ],
            "confidence": int(
                parsed["confidence"]
            ),
            "revision_decision": parsed[
                "revision_decision"
            ],
            "justification": parsed[
                "justification"
            ],
            "evidence_affected": json.dumps(
                parsed["evidence_affected"],
                ensure_ascii=False,
            ),
            "input_tokens": metadata[
                "usage"
            ]["input_tokens"],
            "output_tokens": metadata[
                "usage"
            ]["output_tokens"],
            "total_tokens": metadata[
                "usage"
            ]["total_tokens"],
            "executed_at_utc": metadata[
                "executed_at_utc"
            ],
        }

        dataset.append(
            record
        )

    return dataset

def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            "Cannot write empty analysis dataset."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    round1 = build_round1_dataset()

    round1_path = (
        ANALYSIS_DIR
        / "round1_dataset.csv"
    )

    write_csv(
        round1,
        round1_path,
    )

    round2 = build_round2_dataset()

    round2_path = (
        ANALYSIS_DIR
        / "round2_dataset.csv"
    )

    write_csv(
        round2,
        round2_path,
    )

    print(
        f"Round 1 rows: {len(round1)}"
    )

    print(
        f"Round 2 rows: {len(round2)}"
    )

    print(
        f"Round 1 written to: {round1_path}"
    )

    print(
        f"Round 2 written to: {round2_path}"
    )

if __name__ == "__main__":
    main()
