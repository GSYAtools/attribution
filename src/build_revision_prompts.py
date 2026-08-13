import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.build_prompts import (
    format_criteria,
    format_output_instruction,
    format_vignette,
    load_experiment_materials,
)
from src.utils import ROOT, load_json, load_text, load_yaml


REVISION_CASES = {"PX_1005", "PX_1006"}
REVISION_CONDITIONS = {"C0", "C1", "C2"}
UPDATE_TYPES = ("material_defeater", "non_material_control")


def load_revision_materials() -> dict[str, Any]:
    """Load materials required for Round 2 prompt construction."""

    paths = load_yaml("config/paths.yaml")

    data_files = paths["data_files"]
    prompt_files = paths["prompt_files"]

    return {
        "vignettes": load_json(data_files["vignettes"]),
        "criteria": load_json(data_files["criteria_cards"]),
        "governance": load_text(
            data_files["governance_card"]
        ).strip(),
        "system": load_text(
            prompt_files["system"]
        ).strip(),
        "revision_prompt": load_text(
            prompt_files["revision_prompt"]
        ).strip(),
        "revision_schema": load_json(
            prompt_files["revision_output_schema"]
        ),
        "updates": load_json(
            data_files["revalidation_updates"]
        ),
    }


def get_vignette(
    materials: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Return one vignette by case identifier."""

    for vignette in materials["vignettes"]["vignettes"]:
        if vignette["case_id"] == case_id:
            return vignette

    raise ValueError(
        f"Vignette '{case_id}' not found."
    )


def build_evaluation_context(
    vignette: dict[str, Any],
    condition: str,
    materials: dict[str, Any],
) -> str:
    """Reconstruct the original evaluative context without Round 1 schema."""

    if condition not in REVISION_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'."
        )

    sections = [
        format_vignette(vignette),
    ]

    if condition in {"C1", "C2"}:
        sections.append(
            format_criteria(
                vignette["domain"],
                materials["criteria"],
            )
        )

    if condition == "C2":
        sections.append(
            materials["governance"]
        )

    return "\n\n---\n\n".join(sections)


def load_initial_assessment(
    initial_run_id: str,
) -> dict[str, Any]:
    """Load the complete structured Round 1 assessment."""

    path = (
        ROOT
        / "runs"
        / "parsed"
        / f"{initial_run_id}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Initial assessment not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_updates_for_case(
    materials: dict[str, Any],
    case_id: str,
) -> dict[str, dict[str, Any]]:
    """Return the two hidden update variants for a case."""

    if case_id not in materials["updates"]["updates"]:
        raise ValueError(
            f"No revalidation updates configured for '{case_id}'."
        )

    return materials["updates"]["updates"][case_id]


def build_revision_user_prompt(
    vignette: dict[str, Any],
    condition: str,
    initial_assessment: dict[str, Any],
    update_text: str,
    materials: dict[str, Any],
) -> str:
    """Build one Round 2 prompt without exposing update metadata."""

    evaluation_context = build_evaluation_context(
        vignette,
        condition,
        materials,
    )

    previous_assessment = json.dumps(
        initial_assessment,
        indent=2,
        ensure_ascii=False,
    )

    sections = [
        evaluation_context,

        "Previous assessment",

        previous_assessment,

        "Additional information",

        update_text,

        materials["revision_prompt"],

        format_output_instruction(
            materials["revision_schema"]
        ),
    ]

    return "\n\n---\n\n".join(sections)


def prompt_hash(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Return a SHA-256 hash for the complete revision prompt."""

    payload = (
        system_prompt
        + "\n\n"
        + user_prompt
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def build_revision_records() -> list[dict[str, Any]]:
    """
    Build Round 2 records from completed Round 1 executions.

    Only PX_1005 and PX_1006 are eligible.
    """

    materials = load_revision_materials()

    prepared_manifest_path = (
        ROOT
        / "runs"
        / "prepared_manifest.csv"
    )

    if not prepared_manifest_path.exists():
        raise RuntimeError(
            "Prepared manifest not found."
        )

    with prepared_manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    records = []

    for row in rows:

        if row["case_id"] not in REVISION_CASES:
            continue

        if row["condition"] not in REVISION_CONDITIONS:
            continue

        if row["status"] != "completed":
            continue

        vignette = get_vignette(
            materials,
            row["case_id"],
        )

        initial_assessment = load_initial_assessment(
            row["run_id"]
        )

        updates = get_updates_for_case(
            materials,
            row["case_id"],
        )

        for update_type in UPDATE_TYPES:

            update = updates[update_type]

            user_prompt = build_revision_user_prompt(
                vignette=vignette,
                condition=row["condition"],
                initial_assessment=initial_assessment,
                update_text=update["text"],
                materials=materials,
            )

            records.append(
                {
                    "initial_run_id": row["run_id"],
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "condition": row["condition"],
                    "model_slot": row["model_slot"],
                    "repetition": row["repetition"],
                    "update_id": update["update_id"],
                    "update_type": update_type,
                    "system_prompt": materials["system"],
                    "user_prompt": user_prompt,
                    "prompt_hash": prompt_hash(
                        materials["system"],
                        user_prompt,
                    ),
                    "status": "pending",
                }
            )

    return records


def main() -> None:
    records = build_revision_records()

    print(
        f"Revision records created: {len(records)}"
    )

    if not records:
        print(
            "No completed Round 1 runs are available "
            "for revision yet."
        )
        return

    prompt_dir = (
        ROOT
        / "runs"
        / "revision_prompts"
    )

    prompt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        filename = (
            f"REV_{index:04d}_"
            f"{record['case_id']}_"
            f"{record['condition']}_"
            f"{record['model_slot']}_"
            f"{record['repetition']}_"
            f"{record['update_id']}.txt"
        )

        path = prompt_dir / filename

        content = (
            "===== SYSTEM PROMPT =====\n\n"
            f"{record['system_prompt']}\n\n"
            "===== USER PROMPT =====\n\n"
            f"{record['user_prompt']}\n"
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        record["prompt_file"] = str(
            path.relative_to(ROOT)
        )

    manifest_path = (
        ROOT
        / "runs"
        / "revision_manifest.csv"
    )

    fieldnames = [
        "initial_run_id",
        "case_id",
        "domain",
        "condition",
        "model_slot",
        "repetition",
        "update_id",
        "update_type",
        "prompt_file",
        "prompt_hash",
        "status",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    field: record[field]
                    for field in fieldnames
                }
            )

    print(
        f"Revision prompts written to: {prompt_dir}"
    )

    print(
        f"Revision manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
