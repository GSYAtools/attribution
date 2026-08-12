import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils import ROOT, load_json, load_text, load_yaml


def load_experiment_materials() -> dict[str, Any]:
    """Load the materials required to construct Round 1 prompts."""

    paths = load_yaml("config/paths.yaml")

    data_files = paths["data_files"]
    prompt_files = paths["prompt_files"]

    return {
        "vignettes": load_json(data_files["vignettes"]),
        "criteria": load_json(data_files["criteria_cards"]),
        "governance": load_text(data_files["governance_card"]).strip(),
        "system": load_text(prompt_files["system"]).strip(),
        "output_schema": load_json(prompt_files["output_schema"]),
    }


def format_vignette(vignette: dict[str, Any]) -> str:
    """Render a vignette without exposing experimental metadata."""

    evidence = "\n".join(
        f"- {item}" for item in vignette["evidence"]
    )

    return (
        f"Context\n\n"
        f"{vignette['context']}\n\n"
        f"Available evidence\n\n"
        f"{evidence}\n\n"
        f"Evaluation request\n\n"
        f"{vignette['request']}"
    )


def format_criteria(domain: str, criteria_data: dict[str, Any]) -> str:
    """Return the external evaluation criteria for a domain."""

    cards = criteria_data["cards"]

    if domain not in cards:
        raise ValueError(
            f"No evaluation criteria configured for domain '{domain}'."
        )

    card = cards[domain]

    return (
        f"{card['title']}\n\n"
        f"{card['text']}"
    )


def format_output_instruction(schema: dict[str, Any]) -> str:
    """Render the common structured-output instruction."""

    schema_text = json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )

    return (
        "Return only one JSON object that conforms to the following "
        "schema. Do not include Markdown fences or additional text.\n\n"
        f"{schema_text}"
    )


def build_user_prompt(
    vignette: dict[str, Any],
    condition: str,
    materials: dict[str, Any],
) -> str:
    """Build the user prompt for C0, C1, or C2."""

    if condition not in {"C0", "C1", "C2"}:
        raise ValueError(
            f"Unknown experimental condition '{condition}'."
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
        sections.append(materials["governance"])

    sections.append(
        format_output_instruction(materials["output_schema"])
    )

    return "\n\n---\n\n".join(sections)


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Return a SHA-256 hash for the complete prompt."""

    payload = (
        system_prompt
        + "\n\n"
        + user_prompt
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def get_vignettes(materials: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the configured pilot vignettes."""

    return materials["vignettes"]["vignettes"]


def build_all_prompts() -> list[dict[str, Any]]:
    """Construct every case-condition prompt for Round 1."""

    materials = load_experiment_materials()
    system_prompt = materials["system"]

    records = []

    for vignette in get_vignettes(materials):
        for condition in ("C0", "C1", "C2"):

            user_prompt = build_user_prompt(
                vignette=vignette,
                condition=condition,
                materials=materials,
            )

            records.append(
                {
                    "case_id": vignette["case_id"],
                    "domain": vignette["domain"],
                    "condition": condition,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "prompt_hash": prompt_hash(
                        system_prompt,
                        user_prompt,
                    ),
                }
            )

    return records


def main() -> None:
    records = build_all_prompts()

    print(
        f"Built {len(records)} unique case-condition prompts."
    )

    for record in records:
        print(
            record["case_id"],
            record["condition"],
            record["prompt_hash"][:12],
        )


if __name__ == "__main__":
    main()
