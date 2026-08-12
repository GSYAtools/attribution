import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.utils import ROOT, load_json, load_yaml


def load_output_schema() -> dict[str, Any]:
    """Load the Round 1 output JSON schema."""
    paths = load_yaml("config/paths.yaml")
    schema_path = paths["prompt_files"]["output_schema"]

    return load_json(schema_path)


def validate_output_file(
    output_path: Path,
    validator: Draft202012Validator,
) -> tuple[bool, str]:
    """Validate one parsed JSON output against the schema."""

    try:
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        return False, f"invalid_json: {exc}"

    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: list(error.path),
    )

    if errors:
        messages = []

        for error in errors:
            location = ".".join(
                str(item) for item in error.path
            )

            if not location:
                location = "<root>"

            messages.append(
                f"{location}: {error.message}"
            )

        return False, "schema_invalid: " + " | ".join(messages)

    return True, "valid"


def validate_all_outputs() -> None:
    """Validate all parsed Round 1 outputs."""

    schema = load_output_schema()

    validator = Draft202012Validator(schema)

    parsed_dir = ROOT / "runs" / "parsed"

    output_files = sorted(
        parsed_dir.glob("*.json")
    )

    if not output_files:
        print("No parsed outputs found.")
        return

    valid = 0
    invalid = 0

    for output_path in output_files:
        ok, status = validate_output_file(
            output_path,
            validator,
        )

        if ok:
            valid += 1
            print(f"OK   {output_path.name}")
        else:
            invalid += 1
            print(
                f"FAIL {output_path.name}: {status}"
            )

    print()
    print(f"Valid outputs: {valid}")
    print(f"Invalid outputs: {invalid}")


if __name__ == "__main__":
    validate_all_outputs()
