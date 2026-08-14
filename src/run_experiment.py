import argparse
import csv
import json
from datetime import datetime, timezone
from typing import Any

from jsonschema import Draft202012Validator

from src.providers.errors import ProviderError
from src.providers.provider_factory import create_provider
from src.retry_manager import execute_with_retry
from src.utils import ROOT, load_json, load_yaml


STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_INVALID_JSON = "invalid_json"
STATUS_SCHEMA_INVALID = "schema_invalid"
STATUS_PROVIDER_ERROR = "provider_error"


def load_manifest() -> list[dict[str, Any]]:
    """Load the prepared execution manifest."""

    manifest_path = ROOT / "runs" / "prepared_manifest.csv"

    if not manifest_path.exists():
        raise RuntimeError(
            "Prepared manifest not found. "
            "Run src.prepare_runs first."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def save_manifest(rows: list[dict[str, Any]]) -> None:
    """Write the prepared execution manifest atomically."""

    manifest_path = ROOT / "runs" / "prepared_manifest.csv"
    temporary_path = manifest_path.with_suffix(".tmp")

    if not rows:
        raise RuntimeError("Cannot write an empty manifest.")

    fieldnames = list(rows[0].keys())

    with temporary_path.open(
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

    temporary_path.replace(manifest_path)


def load_prompt(prompt_file: str) -> tuple[str, str]:
    """Read a materialized prompt and split system/user sections."""

    path = ROOT / prompt_file

    if not path.exists():
        raise RuntimeError(
            f"Prompt file not found: {path}"
        )

    text = path.read_text(encoding="utf-8")

    system_marker = "===== SYSTEM PROMPT ====="
    user_marker = "===== USER PROMPT ====="

    if system_marker not in text or user_marker not in text:
        raise RuntimeError(
            f"Invalid prompt structure in {prompt_file}"
        )

    _, remainder = text.split(
        system_marker,
        1,
    )

    system_text, user_text = remainder.split(
        user_marker,
        1,
    )

    return (
        system_text.strip(),
        user_text.strip(),
    )


def get_model_configuration(
    model_slot: str,
) -> tuple[str, str, dict[str, Any], int]:
    """
    Return provider, model identifier, generation configuration,
    and maximum retry count.
    """

    config = load_yaml("config/models.yaml")

    models = config["models"]
    execution = config.get("execution", {})

    if model_slot not in models:
        raise RuntimeError(
            f"Unknown model slot: {model_slot}"
        )

    model = models[model_slot]

    max_retries = int(
        execution.get("max_retries", 0)
    )

    return (
        model["provider"],
        model["model_id"],
        model.get("generation", {}),
        max_retries,
    )


def load_output_validator(
) -> tuple[dict[str, Any], Draft202012Validator]:
    """Load the Round 1 output schema and compile its validator."""

    paths = load_yaml("config/paths.yaml")

    schema_path = paths["prompt_files"]["output_schema"]
    schema = load_json(schema_path)

    return (
        schema,
        Draft202012Validator(schema),
    )

def parse_and_validate(
    response_text: str,
    validator: Draft202012Validator,
) -> tuple[
    str,
    dict[str, Any] | None,
    str | None,
]:
    """Parse JSON and validate it against the experiment schema."""

    try:
        parsed = json.loads(response_text)

    except json.JSONDecodeError as exc:
        return (
            STATUS_INVALID_JSON,
            None,
            str(exc),
        )

    errors = sorted(
        validator.iter_errors(parsed),
        key=lambda error: list(error.path),
    )

    if errors:
        messages = []

        for error in errors:
            location = ".".join(
                str(item)
                for item in error.path
            )

            if not location:
                location = "<root>"

            messages.append(
                f"{location}: {error.message}"
            )

        return (
            STATUS_SCHEMA_INVALID,
            parsed,
            " | ".join(messages),
        )

    return (
        STATUS_COMPLETED,
        parsed,
        None,
    )


def save_provider_error(
    row: dict[str, Any],
    provider_name: str,
    model_id: str,
    exc: ProviderError,
    max_retries: int,
) -> None:
    """Persist normalized provider-error metadata."""

    run_id = row["run_id"]

    metadata_dir = ROOT / "runs" / "metadata"

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        metadata_dir / f"{run_id}.json"
    )

    metadata = {
        "run_id": run_id,
        "case_id": row["case_id"],
        "domain": row["domain"],
        "condition": row["condition"],
        "model_slot": row["model_slot"],
        "repetition": int(row["repetition"]),
        "prompt_hash": row["prompt_hash"],
        "prompt_file": row["prompt_file"],
        "provider": provider_name,
        "model_id": model_id,
        "status": STATUS_PROVIDER_ERROR,
        "error_category": exc.category,
        "retryable": exc.retryable,
        "max_retries": max_retries,
        "provider_error": exc.message,
        "executed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def save_run_artifacts(
    row: dict[str, Any],
    response: Any,
    status: str,
    validation_error: str | None,
    max_retries: int,
) -> None:
    """Persist raw response, parsed response and metadata."""

    run_id = row["run_id"]

    raw_dir = ROOT / "runs" / "raw"
    parsed_dir = ROOT / "runs" / "parsed"
    metadata_dir = ROOT / "runs" / "metadata"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = raw_dir / f"{run_id}.txt"
    parsed_path = parsed_dir / f"{run_id}.json"
    metadata_path = metadata_dir / f"{run_id}.json"

    if raw_path.exists():
        raise RuntimeError(
            f"Raw output already exists for {run_id}."
        )

    raw_path.write_text(
        response.text,
        encoding="utf-8",
    )

    parsed = None

    if status != STATUS_INVALID_JSON:
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError:
            parsed = None

    if parsed is not None:
        parsed_path.write_text(
            json.dumps(
                parsed,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    metadata = {
        "run_id": run_id,
        "case_id": row["case_id"],
        "domain": row["domain"],
        "condition": row["condition"],
        "model_slot": row["model_slot"],
        "repetition": int(row["repetition"]),
        "prompt_hash": row["prompt_hash"],
        "prompt_file": row["prompt_file"],
        "provider": response.provider,
        "model_id": response.model_id,
        "usage": response.usage,
        "status": status,
        "validation_error": validation_error,
        "max_retries": max_retries,
        "executed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SC5 pilot experiment."
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect runs without invoking a provider.",
    )

    mode.add_argument(
        "--real",
        action="store_true",
        help="Execute using configured providers.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of pending runs to process.",
    )

    parser.add_argument(
        "--confirm-pilot",
        action="store_true",
        help=(
            "Explicitly authorize real execution of "
            "more than two experimental runs."
        ),
    )

    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError(
            "--limit must be at least 1."
        )

    if (
        args.real
        and args.limit > 2
        and not args.confirm_pilot
    ):
        raise RuntimeError(
            "Real execution of more than two runs "
            "requires --confirm-pilot."
        )
    rows = load_manifest()

    pending = [
        row
        for row in rows
        if row["status"] == STATUS_PENDING
    ]

    selected = pending[: args.limit]

    print(f"Pending runs: {len(pending)}")
    print(f"Selected runs: {len(selected)}")

    if args.dry_run:
        for row in selected:
            print(
                row["run_id"],
                row["case_id"],
                row["condition"],
                row["model_slot"],
                row["repetition"],
                row["prompt_file"],
            )

        return

    output_schema, validator = load_output_validator()

    for row in selected:
        run_id = row["run_id"]

        system_prompt, user_prompt = load_prompt(
            row["prompt_file"]
        )

        (
            provider_name,
            model_id,
            generation_config,
            max_retries,
        ) = get_model_configuration(
            row["model_slot"]
        )

        provider = create_provider(
            provider_name
        )

        def operation():
            return provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=model_id,
                generation_config=generation_config,
		output_schema=output_schema,
            )

        try:
            response = execute_with_retry(
                operation=operation,
                max_retries=max_retries,
            )

            status, _, validation_error = (
                parse_and_validate(
                    response.text,
                    validator,
                )
            )

        except ProviderError as exc:
            save_provider_error(
                row=row,
                provider_name=provider_name,
                model_id=model_id,
                exc=exc,
                max_retries=max_retries,
            )

            row["status"] = STATUS_PROVIDER_ERROR
            save_manifest(rows)

            print(
                f"{run_id}: {STATUS_PROVIDER_ERROR} - "
                f"{exc.category}"
            )

            continue

        save_run_artifacts(
            row=row,
            response=response,
            status=status,
            validation_error=validation_error,
            max_retries=max_retries,
        )

        row["status"] = status

        if validation_error:
            print(
                f"{run_id}: {status} - "
                f"{validation_error}"
            )
        else:
            print(
                f"{run_id}: {status}"
            )

        save_manifest(rows)


if __name__ == "__main__":
    main()
