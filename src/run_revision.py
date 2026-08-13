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


def load_revision_manifest() -> list[dict[str, Any]]:
    """Load the Round 2 execution manifest."""

    manifest_path = (
        ROOT
        / "runs"
        / "revision_manifest.csv"
    )

    if not manifest_path.exists():
        raise RuntimeError(
            "Revision manifest not found. "
            "Run src.build_revision_prompts after Round 1."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def save_revision_manifest(
    rows: list[dict[str, Any]],
) -> None:
    """Write the Round 2 manifest atomically."""

    manifest_path = (
        ROOT
        / "runs"
        / "revision_manifest.csv"
    )

    temporary_path = manifest_path.with_suffix(
        ".tmp"
    )

    if not rows:
        raise RuntimeError(
            "Cannot write an empty revision manifest."
        )

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

    temporary_path.replace(
        manifest_path
    )


def load_prompt(
    prompt_file: str,
) -> tuple[str, str]:
    """Read and split a materialized Round 2 prompt."""

    path = ROOT / prompt_file

    if not path.exists():
        raise RuntimeError(
            f"Revision prompt not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    system_marker = "===== SYSTEM PROMPT ====="
    user_marker = "===== USER PROMPT ====="

    if (
        system_marker not in text
        or user_marker not in text
    ):
        raise RuntimeError(
            f"Invalid revision prompt structure: "
            f"{prompt_file}"
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
    """Return provider, model, generation config and retries."""

    config = load_yaml(
        "config/models.yaml"
    )

    models = config["models"]
    execution = config.get(
        "execution",
        {},
    )

    if model_slot not in models:
        raise RuntimeError(
            f"Unknown model slot: {model_slot}"
        )

    model = models[model_slot]

    max_retries = int(
        execution.get(
            "max_retries",
            0,
        )
    )

    return (
        model["provider"],
        model["model_id"],
        model.get(
            "generation",
            {},
        ),
        max_retries,
    )


def load_revision_validator() -> Draft202012Validator:
    """Load the Round 2 output schema."""

    paths = load_yaml(
        "config/paths.yaml"
    )

    schema_path = paths["prompt_files"][
        "revision_output_schema"
    ]

    schema = load_json(
        schema_path
    )

    return Draft202012Validator(
        schema
    )


def parse_and_validate(
    response_text: str,
    validator: Draft202012Validator,
) -> tuple[
    str,
    dict[str, Any] | None,
    str | None,
]:
    """Parse and validate one Round 2 response."""

    try:
        parsed = json.loads(
            response_text
        )

    except json.JSONDecodeError as exc:
        return (
            STATUS_INVALID_JSON,
            None,
            str(exc),
        )

    errors = sorted(
        validator.iter_errors(
            parsed
        ),
        key=lambda error: list(
            error.path
        ),
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


def revision_run_id(
    row: dict[str, Any],
) -> str:
    """Create a deterministic Round 2 execution identifier."""

    return (
        f"{row['initial_run_id']}"
        f"__{row['update_id']}"
    )


def save_provider_error(
    row: dict[str, Any],
    provider_name: str,
    model_id: str,
    exc: ProviderError,
    max_retries: int,
) -> None:
    """Persist normalized Round 2 provider-error metadata."""

    run_id = revision_run_id(
        row
    )

    metadata_dir = (
        ROOT
        / "runs"
        / "revision_metadata"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        metadata_dir
        / f"{run_id}.json"
    )

    metadata = {
        "revision_run_id": run_id,
        "initial_run_id": row["initial_run_id"],
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


def save_revision_artifacts(
    row: dict[str, Any],
    response: Any,
    status: str,
    validation_error: str | None,
    max_retries: int,
) -> None:
    """Persist Round 2 raw, parsed, and metadata artifacts."""

    run_id = revision_run_id(
        row
    )

    raw_dir = (
        ROOT
        / "runs"
        / "revision_raw"
    )

    parsed_dir = (
        ROOT
        / "runs"
        / "revision_parsed"
    )

    metadata_dir = (
        ROOT
        / "runs"
        / "revision_metadata"
    )

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

    raw_path = (
        raw_dir
        / f"{run_id}.txt"
    )

    parsed_path = (
        parsed_dir
        / f"{run_id}.json"
    )

    metadata_path = (
        metadata_dir
        / f"{run_id}.json"
    )

    if raw_path.exists():
        raise RuntimeError(
            f"Revision raw output already exists "
            f"for {run_id}."
        )

    raw_path.write_text(
        response.text,
        encoding="utf-8",
    )

    parsed = None

    if status != STATUS_INVALID_JSON:
        try:
            parsed = json.loads(
                response.text
            )
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
        "revision_run_id": run_id,
        "initial_run_id": row["initial_run_id"],
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
        description="Run Round 2 selective-revisability evaluations."
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect revision runs without invoking providers.",
    )

    mode.add_argument(
        "--real",
        action="store_true",
        help="Execute revision runs using real providers.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of pending revision runs.",
    )

    parser.add_argument(
        "--confirm-revision",
        action="store_true",
        help=(
            "Explicitly authorize execution of more "
            "than two real Round 2 runs."
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
        and not args.confirm_revision
    ):
        raise RuntimeError(
            "Real execution of more than two revision "
            "runs requires --confirm-revision."
        )

    rows = load_revision_manifest()

    pending = [
        row
        for row in rows
        if row["status"] == STATUS_PENDING
    ]

    selected = pending[
        : args.limit
    ]

    print(
        f"Pending revision runs: {len(pending)}"
    )

    print(
        f"Selected revision runs: {len(selected)}"
    )

    if args.dry_run:
        for row in selected:
            print(
                revision_run_id(row),
                row["initial_run_id"],
                row["case_id"],
                row["condition"],
                row["model_slot"],
                row["repetition"],
                row["update_id"],
                row["prompt_file"],
            )

        return

    validator = load_revision_validator()

    for row in selected:

        run_id = revision_run_id(
            row
        )

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

            save_revision_manifest(
                rows
            )

            print(
                f"{run_id}: "
                f"{STATUS_PROVIDER_ERROR} - "
                f"{exc.category}"
            )

            continue

        save_revision_artifacts(
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

        save_revision_manifest(
            rows
        )


if __name__ == "__main__":
    main()
