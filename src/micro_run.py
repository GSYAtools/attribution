import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.providers.errors import ProviderError
from src.providers.provider_factory import create_provider
from src.retry_manager import execute_with_retry
from src.utils import ROOT, load_yaml


MICRO_RUN_DIR = ROOT / "micro_runs"

TEST_SYSTEM_PROMPT = """You are participating in a technical connectivity test.
Return the requested information exactly and do not add unrelated content."""

TEST_USER_PROMPT = """Return exactly one JSON object with the following fields:

{
    "status": "ok",
    "message": "provider connectivity test"
}

Do not include Markdown fences or additional text."""


def load_model_configuration(
    model_slot: str,
) -> tuple[str, str, dict[str, Any], int]:
    """Load provider, model and execution configuration."""

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


def save_result(
    result: dict[str, Any],
    filename: str,
) -> Path:
    """Save one micro-run result."""

    MICRO_RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = MICRO_RUN_DIR / filename

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def run_one(model_slot: str) -> None:
    """Execute one isolated provider connectivity test."""

    (
        provider_name,
        model_id,
        generation_config,
        max_retries,
    ) = load_model_configuration(model_slot)

    provider = create_provider(provider_name)

    started_at = datetime.now(timezone.utc)

    def operation():
        return provider.generate(
            system_prompt=TEST_SYSTEM_PROMPT,
            user_prompt=TEST_USER_PROMPT,
            model_id=model_id,
            generation_config=generation_config,
        )

    try:
        response = execute_with_retry(
            operation=operation,
            max_retries=max_retries,
        )

    except ProviderError as exc:
        finished_at = datetime.now(timezone.utc)

        result = {
            "test": "SC5_PROVIDER_MICRO_RUN",
            "model_slot": model_slot,
            "provider": provider_name,
            "model_id": model_id,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "status": "provider_error",
            "error_category": exc.category,
            "retryable": exc.retryable,
            "max_retries": max_retries,
            "error_message": exc.message,
        }

        output_path = save_result(
            result,
            f"{model_slot}_{provider_name}_error.json",
        )

        print("Micro-run failed")
        print(f"Provider: {provider_name}")
        print(f"Model: {model_id}")
        print(f"Error category: {exc.category}")
        print(f"Retryable: {exc.retryable}")
        print(f"Max retries: {max_retries}")
        print(
            f"Diagnostic saved to: {output_path}"
        )

        return

    finished_at = datetime.now(timezone.utc)

    result = {
        "test": "SC5_PROVIDER_MICRO_RUN",
        "model_slot": model_slot,
        "provider": response.provider,
        "model_id": response.model_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "status": "completed",
        "max_retries": max_retries,
        "usage": response.usage,
        "response_text": response.text,
    }

    output_path = save_result(
        result,
        f"{model_slot}_{provider_name}.json",
    )

    print(
        f"{model_slot}: "
        f"{provider_name} / {model_id}"
    )
    print(
        f"Response saved to: {output_path}"
    )
    print(
        f"Usage: {response.usage}"
    )
    print(
        f"Response: {response.text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated provider connectivity tests."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["model_1", "model_2"],
    )

    args = parser.parse_args()

    run_one(args.model)


if __name__ == "__main__":
    main()
