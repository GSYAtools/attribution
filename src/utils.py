import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Project root: attribution/
ROOT = Path(__file__).resolve().parents[1]

# Load local environment variables from attribution/.env
load_dotenv(ROOT / ".env")


def project_path(relative_path: str) -> Path:
    """Return an absolute path relative to the project root."""
    return ROOT / relative_path


def load_json(relative_path: str) -> dict[str, Any]:
    """Load a JSON file from the project."""
    path = project_path(relative_path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(relative_path: str) -> dict[str, Any]:
    """Load a YAML file from the project."""
    path = project_path(relative_path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_text(relative_path: str) -> str:
    """Load a UTF-8 text file from the project."""
    path = project_path(relative_path)

    with path.open("r", encoding="utf-8") as file:
        return file.read()


def get_env(name: str) -> str:
    """Return a required environment variable."""
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not configured."
        )

    return value
