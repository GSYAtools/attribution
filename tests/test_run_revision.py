from pathlib import Path

import pytest

from src.run_revision import (
    STATUS_COMPLETED,
    STATUS_INVALID_JSON,
    STATUS_SCHEMA_INVALID,
    load_revision_validator,
    parse_and_validate,
    revision_run_id,
)


def test_revision_run_id_is_deterministic():
    row = {
        "initial_run_id": "RUN_0037",
        "update_id": "PX_1005_MD",
    }

    assert revision_run_id(row) == (
        "RUN_0037__PX_1005_MD"
    )


def test_revision_run_ids_are_distinct_for_two_updates():
    material = {
        "initial_run_id": "RUN_0037",
        "update_id": "PX_1005_MD",
    }

    control = {
        "initial_run_id": "RUN_0037",
        "update_id": "PX_1005_NC",
    }

    assert revision_run_id(material) != (
        revision_run_id(control)
    )


def test_valid_revision_output_is_completed():
    validator = load_revision_validator()

    response = (
        '{"revised_attribution":"partially_supported",'
        '"confidence":65,'
        '"revision_decision":"weaken",'
        '"justification":"The new evidence weakens the '
        'previous assessment.",'
        '"evidence_affected":["New evidence."]}'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_COMPLETED
    assert parsed is not None
    assert error is None


def test_invalid_json_is_detected():
    validator = load_revision_validator()

    response = (
        '{"revised_attribution":"supported",'
        '"confidence":80,'
        '"revision_decision":"maintain"'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_INVALID_JSON
    assert parsed is None
    assert error is not None


def test_schema_invalid_revision_is_detected():
    validator = load_revision_validator()

    response = (
        '{"revised_attribution":"invalid_value",'
        '"confidence":80,'
        '"revision_decision":"maintain",'
        '"justification":"Test.",'
        '"evidence_affected":[]}'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_SCHEMA_INVALID
    assert parsed is not None
    assert error is not None


def test_revision_schema_rejects_additional_properties():
    validator = load_revision_validator()

    response = (
        '{"revised_attribution":"supported",'
        '"confidence":80,'
        '"revision_decision":"maintain",'
        '"justification":"Test.",'
        '"evidence_affected":[],'
        '"hidden_label":"material_defeater"}'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_SCHEMA_INVALID
    assert parsed is not None
    assert error is not None
