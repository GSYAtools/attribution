from src.run_experiment import (
    STATUS_COMPLETED,
    STATUS_INVALID_JSON,
    STATUS_SCHEMA_INVALID,
    load_output_validator,
    parse_and_validate,
)


def test_valid_output_is_completed():
    _, validator = load_output_validator()

    response = (
        '{"attribution":"insufficient_basis",'
        '"confidence":50,'
        '"justification":"Test response.",'
        '"evidence_used":["Evidence A."],'
        '"limitations":["Limitation A."],'
        '"action":"abstain"}'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_COMPLETED
    assert parsed is not None
    assert error is None


def test_invalid_json_is_detected():
    _, validator = load_output_validator()

    response = (
        '{"attribution":"supported",'
        '"confidence":80,'
        '"justification":"Broken JSON"'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_INVALID_JSON
    assert parsed is None
    assert error is not None


def test_schema_invalid_output_is_detected():
    _, validator = load_output_validator()

    response = (
        '{"attribution":"invalid_value",'
        '"confidence":80,'
        '"justification":"Test response.",'
        '"evidence_used":["Evidence A."],'
        '"limitations":["Limitation A."],'
        '"action":"abstain"}'
    )

    status, parsed, error = parse_and_validate(
        response,
        validator,
    )

    assert status == STATUS_SCHEMA_INVALID
    assert parsed is not None
    assert error is not None
