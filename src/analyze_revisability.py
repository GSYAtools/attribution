from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from src.utils import ROOT


INPUT = (
    ROOT
    / "analysis"
    / "round2_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "revisability.json"
)

UPDATE_TYPES = (
    "material_defeater",
    "non_material_control",
)

MODELS = (
    "model_1",
    "model_2",
)

CONDITIONS = (
    "C0",
    "C1",
    "C2",
)

REVISION_DECISIONS = (
    "maintain",
    "weaken",
    "withdraw",
)

EXPECTED_ROWS = 48
EXPECTED_PAIRS = 24


def load_rows() -> list[dict]:
    if not INPUT.exists():
        raise RuntimeError(
            f"Required input not found: {INPUT}"
        )

    with INPUT.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(rows)}."
        )

    if not rows:
        raise RuntimeError(
            "Round 2 dataset is empty."
        )

    required = {
        "revision_run_id",
        "initial_run_id",
        "case_id",
        "domain",
        "condition",
        "model_slot",
        "repetition",
        "update_id",
        "update_type",
        "initial_attribution",
        "initial_confidence",
        "initial_action",
        "revised_attribution",
        "confidence",
        "revision_decision",
    }

    missing = (
        required - set(rows[0])
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    seen_revision_runs = set()

    for row in rows:
        revision_run_id = (
            row["revision_run_id"]
        )

        if (
            revision_run_id
            in seen_revision_runs
        ):
            raise RuntimeError(
                "Duplicate revision_run_id: "
                f"{revision_run_id}"
            )

        seen_revision_runs.add(
            revision_run_id
        )

        if (
            row["update_type"]
            not in UPDATE_TYPES
        ):
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected update_type "
                f"{row['update_type']}"
            )

        if (
            row["model_slot"]
            not in MODELS
        ):
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected model_slot "
                f"{row['model_slot']}"
            )

        if (
            row["condition"]
            not in CONDITIONS
        ):
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected condition "
                f"{row['condition']}"
            )

        if (
            row["revision_decision"]
            not in REVISION_DECISIONS
        ):
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected revision_decision "
                f"{row['revision_decision']}"
            )

        try:
            repetition = int(
                row["repetition"]
            )

        except (TypeError, ValueError):
            raise RuntimeError(
                f"{revision_run_id}: "
                f"invalid repetition "
                f"{row['repetition']!r}"
            )

        if repetition not in {1, 2}:
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected repetition "
                f"{repetition}"
            )

        row["repetition"] = repetition

        try:
            row[
                "initial_confidence"
            ] = float(
                row["initial_confidence"]
            )

            row["confidence"] = float(
                row["confidence"]
            )

        except (TypeError, ValueError):
            raise RuntimeError(
                f"{revision_run_id}: "
                "invalid confidence value."
            )

    return rows


def expected_response(
    row: dict,
) -> bool:
    """
    Study-specific directional rule.

    Material defeater:
        weaken or withdraw

    Non-material control:
        maintain

    This rule is specific to the controlled
    Stage 2 design. It is not a universal
    normative rule for trustworthiness
    attribution.
    """

    if (
        row["update_type"]
        == "material_defeater"
    ):
        return (
            row["revision_decision"]
            in {
                "weaken",
                "withdraw",
            }
        )

    if (
        row["update_type"]
        == "non_material_control"
    ):
        return (
            row["revision_decision"]
            == "maintain"
        )

    raise RuntimeError(
        f"Unexpected update_type: "
        f"{row['update_type']}"
    )


def decision_counts(
    rows: list[dict],
) -> dict:
    counts = Counter(
        row["revision_decision"]
        for row in rows
    )

    return {
        decision: counts[decision]
        for decision
        in REVISION_DECISIONS
    }


def summarize_update_type(
    rows: list[dict],
) -> dict:
    if not rows:
        raise RuntimeError(
            "Cannot summarize "
            "an empty group."
        )

    expected_count = sum(
        expected_response(row)
        for row in rows
    )

    return {
        "n": len(rows),
        "decisions": (
            decision_counts(
                rows
            )
        ),
        "expected_direction": {
            "count": expected_count,
            "rate": round(
                expected_count
                / len(rows),
                3,
            ),
        },
    }


def summarize_by_update_type(
    rows: list[dict],
) -> dict:
    result = {}

    for update_type in UPDATE_TYPES:
        subset = [
            row
            for row in rows
            if row["update_type"]
            == update_type
        ]

        if not subset:
            continue

        result[
            update_type
        ] = summarize_update_type(
            subset
        )

    return result


def summarize_by_field(
    rows: list[dict],
    field: str,
) -> dict:
    groups = defaultdict(list)

    for row in rows:
        groups[
            row[field]
        ].append(row)

    return {
        key: summarize_by_update_type(
            group_rows
        )
        for key, group_rows
        in sorted(
            groups.items()
        )
    }


def build_pairs(
    rows: list[dict],
) -> tuple[
    list[dict],
    dict[str, int],
]:
    """
    Pair the material defeater and
    non-material control derived from the
    same initial Stage 1 run.
    """

    groups = defaultdict(list)

    for row in rows:
        groups[
            row["initial_run_id"]
        ].append(row)

    pairs = []
    incomplete = {}

    for initial_run_id, group_rows in (
        sorted(groups.items())
    ):
        if len(group_rows) != 2:
            incomplete[
                initial_run_id
            ] = len(group_rows)
            continue

        by_type = {
            row["update_type"]: row
            for row in group_rows
        }

        if set(by_type) != set(
            UPDATE_TYPES
        ):
            incomplete[
                initial_run_id
            ] = len(group_rows)
            continue

        md = by_type[
            "material_defeater"
        ]

        nc = by_type[
            "non_material_control"
        ]

        identifying_fields = (
            "case_id",
            "condition",
            "model_slot",
            "repetition",
            "initial_attribution",
            "initial_confidence",
            "initial_action",
        )

        for field in identifying_fields:
            if md[field] != nc[field]:
                raise RuntimeError(
                    f"{initial_run_id}: "
                    f"paired updates disagree "
                    f"on {field}: "
                    f"{md[field]!r} vs "
                    f"{nc[field]!r}"
                )

        md_expected = (
            expected_response(md)
        )

        nc_expected = (
            expected_response(nc)
        )

        selective = (
            md_expected
            and nc_expected
        )

        pairs.append(
            {
                "initial_run_id": (
                    initial_run_id
                ),
                "case_id": (
                    md["case_id"]
                ),
                "domain": (
                    md["domain"]
                ),
                "condition": (
                    md["condition"]
                ),
                "model_slot": (
                    md["model_slot"]
                ),
                "repetition": (
                    md["repetition"]
                ),
                "initial_attribution": (
                    md[
                        "initial_attribution"
                    ]
                ),
                "initial_confidence": (
                    md[
                        "initial_confidence"
                    ]
                ),
                "initial_action": (
                    md["initial_action"]
                ),
                "material_defeater": {
                    "revision_run_id": (
                        md[
                            "revision_run_id"
                        ]
                    ),
                    "update_id": (
                        md["update_id"]
                    ),
                    "revised_attribution": (
                        md[
                            "revised_attribution"
                        ]
                    ),
                    "confidence": (
                        md["confidence"]
                    ),
                    "revision_decision": (
                        md[
                            "revision_decision"
                        ]
                    ),
                    "expected_direction": (
                        md_expected
                    ),
                },
                "non_material_control": {
                    "revision_run_id": (
                        nc[
                            "revision_run_id"
                        ]
                    ),
                    "update_id": (
                        nc["update_id"]
                    ),
                    "revised_attribution": (
                        nc[
                            "revised_attribution"
                        ]
                    ),
                    "confidence": (
                        nc["confidence"]
                    ),
                    "revision_decision": (
                        nc[
                            "revision_decision"
                        ]
                    ),
                    "expected_direction": (
                        nc_expected
                    ),
                },
                "selective_revisability": (
                    selective
                ),
            }
        )

    return pairs, incomplete


def summarize_pairs(
    pairs: list[dict],
) -> dict:
    if not pairs:
        raise RuntimeError(
            "Cannot summarize zero pairs."
        )

    selective = sum(
        pair[
            "selective_revisability"
        ]
        for pair in pairs
    )

    return {
        "pairs": len(pairs),
        "both_expected": {
            "count": selective,
            "rate": round(
                selective
                / len(pairs),
                3,
            ),
        },
        "non_selective": (
            len(pairs) - selective
        ),
    }


def pair_summary_by_field(
    pairs: list[dict],
    field: str,
) -> dict:
    groups = defaultdict(list)

    for pair in pairs:
        groups[
            pair[field]
        ].append(pair)

    return {
        key: summarize_pairs(
            group_pairs
        )
        for key, group_pairs
        in sorted(
            groups.items()
        )
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    by_update_type = (
        summarize_by_update_type(
            rows
        )
    )

    by_model = (
        summarize_by_field(
            rows,
            "model_slot",
        )
    )

    by_condition = (
        summarize_by_field(
            rows,
            "condition",
        )
    )

    pairs, incomplete_pairs = (
        build_pairs(
            rows
        )
    )

    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} "
            f"complete MD/NC pairs, "
            f"found {len(pairs)}."
        )

    paired_summary = (
        summarize_pairs(
            pairs
        )
    )

    paired_by_model = (
        pair_summary_by_field(
            pairs,
            "model_slot",
        )
    )

    paired_by_condition = (
        pair_summary_by_field(
            pairs,
            "condition",
        )
    )

    non_selective_pairs = [
        pair
        for pair in pairs
        if not pair[
            "selective_revisability"
        ]
    ]

    result = {
        "design": {
            "revision_runs": len(rows),
            "paired_initial_runs": (
                len(pairs)
            ),
            "source_dataset": (
                "analysis/"
                "round2_dataset.csv"
            ),
            "update_types": list(
                UPDATE_TYPES
            ),
            "interpretation": (
                "Selective revisability is "
                "assessed through paired "
                "responses to two controlled "
                "updates of the same initial "
                "Stage 1 assessment. "
                "A material defeater is "
                "expected to induce weakening "
                "or withdrawal, whereas a "
                "non-material control is "
                "expected to preserve the "
                "prior attribution. These "
                "directional expectations are "
                "specific to the controlled "
                "Stage 2 design and do not "
                "define a universal normative "
                "trustworthiness attribution."
            ),
            "material_defeater_rule": (
                "revision_decision in "
                "{weaken, withdraw}"
            ),
            "non_material_control_rule": (
                "revision_decision == maintain"
            ),
        },
        "by_update_type": (
            by_update_type
        ),
        "by_model": (
            by_model
        ),
        "by_condition": (
            by_condition
        ),
        "paired_selective_revisability": {
            "overall": (
                paired_summary
            ),
            "by_model": (
                paired_by_model
            ),
            "by_condition": (
                paired_by_condition
            ),
        },
        "pairs_without_required_updates": (
            incomplete_pairs
        ),
        "pairs": pairs,
        "non_selective_pairs": (
            non_selective_pairs
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Selective revisability "
        "analysis: OK"
    )

    print()
    print(
        "ROWS:",
        len(rows),
    )

    print()
    print(
        "REVISION DECISION "
        "BY UPDATE TYPE"
    )

    for update_type in UPDATE_TYPES:
        values = by_update_type[
            update_type
        ]

        expected = values[
            "expected_direction"
        ]

        print()
        print(update_type)

        print(
            "n =",
            values["n"],
        )

        print(
            "decisions =",
            values["decisions"],
        )

        print(
            "expected_direction =",
            expected["count"],
            "/",
            values["n"],
            "=",
            expected["rate"],
        )

    print()
    print("BY MODEL")

    for model in MODELS:
        print()
        print(model)

        for update_type in UPDATE_TYPES:
            values = by_model[
                model
            ][update_type]

            expected = values[
                "expected_direction"
            ]

            print(
                update_type,
                values["decisions"],
                "expected=",
                f"{expected['count']}"
                f"/{values['n']}",
            )

    print()
    print("BY CONDITION")

    for condition in CONDITIONS:
        print()
        print(condition)

        for update_type in UPDATE_TYPES:
            values = by_condition[
                condition
            ][update_type]

            expected = values[
                "expected_direction"
            ]

            print(
                update_type,
                values["decisions"],
                "expected=",
                f"{expected['count']}"
                f"/{values['n']}",
            )

    print()
    print(
        "PAIRED SELECTIVE REVISABILITY"
    )

    overall = paired_summary[
        "both_expected"
    ]

    print(
        "Pairs with both expected "
        "responses:",
        overall["count"],
        "/",
        paired_summary["pairs"],
        "=",
        overall["rate"],
    )

    print()
    print(
        "PAIRED BY MODEL"
    )

    for model in MODELS:
        values = paired_by_model[
            model
        ]

        expected = values[
            "both_expected"
        ]

        print(
            model,
            expected["count"],
            "/",
            values["pairs"],
            "=",
            expected["rate"],
        )

    print()
    print(
        "PAIRED BY CONDITION"
    )

    for condition in CONDITIONS:
        values = (
            paired_by_condition[
                condition
            ]
        )

        expected = values[
            "both_expected"
        ]

        print(
            condition,
            expected["count"],
            "/",
            values["pairs"],
            "=",
            expected["rate"],
        )

    print()
    print(
        "NON-SELECTIVE PAIRS"
    )

    for pair in non_selective_pairs:
        print(
            pair["initial_run_id"],
            pair["model_slot"],
            pair["condition"],
            pair[
                "material_defeater"
            ][
                "revision_decision"
            ],
            pair[
                "non_material_control"
            ][
                "revision_decision"
            ],
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
