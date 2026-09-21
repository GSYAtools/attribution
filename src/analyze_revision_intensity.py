from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from statistics import mean

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
    / "revision_intensity.json"
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

EXPECTED_ROWS = 48


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

    seen_runs = set()

    for row in rows:
        revision_run_id = (
            row["revision_run_id"]
        )

        if revision_run_id in seen_runs:
            raise RuntimeError(
                "Duplicate revision_run_id: "
                f"{revision_run_id}"
            )

        seen_runs.add(
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

        try:
            repetition = int(
                row["repetition"]
            )

            initial_confidence = float(
                row["initial_confidence"]
            )

            revised_confidence = float(
                row["confidence"]
            )

        except (TypeError, ValueError):
            raise RuntimeError(
                f"{revision_run_id}: "
                "invalid numeric value."
            )

        if repetition not in {1, 2}:
            raise RuntimeError(
                f"{revision_run_id}: "
                f"unexpected repetition "
                f"{repetition}"
            )

        row["repetition"] = repetition

        row[
            "initial_confidence"
        ] = initial_confidence

        row[
            "confidence"
        ] = revised_confidence

        row[
            "confidence_delta"
        ] = (
            revised_confidence
            - initial_confidence
        )

        row[
            "attribution_changed"
        ] = (
            row["initial_attribution"]
            != row["revised_attribution"]
        )

    return rows


def number_key(
    value: float,
) -> str:
    """
    Produce readable JSON keys for confidence
    delta distributions.
    """

    if value.is_integer():
        return str(
            int(value)
        )

    return str(
        round(value, 3)
    )


def summarize(
    rows: list[dict],
) -> dict:
    if not rows:
        raise RuntimeError(
            "Cannot summarize an empty group."
        )

    changed = sum(
        row["attribution_changed"]
        for row in rows
    )

    deltas = [
        row["confidence_delta"]
        for row in rows
    ]

    delta_counts = Counter(
        deltas
    )

    return {
        "n": len(rows),
        "attribution_changed": {
            "count": changed,
            "rate": round(
                changed / len(rows),
                3,
            ),
        },
        "confidence_delta": {
            "mean": round(
                mean(deltas),
                3,
            ),
            "min": min(deltas),
            "max": max(deltas),
            "distribution": {
                number_key(key): value
                for key, value
                in sorted(
                    delta_counts.items()
                )
            },
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

        result[
            update_type
        ] = summarize(
            subset
        )

    return result


def summarize_case_model(
    rows: list[dict],
) -> dict:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["case_id"],
            row["model_slot"],
            row["update_type"],
        )

        groups[key].append(row)

    result = {}

    for (
        case_id,
        model,
        update_type,
    ), group_rows in sorted(
        groups.items()
    ):
        result.setdefault(
            case_id,
            {},
        ).setdefault(
            model,
            {},
        )[update_type] = {
            **summarize(
                group_rows
            ),
            "confidence_deltas": [
                row[
                    "confidence_delta"
                ]
                for row in group_rows
            ],
        }

    return result


def summarize_by_model(
    rows: list[dict],
) -> dict:
    result = {}

    for model in MODELS:
        subset = [
            row
            for row in rows
            if row["model_slot"]
            == model
        ]

        result[model] = (
            summarize_by_update_type(
                subset
            )
        )

    return result


def summarize_by_condition(
    rows: list[dict],
) -> dict:
    result = {}

    for condition in CONDITIONS:
        subset = [
            row
            for row in rows
            if row["condition"]
            == condition
        ]

        result[condition] = (
            summarize_by_update_type(
                subset
            )
        )

    return result


def non_selective_material_defeaters(
    rows: list[dict],
) -> list[dict]:
    """
    Return material-defeater runs where the
    revision decision was maintain.

    These are diagnostic cases for selective
    revisability. Confidence change alone is
    not treated as evidence that the expected
    revision occurred.
    """

    result = []

    for row in rows:
        if (
            row["update_type"]
            != "material_defeater"
        ):
            continue

        if (
            row["revision_decision"]
            != "maintain"
        ):
            continue

        result.append(
            {
                "initial_run_id": (
                    row["initial_run_id"]
                ),
                "revision_run_id": (
                    row["revision_run_id"]
                ),
                "case_id": (
                    row["case_id"]
                ),
                "condition": (
                    row["condition"]
                ),
                "model_slot": (
                    row["model_slot"]
                ),
                "repetition": (
                    row["repetition"]
                ),
                "initial_attribution": (
                    row[
                        "initial_attribution"
                    ]
                ),
                "revised_attribution": (
                    row[
                        "revised_attribution"
                    ]
                ),
                "initial_confidence": (
                    row[
                        "initial_confidence"
                    ]
                ),
                "revised_confidence": (
                    row["confidence"]
                ),
                "confidence_delta": (
                    row[
                        "confidence_delta"
                    ]
                ),
                "revision_decision": (
                    row[
                        "revision_decision"
                    ]
                ),
            }
        )

    return result


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

    by_case_and_model = (
        summarize_case_model(
            rows
        )
    )

    by_model = (
        summarize_by_model(
            rows
        )
    )

    by_condition = (
        summarize_by_condition(
            rows
        )
    )

    non_selective = (
        non_selective_material_defeaters(
            rows
        )
    )

    result = {
        "design": {
            "revision_runs": len(rows),
            "source_dataset": (
                "analysis/"
                "round2_dataset.csv"
            ),
            "interpretation": (
                "Revision intensity is a "
                "secondary diagnostic of "
                "selective revisability. "
                "It characterises whether "
                "the categorical attribution "
                "changed and how confidence "
                "changed after an evidential "
                "update. It does not replace "
                "the paired selective-"
                "revisability analysis."
            ),
            "confidence_interpretation": (
                "Confidence represents "
                "commitment to the resulting "
                "attribution. A material "
                "defeater therefore need not "
                "produce a negative confidence "
                "delta: the evaluator may "
                "become more confident in a "
                "weakened attribution."
            ),
        },
        "by_update_type": (
            by_update_type
        ),
        "by_case_and_model": (
            by_case_and_model
        ),
        "by_model": (
            by_model
        ),
        "by_condition": (
            by_condition
        ),
        "non_selective_material_defeaters": (
            non_selective
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
        "Revision intensity analysis: OK"
    )

    print()
    print("BY UPDATE TYPE")

    for update_type in UPDATE_TYPES:
        values = by_update_type[
            update_type
        ]

        changed = values[
            "attribution_changed"
        ]

        delta = values[
            "confidence_delta"
        ]

        print()
        print(update_type)

        print(
            "attribution_changed =",
            changed["count"],
            "/",
            values["n"],
        )

        print(
            "mean_confidence_delta =",
            delta["mean"],
        )

        print(
            "confidence_delta_distribution =",
            delta["distribution"],
        )

    print()
    print("BY CASE AND MODEL")

    for case_id in sorted(
        by_case_and_model
    ):
        for model in MODELS:
            model_values = (
                by_case_and_model[
                    case_id
                ].get(
                    model,
                    {},
                )
            )

            for update_type in UPDATE_TYPES:
                if (
                    update_type
                    not in model_values
                ):
                    continue

                values = model_values[
                    update_type
                ]

                changed = values[
                    "attribution_changed"
                ]

                delta = values[
                    "confidence_delta"
                ]

                print(
                    case_id,
                    model,
                    update_type,
                    "attr_changed=",
                    f"{changed['count']}"
                    f"/{values['n']}",
                    "mean_delta=",
                    delta["mean"],
                    "deltas=",
                    values[
                        "confidence_deltas"
                    ],
                )

    print()
    print(
        "NON-SELECTIVE "
        "MATERIAL DEFEATERS"
    )

    for row in non_selective:
        print(
            row["initial_run_id"],
            row["model_slot"],
            row["condition"],
            row[
                "initial_attribution"
            ],
            "->",
            row[
                "revised_attribution"
            ],
            "confidence",
            row[
                "initial_confidence"
            ],
            "->",
            row[
                "revised_confidence"
            ],
            "delta=",
            row[
                "confidence_delta"
            ],
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
