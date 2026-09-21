from __future__ import annotations

import csv
import json
from collections import defaultdict
from statistics import mean

from src.utils import ROOT


INPUT = (
    ROOT
    / "analysis"
    / "round1_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "consistency.json"
)

EXPECTED_ROWS = 72
EXPECTED_PAIRS = 36

MODELS = (
    "model_1",
    "model_2",
)

CONDITIONS = (
    "C0",
    "C1",
    "C2",
)


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
            "Round 1 dataset is empty."
        )

    required = {
        "run_id",
        "case_id",
        "condition",
        "model_slot",
        "repetition",
        "attribution",
        "confidence",
        "action",
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
        run_id = row["run_id"]

        if run_id in seen_runs:
            raise RuntimeError(
                f"Duplicate run_id: {run_id}"
            )

        seen_runs.add(run_id)

        if row["model_slot"] not in MODELS:
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"model_slot "
                f"{row['model_slot']}"
            )

        if row["condition"] not in CONDITIONS:
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"condition "
                f"{row['condition']}"
            )

        try:
            repetition = int(
                row["repetition"]
            )
        except (TypeError, ValueError):
            raise RuntimeError(
                f"{run_id}: invalid repetition "
                f"{row['repetition']!r}"
            )

        if repetition not in {1, 2}:
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"repetition {repetition}"
            )

        row["repetition"] = repetition

        try:
            confidence = float(
                row["confidence"]
            )
        except (TypeError, ValueError):
            raise RuntimeError(
                f"{run_id}: invalid confidence "
                f"{row['confidence']!r}"
            )

        row["confidence"] = confidence

    return rows


def build_pairs(
    rows: list[dict],
) -> tuple[
    list[dict],
    dict[str, int],
]:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["case_id"],
            row["condition"],
            row["model_slot"],
        )

        groups[key].append(row)

    pairs = []
    incomplete = {}

    for key, group_rows in sorted(
        groups.items()
    ):
        if len(group_rows) != 2:
            incomplete[
                "|".join(key)
            ] = len(group_rows)
            continue

        ordered = sorted(
            group_rows,
            key=lambda row: (
                row["repetition"]
            ),
        )

        left = ordered[0]
        right = ordered[1]

        if (
            left["repetition"] != 1
            or right["repetition"] != 2
        ):
            incomplete[
                "|".join(key)
            ] = len(group_rows)
            continue

        attribution_agreement = (
            left["attribution"]
            == right["attribution"]
        )

        action_agreement = (
            left["action"]
            == right["action"]
        )

        confidence_difference = abs(
            left["confidence"]
            - right["confidence"]
        )

        pairs.append(
            {
                "case_id": left["case_id"],
                "condition": left["condition"],
                "model_slot": (
                    left["model_slot"]
                ),
                "run_1": left["run_id"],
                "run_2": right["run_id"],
                "attribution_1": (
                    left["attribution"]
                ),
                "attribution_2": (
                    right["attribution"]
                ),
                "attribution_agreement": (
                    attribution_agreement
                ),
                "action_1": left["action"],
                "action_2": right["action"],
                "action_agreement": (
                    action_agreement
                ),
                "confidence_1": (
                    left["confidence"]
                ),
                "confidence_2": (
                    right["confidence"]
                ),
                "absolute_confidence_difference": (
                    confidence_difference
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

    attribution_matches = sum(
        pair[
            "attribution_agreement"
        ]
        for pair in pairs
    )

    action_matches = sum(
        pair[
            "action_agreement"
        ]
        for pair in pairs
    )

    confidence_differences = [
        pair[
            "absolute_confidence_difference"
        ]
        for pair in pairs
    ]

    return {
        "pairs": len(pairs),
        "attribution_exact_agreement": {
            "count": attribution_matches,
            "rate": round(
                attribution_matches
                / len(pairs),
                3,
            ),
        },
        "action_exact_agreement": {
            "count": action_matches,
            "rate": round(
                action_matches
                / len(pairs),
                3,
            ),
        },
        "confidence": {
            "mean_absolute_difference": (
                round(
                    mean(
                        confidence_differences
                    ),
                    3,
                )
            ),
            "max_absolute_difference": (
                max(
                    confidence_differences
                )
            ),
        },
    }


def group_pairs(
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
            group_pairs_
        )
        for key, group_pairs_
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

    pairs, incomplete = (
        build_pairs(
            rows
        )
    )

    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} "
            f"complete repetition pairs, "
            f"found {len(pairs)}."
        )

    overall = summarize_pairs(
        pairs
    )

    by_model = group_pairs(
        pairs,
        "model_slot",
    )

    by_condition = group_pairs(
        pairs,
        "condition",
    )

    disagreements = [
        pair
        for pair in pairs
        if not pair[
            "attribution_agreement"
        ]
    ]

    result = {
        "design": {
            "runs": len(rows),
            "pairs": len(pairs),
            "repetitions_per_pair": 2,
            "pairing_key": [
                "case_id",
                "condition",
                "model_slot",
            ],
            "interpretation": (
                "Consistency is assessed as "
                "stability across independent "
                "repetitions preserving case, "
                "evidential condition, and "
                "evaluator-model realisation. "
                "Exact textual identity is not "
                "required."
            ),
        },
        "pairs_without_exactly_two_repetitions": (
            incomplete
        ),
        "overall": overall,
        "by_model": by_model,
        "by_condition": by_condition,
        "attribution_disagreements": (
            disagreements
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
        "Consistency analysis: OK"
    )

    print()
    print(
        "PAIRS:",
        len(pairs),
    )

    print(
        "Pairs without exactly "
        "2 repetitions:",
        incomplete,
    )

    print()
    print("OVERALL")

    print(
        "Attribution exact agreement:",
        overall[
            "attribution_exact_agreement"
        ]["count"],
        "/",
        overall["pairs"],
        "=",
        overall[
            "attribution_exact_agreement"
        ]["rate"],
    )

    print(
        "Action exact agreement:",
        overall[
            "action_exact_agreement"
        ]["count"],
        "/",
        overall["pairs"],
        "=",
        overall[
            "action_exact_agreement"
        ]["rate"],
    )

    print(
        "Mean absolute confidence "
        "difference:",
        overall[
            "confidence"
        ][
            "mean_absolute_difference"
        ],
    )

    print()
    print("BY MODEL")

    for model in MODELS:
        values = by_model[model]

        print(
            model,
            "attribution_agreement=",
            values[
                "attribution_exact_agreement"
            ]["rate"],
            "action_agreement=",
            values[
                "action_exact_agreement"
            ]["rate"],
            "mean_abs_conf_diff=",
            values[
                "confidence"
            ][
                "mean_absolute_difference"
            ],
        )

    print()
    print("BY CONDITION")

    for condition in CONDITIONS:
        values = by_condition[
            condition
        ]

        print(
            condition,
            "attribution_agreement=",
            values[
                "attribution_exact_agreement"
            ]["rate"],
            "action_agreement=",
            values[
                "action_exact_agreement"
            ]["rate"],
            "mean_abs_conf_diff=",
            values[
                "confidence"
            ][
                "mean_absolute_difference"
            ],
        )

    print()
    print(
        "ATTRIBUTION DISAGREEMENTS"
    )

    for pair in disagreements:
        print(
            pair["case_id"],
            pair["condition"],
            pair["model_slot"],
            pair["attribution_1"],
            pair["attribution_2"],
            pair["confidence_1"],
            pair["confidence_2"],
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
