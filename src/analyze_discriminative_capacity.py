from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from statistics import mean, median

from src.utils import ROOT


INPUT = (
    ROOT
    / "analysis"
    / "round1_dataset.csv"
)

CASE_REGISTRY = (
    ROOT
    / "data"
    / "case_registry.json"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "discriminative_capacity.json"
)

ATTRIBUTIONS = (
    "supported",
    "partially_supported",
    "insufficient_basis",
    "not_supported",
)

ACTIONS = (
    "affirm",
    "abstain",
    "escalate",
)

CONDITIONS = (
    "C0",
    "C1",
    "C2",
)

MODELS = (
    "model_1",
    "model_2",
)

EXPECTED_ROWS = 72


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
        rows = list(csv.DictReader(file))

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
        "domain",
        "condition",
        "model_slot",
        "repetition",
        "attribution",
        "confidence",
        "action",
    }

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    seen_runs = set()

    for row in rows:
        run_id = row["run_id"]

        if run_id in seen_runs:
            raise RuntimeError(
                f"Duplicate run_id: {run_id}"
            )

        seen_runs.add(run_id)

        if row["condition"] not in CONDITIONS:
            raise RuntimeError(
                f"Unexpected condition: "
                f"{row['condition']}"
            )

        if row["model_slot"] not in MODELS:
            raise RuntimeError(
                f"Unexpected model_slot: "
                f"{row['model_slot']}"
            )

        if row["attribution"] not in ATTRIBUTIONS:
            raise RuntimeError(
                f"Unexpected attribution: "
                f"{row['attribution']}"
            )

        if row["action"] not in ACTIONS:
            raise RuntimeError(
                f"Unexpected action: "
                f"{row['action']}"
            )

        try:
            row["confidence"] = float(
                row["confidence"]
            )
        except (TypeError, ValueError):
            raise RuntimeError(
                f"{run_id}: invalid confidence "
                f"value {row['confidence']!r}"
            )

    return rows


def load_case_registry() -> dict[str, dict]:
    if not CASE_REGISTRY.exists():
        raise RuntimeError(
            f"Required case registry not found: "
            f"{CASE_REGISTRY}"
        )

    data = json.loads(
        CASE_REGISTRY.read_text(
            encoding="utf-8"
        )
    )

    if "cases" not in data:
        raise RuntimeError(
            "case_registry.json does not "
            "contain a 'cases' collection."
        )

    registry = {}

    for case in data["cases"]:
        case_id = case.get("case_id")

        evidential_state = case.get(
            "expected_evidential_state"
        )

        if not case_id:
            raise RuntimeError(
                "Case without case_id in registry."
            )

        if not evidential_state:
            raise RuntimeError(
                f"{case_id}: missing "
                f"expected_evidential_state."
            )

        if case_id in registry:
            raise RuntimeError(
                f"Duplicate case_id: {case_id}"
            )

        registry[case_id] = case

    return registry


def enrich_rows(
    rows: list[dict],
    registry: dict[str, dict],
) -> None:
    for row in rows:
        case_id = row["case_id"]

        if case_id not in registry:
            raise RuntimeError(
                f"Unknown case_id: {case_id}"
            )

        row[
            "expected_evidential_state"
        ] = registry[
            case_id
        ][
            "expected_evidential_state"
        ]


def complete_counter(
    values,
    categories,
) -> dict[str, int]:
    counts = Counter(values)

    return {
        category: counts[category]
        for category in categories
    }


def proportions(
    counts: dict[str, int],
) -> dict[str, float]:
    total = sum(counts.values())

    if total == 0:
        return {
            key: 0.0
            for key in counts
        }

    return {
        key: round(
            value / total,
            3,
        )
        for key, value in counts.items()
    }


def summarize(
    rows: list[dict],
) -> dict:
    if not rows:
        raise RuntimeError(
            "Cannot summarize an empty group."
        )

    attribution_counts = (
        complete_counter(
            (
                row["attribution"]
                for row in rows
            ),
            ATTRIBUTIONS,
        )
    )

    action_counts = (
        complete_counter(
            (
                row["action"]
                for row in rows
            ),
            ACTIONS,
        )
    )

    confidence = [
        row["confidence"]
        for row in rows
    ]

    return {
        "n": len(rows),
        "case_ids": sorted({
            row["case_id"]
            for row in rows
        }),
        "attribution": {
            "counts": attribution_counts,
            "proportions": proportions(
                attribution_counts
            ),
            "categories_observed": sum(
                value > 0
                for value
                in attribution_counts.values()
            ),
        },
        "action": {
            "counts": action_counts,
            "proportions": proportions(
                action_counts
            ),
        },
        "confidence": {
            "mean": round(
                mean(confidence),
                3,
            ),
            "median": median(
                confidence
            ),
            "min": min(confidence),
            "max": max(confidence),
        },
    }


def group_summary(
    rows: list[dict],
    field: str,
) -> dict:
    groups = defaultdict(list)

    for row in rows:
        groups[
            row[field]
        ].append(row)

    return {
        key: summarize(
            group_rows
        )
        for key, group_rows
        in sorted(groups.items())
    }


def state_by_condition(
    rows: list[dict],
) -> dict:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row[
                "expected_evidential_state"
            ],
            row["condition"],
        )

        groups[key].append(row)

    result = {}

    for (
        state,
        condition,
    ), group_rows in sorted(
        groups.items()
    ):
        result.setdefault(
            state,
            {},
        )[condition] = summarize(
            group_rows
        )

    return result


def state_by_model(
    rows: list[dict],
) -> dict:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row[
                "expected_evidential_state"
            ],
            row["model_slot"],
        )

        groups[key].append(row)

    result = {}

    for (
        state,
        model,
    ), group_rows in sorted(
        groups.items()
    ):
        result.setdefault(
            state,
            {},
        )[model] = summarize(
            group_rows
        )

    return result


def attribution_profiles(
    rows: list[dict],
) -> dict:
    """
    Return the attribution distribution associated
    with each pre-specified evidential state.

    These profiles are descriptive. They are not
    treated as gold-standard attribution labels.
    """

    groups = defaultdict(list)

    for row in rows:
        groups[
            row[
                "expected_evidential_state"
            ]
        ].append(row)

    profiles = {}

    for state, group_rows in sorted(
        groups.items()
    ):
        counts = complete_counter(
            (
                row["attribution"]
                for row in group_rows
            ),
            ATTRIBUTIONS,
        )

        profiles[state] = {
            "n": len(group_rows),
            "counts": counts,
            "proportions": proportions(
                counts
            ),
        }

    return profiles


def pairwise_profile_distance(
    profiles: dict,
) -> dict:
    """
    Compute total variation distance between
    attribution profiles of evidential states.

    TV distance is used descriptively:
        0 = identical distributions
        1 = non-overlapping distributions

    It is NOT an accuracy measure and does not
    imply that either profile is normatively
    correct.
    """

    states = sorted(
        profiles
    )

    result = {}

    for index, left in enumerate(states):
        for right in states[
            index + 1:
        ]:
            p = profiles[
                left
            ]["proportions"]

            q = profiles[
                right
            ]["proportions"]

            distance = (
                0.5
                * sum(
                    abs(
                        p[category]
                        - q[category]
                    )
                    for category
                    in ATTRIBUTIONS
                )
            )

            key = (
                f"{left}__vs__{right}"
            )

            result[key] = round(
                distance,
                3,
            )

    return result


def print_summary(
    values: dict,
    indent: str = "  ",
) -> None:
    print(
        f"{indent}n =",
        values["n"],
    )

    print(
        f"{indent}cases =",
        ", ".join(
            values["case_ids"]
        ),
    )

    print(
        f"{indent}attribution =",
        values[
            "attribution"
        ]["counts"],
    )

    print(
        f"{indent}attribution proportions =",
        values[
            "attribution"
        ]["proportions"],
    )

    print(
        f"{indent}action =",
        values[
            "action"
        ]["counts"],
    )

    print(
        f"{indent}confidence mean =",
        values[
            "confidence"
        ]["mean"],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    registry = load_case_registry()

    enrich_rows(
        rows,
        registry,
    )

    by_state = group_summary(
        rows,
        "expected_evidential_state",
    )

    by_condition = group_summary(
        rows,
        "condition",
    )

    by_model = group_summary(
        rows,
        "model_slot",
    )

    by_state_condition = (
        state_by_condition(
            rows
        )
    )

    by_state_model = (
        state_by_model(
            rows
        )
    )

    profiles = attribution_profiles(
        rows
    )

    profile_distances = (
        pairwise_profile_distance(
            profiles
        )
    )

    result = {
        "design": {
            "runs": len(rows),
            "interpretation": (
                "Discriminative capacity is "
                "examined descriptively through "
                "the relation between "
                "pre-specified evidential states "
                "and resulting attributional "
                "profiles. No state is assigned "
                "a normatively correct "
                "attribution."
            ),
        },
        "overall": summarize(
            rows
        ),
        "by_pre_specified_evidential_state": (
            by_state
        ),
        "by_evidential_condition": (
            by_condition
        ),
        "by_evaluator_model": (
            by_model
        ),
        "by_state_and_condition": (
            by_state_condition
        ),
        "by_state_and_model": (
            by_state_model
        ),
        "attribution_profiles": (
            profiles
        ),
        "pairwise_attribution_profile_tv_distance": (
            profile_distances
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
        "Discriminative capacity analysis: OK"
    )

    print()
    print(
        "BY PRE-SPECIFIED EVIDENTIAL STATE"
    )

    for state, values in (
        by_state.items()
    ):
        print()
        print(state)
        print_summary(
            values
        )

    print()
    print(
        "PAIRWISE ATTRIBUTION PROFILE "
        "DISTANCES"
    )

    for comparison, distance in (
        profile_distances.items()
    ):
        print(
            comparison,
            "TV=",
            distance,
        )

    print()
    print("BY EVIDENTIAL CONDITION")

    condition_labels = {
        "C0": "E0",
        "C1": "E1",
        "C2": "E2",
    }

    for condition in CONDITIONS:
        values = by_condition[
            condition
        ]

        print()
        print(
            condition_labels[
                condition
            ],
            f"({condition})",
        )

        print_summary(
            values
        )

    print()
    print("BY EVALUATOR MODEL")

    for model in MODELS:
        print()
        print(model)

        print_summary(
            by_model[model]
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
