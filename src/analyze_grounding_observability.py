from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from statistics import mean, median

from src.utils import ROOT


INPUT = (
    ROOT
    / "analysis"
    / "human_evaluation"
    / "blind_scores.csv"
)

CASE_REGISTRY = (
    ROOT
    / "data"
    / "case_registry.json"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "human_evaluation"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "grounding_observability.json"
)

OUTPUT_ITEMS = (
    OUTPUT_DIR
    / "grounding_observability_items.csv"
)

EVALUATORS = ("A", "B", "C")
MODELS = ("model_1", "model_2")
EXPECTED_ITEMS = 72


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

    expected_total = (
        EXPECTED_ITEMS * len(EVALUATORS)
    )

    if len(rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} human "
            f"observations, found {len(rows)}."
        )

    if not rows:
        raise RuntimeError(
            "Human evaluation file is empty."
        )

    required = {
        "blind_id",
        "run_id",
        "case_id",
        "domain",
        "condition",
        "model_slot",
        "repetition",
        "evaluator",
        "grounding_score",
        "limitation_recognition",
        "unsupported_evidence",
    }

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    seen = set()

    for row in rows:
        key = (
            row["blind_id"],
            row["evaluator"],
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate human observation: {key}"
            )

        seen.add(key)

        if row["evaluator"] not in EVALUATORS:
            raise RuntimeError(
                f"Unexpected evaluator: "
                f"{row['evaluator']}"
            )

        if row["model_slot"] not in MODELS:
            raise RuntimeError(
                f"Unexpected model_slot: "
                f"{row['model_slot']}"
            )

        try:
            grounding = int(
                row["grounding_score"]
            )
            limitation = int(
                row["limitation_recognition"]
            )
            unsupported = int(
                row["unsupported_evidence"]
            )

        except (TypeError, ValueError):
            raise RuntimeError(
                f"Invalid human score for "
                f"{row['blind_id']} / "
                f"{row['evaluator']}"
            )

        if grounding not in {
            0, 1, 2, 3, 4
        }:
            raise RuntimeError(
                f"Invalid grounding score: "
                f"{grounding}"
            )

        if limitation not in {0, 1}:
            raise RuntimeError(
                f"Invalid limitation score: "
                f"{limitation}"
            )

        if unsupported not in {0, 1}:
            raise RuntimeError(
                f"Invalid unsupported-evidence "
                f"score: {unsupported}"
            )

        row["grounding_score"] = grounding

        row[
            "limitation_recognition"
        ] = limitation

        row[
            "unsupported_evidence"
        ] = unsupported

    by_evaluator = Counter(
        row["evaluator"]
        for row in rows
    )

    for evaluator in EVALUATORS:
        if (
            by_evaluator[evaluator]
            != EXPECTED_ITEMS
        ):
            raise RuntimeError(
                f"Expected {EXPECTED_ITEMS} rows "
                f"for evaluator {evaluator}, "
                f"found "
                f"{by_evaluator[evaluator]}"
            )

    blind_ids = {
        evaluator: {
            row["blind_id"]
            for row in rows
            if row["evaluator"]
            == evaluator
        }
        for evaluator in EVALUATORS
    }

    reference = blind_ids["A"]

    if len(reference) != EXPECTED_ITEMS:
        raise RuntimeError(
            "Evaluator A does not contain "
            "72 unique blind IDs."
        )

    for evaluator in ("B", "C"):
        if blind_ids[evaluator] != reference:
            raise RuntimeError(
                f"Blind IDs differ for evaluator "
                f"{evaluator}."
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
        case_id = case.get(
            "case_id"
        )

        evidential_state = case.get(
            "expected_evidential_state"
        )

        if not case_id:
            raise RuntimeError(
                "Case without case_id in "
                "case_registry.json."
            )

        if not evidential_state:
            raise RuntimeError(
                f"{case_id}: missing "
                f"expected_evidential_state."
            )

        if case_id in registry:
            raise RuntimeError(
                f"Duplicate case_id in registry: "
                f"{case_id}"
            )

        registry[case_id] = case

    return registry


def enrich_with_evidential_state(
    rows: list[dict],
    registry: dict[str, dict],
) -> None:
    for row in rows:
        case_id = row["case_id"]

        if case_id not in registry:
            raise RuntimeError(
                f"Human observation references "
                f"unknown case: {case_id}"
            )

        row[
            "expected_evidential_state"
        ] = registry[
            case_id
        ][
            "expected_evidential_state"
        ]


def distribution(
    values: list[int],
) -> dict[str, int]:
    counts = Counter(values)

    return {
        str(value): counts[value]
        for value in sorted(counts)
    }


def grounding_summary(
    values: list[int],
) -> dict:
    well_grounded = sum(
        value >= 3
        for value in values
    )

    return {
        "n": len(values),
        "mean": round(
            mean(values),
            3,
        ),
        "median": median(values),
        "distribution": distribution(
            values
        ),
        "well_or_critically_grounded": (
            well_grounded
        ),
        "well_or_critically_grounded_rate": (
            round(
                well_grounded
                / len(values),
                3,
            )
        ),
    }


def binary_summary(
    values: list[int],
) -> dict:
    counts = Counter(values)

    return {
        "n": len(values),
        "0": counts[0],
        "1": counts[1],
        "positive_rate": round(
            counts[1] / len(values),
            3,
        ),
    }


def observation_summary(
    rows: list[dict],
) -> dict:
    grounding = [
        row["grounding_score"]
        for row in rows
    ]

    limitation = [
        row["limitation_recognition"]
        for row in rows
    ]

    unsupported = [
        row["unsupported_evidence"]
        for row in rows
    ]

    return {
        "human_observations": len(rows),
        "grounding": grounding_summary(
            grounding
        ),
        "limitation_recognition": (
            binary_summary(
                limitation
            )
        ),
        "unsupported_evidence": (
            binary_summary(
                unsupported
            )
        ),
    }


def per_evaluator_summary(
    rows: list[dict],
) -> dict:
    result = {}

    for evaluator in EVALUATORS:
        subset = [
            row
            for row in rows
            if row["evaluator"]
            == evaluator
        ]

        result[evaluator] = (
            observation_summary(
                subset
            )
        )

    return result


def convergence_summary(
    rows: list[dict],
) -> dict:
    """
    Describe convergence among the three human
    assessors without constructing a consensus
    ground-truth label.
    """

    by_item = defaultdict(list)

    for row in rows:
        by_item[
            row["blind_id"]
        ].append(row)

    grounding_patterns = Counter()
    grounding_ranges = Counter()
    grounding_support = Counter()

    lr_unanimous = 0
    ue_unanimous = 0

    lr_patterns = Counter()
    ue_patterns = Counter()

    for blind_id, observations in (
        by_item.items()
    ):
        if len(observations) != 3:
            raise RuntimeError(
                f"{blind_id}: expected 3 human "
                f"observations, found "
                f"{len(observations)}"
            )

        observations = sorted(
            observations,
            key=lambda row: row["evaluator"],
        )

        grounding = [
            row["grounding_score"]
            for row in observations
        ]

        limitation = [
            row["limitation_recognition"]
            for row in observations
        ]

        unsupported = [
            row["unsupported_evidence"]
            for row in observations
        ]

        counts = Counter(
            grounding
        )

        if len(counts) == 1:
            grounding_patterns[
                "unanimous"
            ] += 1

        elif max(
            counts.values()
        ) == 2:
            grounding_patterns[
                "two_same_one_different"
            ] += 1

        else:
            grounding_patterns[
                "all_different"
            ] += 1

        grounding_ranges[
            max(grounding)
            - min(grounding)
        ] += 1

        grounding_support[
            sum(
                value >= 3
                for value in grounding
            )
        ] += 1

        if len(set(limitation)) == 1:
            lr_unanimous += 1

        if len(set(unsupported)) == 1:
            ue_unanimous += 1

        lr_patterns[
            str(sum(limitation))
        ] += 1

        ue_patterns[
            str(sum(unsupported))
        ] += 1

    items = len(by_item)

    return {
        "items": items,
        "grounding_convergence": {
            "unanimous": (
                grounding_patterns[
                    "unanimous"
                ]
            ),
            "two_same_one_different": (
                grounding_patterns[
                    "two_same_one_different"
                ]
            ),
            "all_different": (
                grounding_patterns[
                    "all_different"
                ]
            ),
        },
        "grounding_range": {
            str(key): value
            for key, value
            in sorted(
                grounding_ranges.items()
            )
        },
        "number_of_assessors_rating_grounding_ge_3": {
            str(key): value
            for key, value
            in sorted(
                grounding_support.items()
            )
        },
        "limitation_recognition": {
            "unanimous_items": (
                lr_unanimous
            ),
            "unanimous_rate": round(
                lr_unanimous / items,
                3,
            ),
            "number_of_positive_assessors": {
                key: value
                for key, value
                in sorted(
                    lr_patterns.items()
                )
            },
        },
        "unsupported_evidence": {
            "unanimous_items": (
                ue_unanimous
            ),
            "unanimous_rate": round(
                ue_unanimous / items,
                3,
            ),
            "number_of_positive_assessors": {
                key: value
                for key, value
                in sorted(
                    ue_patterns.items()
                )
            },
        },
    }


def item_observability(
    rows: list[dict],
) -> tuple[list[dict], dict]:
    by_item = defaultdict(list)

    for row in rows:
        by_item[
            row["blind_id"]
        ].append(row)

    item_rows = []

    for blind_id in sorted(by_item):
        observations = by_item[
            blind_id
        ]

        if len(observations) != 3:
            raise RuntimeError(
                f"{blind_id}: expected 3 human "
                f"observations, found "
                f"{len(observations)}"
            )

        observations = sorted(
            observations,
            key=lambda row: row["evaluator"],
        )

        first = observations[0]

        grounding = [
            row["grounding_score"]
            for row in observations
        ]

        limitation = [
            row["limitation_recognition"]
            for row in observations
        ]

        unsupported = [
            row["unsupported_evidence"]
            for row in observations
        ]

        grounding_counts = Counter(
            grounding
        )

        if len(grounding_counts) == 1:
            grounding_pattern = (
                "unanimous"
            )

        elif max(
            grounding_counts.values()
        ) == 2:
            grounding_pattern = (
                "two_same_one_different"
            )

        else:
            grounding_pattern = (
                "all_different"
            )

        item_rows.append(
            {
                "blind_id": blind_id,
                "run_id": first["run_id"],
                "case_id": first["case_id"],
                "domain": first["domain"],
                "expected_evidential_state": (
                    first[
                        "expected_evidential_state"
                    ]
                ),
                "condition": first[
                    "condition"
                ],
                "model_slot": first[
                    "model_slot"
                ],
                "repetition": first[
                    "repetition"
                ],
                "grounding_A": grounding[0],
                "grounding_B": grounding[1],
                "grounding_C": grounding[2],
                "grounding_pattern": (
                    grounding_pattern
                ),
                "grounding_range": (
                    max(grounding)
                    - min(grounding)
                ),
                "n_grounding_ge_3": sum(
                    value >= 3
                    for value in grounding
                ),
                "limitation_A": limitation[0],
                "limitation_B": limitation[1],
                "limitation_C": limitation[2],
                "limitation_unanimous": int(
                    len(set(limitation)) == 1
                ),
                "unsupported_A": unsupported[0],
                "unsupported_B": unsupported[1],
                "unsupported_C": unsupported[2],
                "unsupported_unanimous": int(
                    len(set(unsupported)) == 1
                ),
            }
        )

    return (
        item_rows,
        convergence_summary(
            rows
        ),
    )


def condition_observability(
    rows: list[dict],
) -> dict:
    """
    C0, C1 and C2 correspond to E0, E1 and E2
    in the manuscript.

    No monotonic improvement is assumed.
    """

    condition_labels = {
        "C0": "E0",
        "C1": "E1",
        "C2": "E2",
    }

    result = {}

    for condition in (
        "C0",
        "C1",
        "C2",
    ):
        subset = [
            row
            for row in rows
            if row["condition"]
            == condition
        ]

        expected = (
            24 * len(EVALUATORS)
        )

        if len(subset) != expected:
            raise RuntimeError(
                f"{condition}: expected "
                f"{expected} human observations, "
                f"found {len(subset)}"
            )

        result[
            condition_labels[condition]
        ] = {
            "source_condition": condition,
            "items": 24,
            **observation_summary(
                subset
            ),
            "by_evaluator": (
                per_evaluator_summary(
                    subset
                )
            ),
        }

    return result


def evidential_state_observability(
    rows: list[dict],
) -> dict:
    """
    Describe human observations by the
    pre-specified evidential state of each
    synthetic vignette.

    Evidential state is a fixed property of
    the case, not a normative ground-truth
    attribution.
    """

    groups = defaultdict(list)

    for row in rows:
        groups[
            row[
                "expected_evidential_state"
            ]
        ].append(row)

    result = {}

    for state in sorted(groups):
        subset = groups[state]

        blind_ids = {
            row["blind_id"]
            for row in subset
        }

        case_ids = sorted({
            row["case_id"]
            for row in subset
        })

        result[state] = {
            "case_ids": case_ids,
            "items": len(blind_ids),
            **observation_summary(
                subset
            ),
            "by_evaluator": (
                per_evaluator_summary(
                    subset
                )
            ),
        }

    return result


def model_observability(
    rows: list[dict],
) -> dict:
    """
    Describe external human observations
    separately for the two evaluator-model
    realisations W1 and W2.

    The comparison is descriptive and does
    not constitute a model ranking.
    """

    result = {}

    expected_items_per_model = (
        EXPECTED_ITEMS // len(MODELS)
    )

    for model in MODELS:
        subset = [
            row
            for row in rows
            if row["model_slot"] == model
        ]

        blind_ids = {
            row["blind_id"]
            for row in subset
        }

        if (
            len(blind_ids)
            != expected_items_per_model
        ):
            raise RuntimeError(
                f"{model}: expected "
                f"{expected_items_per_model} "
                f"unique items, found "
                f"{len(blind_ids)}"
            )

        expected_observations = (
            expected_items_per_model
            * len(EVALUATORS)
        )

        if (
            len(subset)
            != expected_observations
        ):
            raise RuntimeError(
                f"{model}: expected "
                f"{expected_observations} human "
                f"observations, found "
                f"{len(subset)}"
            )

        result[model] = {
            "items": len(blind_ids),
            **observation_summary(
                subset
            ),
            "by_evaluator": (
                per_evaluator_summary(
                    subset
                )
            ),
            "human_convergence": (
                convergence_summary(
                    subset
                )
            ),
        }

    return result


def write_item_csv(
    rows: list[dict],
):
    with OUTPUT_ITEMS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return OUTPUT_ITEMS


def print_observation_block(
    values: dict,
    indent: str = "  ",
) -> None:
    g = values["grounding"]

    lr = values[
        "limitation_recognition"
    ]

    ue = values[
        "unsupported_evidence"
    ]

    print(
        f"{indent}Grounding distribution:",
        g["distribution"],
    )

    print(
        f"{indent}Grounding mean:",
        g["mean"],
    )

    print(
        f"{indent}Grounding >= 3:",
        g[
            "well_or_critically_grounded"
        ],
        "/",
        g["n"],
        "=",
        g[
            "well_or_critically_grounded_rate"
        ],
    )

    print(
        f"{indent}Limitation recognition:",
        lr["1"],
        "/",
        lr["n"],
        "=",
        lr["positive_rate"],
    )

    print(
        f"{indent}Unsupported evidence:",
        ue["1"],
        "/",
        ue["n"],
        "=",
        ue["positive_rate"],
    )


def print_evaluator_grounding(
    values: dict,
) -> None:
    print(
        "  Grounding by evaluator:"
    )

    for evaluator in EVALUATORS:
        evaluator_g = values[
            "by_evaluator"
        ][evaluator]["grounding"]

        print(
            "   ",
            evaluator,
            "mean=",
            evaluator_g["mean"],
            ">=3=",
            evaluator_g[
                "well_or_critically_grounded"
            ],
            "/",
            evaluator_g["n"],
        )


def print_convergence_block(
    convergence: dict,
    indent: str = "  ",
) -> None:
    grounding = convergence[
        "grounding_convergence"
    ]

    print(
        f"{indent}Grounding convergence:"
    )

    print(
        f"{indent}  unanimous=",
        grounding["unanimous"],
    )

    print(
        f"{indent}  two_same_one_different=",
        grounding[
            "two_same_one_different"
        ],
    )

    print(
        f"{indent}  all_different=",
        grounding["all_different"],
    )

    print(
        f"{indent}Grounding range:",
        convergence[
            "grounding_range"
        ],
    )

    print(
        f"{indent}Assessors rating "
        f"grounding >= 3:",
        convergence[
            "number_of_assessors_rating_grounding_ge_3"
        ],
    )

    lr = convergence[
        "limitation_recognition"
    ]

    print(
        f"{indent}LR unanimous:",
        lr["unanimous_items"],
        "/",
        convergence["items"],
        "=",
        lr["unanimous_rate"],
    )

    print(
        f"{indent}LR positive assessors:",
        lr[
            "number_of_positive_assessors"
        ],
    )

    ue = convergence[
        "unsupported_evidence"
    ]

    print(
        f"{indent}UE unanimous:",
        ue["unanimous_items"],
        "/",
        convergence["items"],
        "=",
        ue["unanimous_rate"],
    )

    print(
        f"{indent}UE positive assessors:",
        ue[
            "number_of_positive_assessors"
        ],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    registry = load_case_registry()

    enrich_with_evidential_state(
        rows,
        registry,
    )

    item_rows, item_summary = (
        item_observability(
            rows
        )
    )

    by_condition = (
        condition_observability(
            rows
        )
    )

    by_evidential_state = (
        evidential_state_observability(
            rows
        )
    )

    by_model = (
        model_observability(
            rows
        )
    )

    result = {
        "design": {
            "items": EXPECTED_ITEMS,
            "evaluators": len(
                EVALUATORS
            ),
            "human_observations": len(
                rows
            ),
            "interpretation": (
                "Descriptive external "
                "observations of the "
                "evidence-justification "
                "relationship; no human "
                "ground truth or consensus "
                "label is constructed."
            ),
        },
        "aggregate": (
            observation_summary(
                rows
            )
        ),
        "by_evaluator": (
            per_evaluator_summary(
                rows
            )
        ),
        "item_level_observability": (
            item_summary
        ),
        "by_evidential_condition": (
            by_condition
        ),
        "by_pre_specified_evidential_state": (
            by_evidential_state
        ),
        "by_evaluator_model": (
            by_model
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

    item_csv = write_item_csv(
        item_rows
    )

    print(
        "Grounding observability analysis: OK"
    )

    print()
    print(
        "AGGREGATE HUMAN OBSERVATIONS"
    )

    print(
        "n =",
        result[
            "aggregate"
        ]["human_observations"],
    )

    print_observation_block(
        result["aggregate"],
        indent="",
    )

    print()
    print("GROUNDING CONVERGENCE")

    print_convergence_block(
        item_summary,
        indent="",
    )

    print()
    print("BY EVIDENTIAL CONDITION")

    for evidential_condition in (
        "E0",
        "E1",
        "E2",
    ):
        values = by_condition[
            evidential_condition
        ]

        print()
        print(
            evidential_condition,
            f"({values['source_condition']})",
        )

        print_observation_block(
            values
        )

        print_evaluator_grounding(
            values
        )

    print()
    print(
        "BY PRE-SPECIFIED EVIDENTIAL STATE"
    )

    for state, values in (
        by_evidential_state.items()
    ):
        print()
        print(state)

        print(
            "  Cases:",
            ", ".join(
                values["case_ids"]
            ),
        )

        print(
            "  Items:",
            values["items"],
        )

        print_observation_block(
            values
        )

        print_evaluator_grounding(
            values
        )

    print()
    print("BY EVALUATOR MODEL")

    for model in MODELS:
        values = by_model[model]

        print()
        print(model)

        print(
            "  Items:",
            values["items"],
        )

        print_observation_block(
            values
        )

        print_evaluator_grounding(
            values
        )

        print_convergence_block(
            values[
                "human_convergence"
            ]
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )

    print(
        "Items:",
        item_csv,
    )


if __name__ == "__main__":
    main()
