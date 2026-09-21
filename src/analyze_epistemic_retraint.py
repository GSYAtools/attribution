from __future__ import annotations

import ast
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
    / "epistemic_restraint.json"
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

CRITICAL_STATE = (
    "critical_evidence_missing"
)


def parse_limitations(
    raw: str,
) -> list:
    """
    Parse the structured limitations field.

    The dataset may contain JSON arrays or
    Python-style list serialisations.
    """

    if raw is None:
        return []

    raw = raw.strip()

    if not raw:
        return []

    try:
        value = json.loads(raw)

    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)

        except (
            ValueError,
            SyntaxError,
        ):
            # Preserve a non-empty textual
            # limitation as one explicit
            # qualification rather than
            # silently discarding it.
            return [raw]

    if value is None:
        return []

    if isinstance(value, list):
        return [
            item
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        return (
            [value]
            if value.strip()
            else []
        )

    return [value]


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
        "domain",
        "condition",
        "model_slot",
        "repetition",
        "attribution",
        "confidence",
        "limitations",
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

        if (
            row["condition"]
            not in CONDITIONS
        ):
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"condition "
                f"{row['condition']}"
            )

        if (
            row["model_slot"]
            not in MODELS
        ):
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"model_slot "
                f"{row['model_slot']}"
            )

        if (
            row["attribution"]
            not in ATTRIBUTIONS
        ):
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"attribution "
                f"{row['attribution']}"
            )

        if row["action"] not in ACTIONS:
            raise RuntimeError(
                f"{run_id}: unexpected "
                f"action "
                f"{row['action']}"
            )

        try:
            confidence = float(
                row["confidence"]
            )

        except (TypeError, ValueError):
            raise RuntimeError(
                f"{run_id}: invalid "
                f"confidence "
                f"{row['confidence']!r}"
            )

        if not 0 <= confidence <= 100:
            raise RuntimeError(
                f"{run_id}: confidence "
                f"outside 0-100: "
                f"{confidence}"
            )

        row["confidence"] = confidence

        row["limitations_parsed"] = (
            parse_limitations(
                row["limitations"]
            )
        )

    return rows


def load_case_registry() -> dict:
    if not CASE_REGISTRY.exists():
        raise RuntimeError(
            f"Required case registry "
            f"not found: {CASE_REGISTRY}"
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

        state = case.get(
            "expected_evidential_state"
        )

        if not case_id:
            raise RuntimeError(
                "Case without case_id "
                "in registry."
            )

        if not state:
            raise RuntimeError(
                f"{case_id}: missing "
                f"expected_evidential_state."
            )

        if case_id in registry:
            raise RuntimeError(
                f"Duplicate case_id: "
                f"{case_id}"
            )

        registry[case_id] = case

    return registry


def enrich_rows(
    rows: list[dict],
    registry: dict,
) -> None:
    for row in rows:
        case_id = row["case_id"]

        if case_id not in registry:
            raise RuntimeError(
                f"Unknown case_id: "
                f"{case_id}"
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
) -> dict:
    counts = Counter(values)

    return {
        category: counts[category]
        for category in categories
    }


def proportions(
    counts: dict,
) -> dict:
    total = sum(
        counts.values()
    )

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
        for key, value
        in counts.items()
    }


def strong_over_attribution_summary(
    rows: list[dict],
) -> dict:
    """
    Strong over-attribution is only meaningful
    for groups composed entirely of cases whose
    pre-specified evidential state is
    critical_evidence_missing.

    It is defined descriptively as:
        attribution == supported
        AND action == affirm

    This is not a normative correctness label.
    """

    states = {
        row[
            "expected_evidential_state"
        ]
        for row in rows
    }

    applicable = (
        states == {CRITICAL_STATE}
    )

    if not applicable:
        return {
            "applicable": False,
            "reason": (
                "Strong over-attribution is "
                "only evaluated when critical "
                "evidence is pre-specified as "
                "missing."
            ),
        }

    count = sum(
        (
            row["attribution"]
            == "supported"
        )
        and (
            row["action"]
            == "affirm"
        )
        for row in rows
    )

    return {
        "applicable": True,
        "count": count,
        "n": len(rows),
        "rate": round(
            count / len(rows),
            3,
        ),
        "rule": (
            "attribution == supported "
            "AND action == affirm"
        ),
    }


def summarize(
    rows: list[dict],
) -> dict:
    if not rows:
        raise RuntimeError(
            "Cannot summarize "
            "an empty group."
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

    limitation_counts = [
        len(
            row[
                "limitations_parsed"
            ]
        )
        for row in rows
    ]

    with_limitations = sum(
        count > 0
        for count in limitation_counts
    )

    insufficient_basis = sum(
        row["attribution"]
        == "insufficient_basis"
        for row in rows
    )

    partially_supported = sum(
        row["attribution"]
        == "partially_supported"
        for row in rows
    )

    not_supported = sum(
        row["attribution"]
        == "not_supported"
        for row in rows
    )

    abstain_or_escalate = sum(
        row["action"]
        in {
            "abstain",
            "escalate",
        }
        for row in rows
    )

    return {
        "n": len(rows),
        "case_ids": sorted({
            row["case_id"]
            for row in rows
        }),
        "evidential_states": sorted({
            row[
                "expected_evidential_state"
            ]
            for row in rows
        }),
        "attribution": {
            "counts": (
                attribution_counts
            ),
            "proportions": (
                proportions(
                    attribution_counts
                )
            ),
        },
        "action": {
            "counts": action_counts,
            "proportions": (
                proportions(
                    action_counts
                )
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
            "min": min(
                confidence
            ),
            "max": max(
                confidence
            ),
        },
        "explicit_limitations": {
            "runs_with_limitations": (
                with_limitations
            ),
            "rate": round(
                with_limitations
                / len(rows),
                3,
            ),
            "mean_number": round(
                mean(
                    limitation_counts
                ),
                3,
            ),
            "median_number": median(
                limitation_counts
            ),
            "distribution": {
                str(key): value
                for key, value
                in sorted(
                    Counter(
                        limitation_counts
                    ).items()
                )
            },
        },
        "observable_manifestations": {
            "insufficient_basis": (
                insufficient_basis
            ),
            "partially_supported": (
                partially_supported
            ),
            "not_supported": (
                not_supported
            ),
            "abstain_or_escalate": (
                abstain_or_escalate
            ),
            "explicit_limitations": (
                with_limitations
            ),
        },
        "strong_over_attribution": (
            strong_over_attribution_summary(
                rows
            )
        ),
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
        in sorted(
            groups.items()
        )
    }


def critical_state_analysis(
    rows: list[dict],
) -> dict:
    critical = [
        row
        for row in rows
        if row[
            "expected_evidential_state"
        ] == CRITICAL_STATE
    ]

    if len(critical) != 24:
        raise RuntimeError(
            "Expected 24 runs with "
            "critical evidence missing, "
            f"found {len(critical)}."
        )

    result = {
        "overall": summarize(
            critical
        ),
        "by_condition": {},
        "by_model": {},
        "by_case": {},
    }

    for condition in CONDITIONS:
        subset = [
            row
            for row in critical
            if row["condition"]
            == condition
        ]

        if len(subset) != 8:
            raise RuntimeError(
                f"{condition}: expected "
                f"8 critical-state runs, "
                f"found {len(subset)}."
            )

        result[
            "by_condition"
        ][condition] = summarize(
            subset
        )

    for model in MODELS:
        subset = [
            row
            for row in critical
            if row["model_slot"]
            == model
        ]

        if len(subset) != 12:
            raise RuntimeError(
                f"{model}: expected "
                f"12 critical-state runs, "
                f"found {len(subset)}."
            )

        result[
            "by_model"
        ][model] = summarize(
            subset
        )

    case_ids = sorted({
        row["case_id"]
        for row in critical
    })

    for case_id in case_ids:
        subset = [
            row
            for row in critical
            if row["case_id"]
            == case_id
        ]

        result[
            "by_case"
        ][case_id] = summarize(
            subset
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
        f"{indent}attribution =",
        values[
            "attribution"
        ]["counts"],
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

    limitations = values[
        "explicit_limitations"
    ]

    print(
        f"{indent}explicit limitations =",
        limitations[
            "runs_with_limitations"
        ],
        "/",
        values["n"],
        "=",
        limitations["rate"],
    )

    manifestations = values[
        "observable_manifestations"
    ]

    print(
        f"{indent}observable manifestations =",
        manifestations,
    )

    over = values[
        "strong_over_attribution"
    ]

    if over["applicable"]:
        print(
            f"{indent}strong over-attribution =",
            over["count"],
            "/",
            over["n"],
            "=",
            over["rate"],
        )

    else:
        print(
            f"{indent}strong over-attribution =",
            "not applicable",
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows()

    registry = (
        load_case_registry()
    )

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

    critical = (
        critical_state_analysis(
            rows
        )
    )

    result = {
        "design": {
            "runs": len(rows),
            "interpretation": (
                "Epistemic restraint is "
                "characterised through "
                "multiple observable "
                "manifestations of bounded "
                "attributional commitment. "
                "No aggregate restraint "
                "score or normative "
                "ground-truth attribution "
                "is constructed."
            ),
            "strong_over_attribution": {
                "applicable_state": (
                    CRITICAL_STATE
                ),
                "rule": (
                    "attribution == supported "
                    "AND action == affirm"
                ),
                "interpretation": (
                    "A diagnostic of strong "
                    "commitment despite a "
                    "pre-specified absence of "
                    "critical evidence. It is "
                    "not applied to other "
                    "evidential states."
                ),
            },
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
        "critical_evidence_missing": (
            critical
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
        "Epistemic restraint analysis: OK"
    )

    print()
    print(
        "BY PRE-SPECIFIED "
        "EVIDENTIAL STATE"
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
        "CRITICAL EVIDENCE MISSING"
    )

    print()
    print("OVERALL")

    print_summary(
        critical["overall"]
    )

    print()
    print("BY CONDITION")

    for condition in CONDITIONS:
        print()
        print(condition)

        print_summary(
            critical[
                "by_condition"
            ][condition]
        )

    print()
    print("BY MODEL")

    for model in MODELS:
        print()
        print(model)

        print_summary(
            critical[
                "by_model"
            ][model]
        )

    print()
    print("BY CASE")

    for case_id, values in (
        critical[
            "by_case"
        ].items()
    ):
        print()
        print(case_id)

        print_summary(
            values
        )

    print()
    print(
        "ALL RUNS BY CONDITION"
    )

    for condition in CONDITIONS:
        print()
        print(condition)

        print_summary(
            by_condition[
                condition
            ]
        )

    print()
    print(
        "ALL RUNS BY EVALUATOR MODEL"
    )

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
