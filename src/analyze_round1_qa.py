import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from src.utils import ROOT


DATASET = ROOT / "analysis" / "round1_dataset.csv"


def load_rows() -> list[dict[str, str]]:
    with DATASET.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    rows = load_rows()

    print("ROWS:", len(rows))
    print()

    print("ATTRIBUTION")
    print(
        Counter(
            row["attribution"]
            for row in rows
        )
    )
    print()

    print("ATTRIBUTION BY CONDITION")

    for condition in (
        "C0",
        "C1",
        "C2",
    ):
        subset = [
            row
            for row in rows
            if row["condition"] == condition
        ]

        print(
            condition,
            Counter(
                row["attribution"]
                for row in subset
            ),
        )

    print()

    print("ATTRIBUTION BY MODEL")

    for model in (
        "model_1",
        "model_2",
    ):
        subset = [
            row
            for row in rows
            if row["model_slot"] == model
        ]

        print(
            model,
            Counter(
                row["attribution"]
                for row in subset
            ),
        )

    print()

    print("MEAN CONFIDENCE BY CONDITION")

    for condition in (
        "C0",
        "C1",
        "C2",
    ):
        values = [
            int(row["confidence"])
            for row in rows
            if row["condition"] == condition
        ]

        print(
            condition,
            round(mean(values), 2),
        )

    print()

    print("ACTION BY CONDITION")

    for condition in (
        "C0",
        "C1",
        "C2",
    ):
        subset = [
            row
            for row in rows
            if row["condition"] == condition
        ]

        print(
            condition,
            Counter(
                row["action"]
                for row in subset
            ),
        )

    print()

    print("MODEL COUNTS")
    print(
        Counter(
            row["model_slot"]
            for row in rows
        )
    )

    print()

    print("DOMAIN COUNTS")
    print(
        Counter(
            row["domain"]
            for row in rows
        )
    )

    print()

    print("REPETITION COUNTS")
    print(
        Counter(
            row["repetition"]
            for row in rows
        )
    )

    print()

    print("TOKEN USAGE")
    input_tokens = [
        int(row["input_tokens"])
        for row in rows
    ]

    output_tokens = [
        int(row["output_tokens"])
        for row in rows
    ]

    total_tokens = [
        int(row["total_tokens"])
        for row in rows
    ]

    print(
        "mean_input_tokens:",
        round(mean(input_tokens), 2),
    )

    print(
        "mean_output_tokens:",
        round(mean(output_tokens), 2),
    )

    print(
        "mean_total_tokens:",
        round(mean(total_tokens), 2),
    )


if __name__ == "__main__":
    main()
