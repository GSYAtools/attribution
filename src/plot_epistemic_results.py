from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import ROOT


# ============================================================
# Paths
# ============================================================

ANALYSIS_DIR = ROOT / "analysis"

FIGURE_DIR = (
    ANALYSIS_DIR
    / "figures"
)

D_FILE = (
    ANALYSIS_DIR
    / "discriminative_capacity.json"
)

R_FILE = (
    ANALYSIS_DIR
    / "epistemic_restraint.json"
)

G_FILE = (
    ANALYSIS_DIR
    / "human_evaluation"
    / "grounding_observability.json"
)

C_FILE = (
    ANALYSIS_DIR
    / "consistency.json"
)

V_FILE = (
    ANALYSIS_DIR
    / "revisability.json"
)


D_OUTPUT_PDF = (
    FIGURE_DIR
    / "discriminative_capacity_profiles.pdf"
)

D_OUTPUT_PNG = (
    FIGURE_DIR
    / "discriminative_capacity_profiles.png"
)


SUMMARY_OUTPUT_PDF = (
    FIGURE_DIR
    / "epistemic_white_box_results.pdf"
)

SUMMARY_OUTPUT_PNG = (
    FIGURE_DIR
    / "epistemic_white_box_results.png"
)


# ============================================================
# General utilities
# ============================================================

def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"Required analysis file "
            f"not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Figure 1
# Discriminative-capacity attribution profiles
# ============================================================

def extract_discriminative_profiles() -> dict:
    data = load_json(
        D_FILE
    )

    profiles = data[
        "attribution_profiles"
    ]

    broad_key = (
        "broad_evidence_present"
    )

    critical_key = (
        "critical_evidence_missing"
    )

    if broad_key not in profiles:
        raise RuntimeError(
            "Broad-evidence profile "
            "not found."
        )

    if critical_key not in profiles:
        raise RuntimeError(
            "Critical-evidence-missing "
            "profile not found."
        )

    broad = profiles[
        broad_key
    ]

    critical = profiles[
        critical_key
    ]

    tv_key = (
        "broad_evidence_present"
        "__vs__"
        "critical_evidence_missing"
    )

    tv = data[
        "pairwise_attribution_profile_tv_distance"
    ][tv_key]

    categories = [
        "supported",
        "partially_supported",
        "insufficient_basis",
        "not_supported",
    ]

    labels = [
        "Supported",
        "Partially supported",
        "Insufficient basis",
        "Not supported",
    ]

    return {
        "categories": categories,
        "labels": labels,
        "broad": {
            "label": (
                "Broad evidence present"
            ),
            "n": broad["n"],
            "counts": [
                broad["counts"][
                    category
                ]
                for category
                in categories
            ],
        },
        "critical": {
            "label": (
                "Critical evidence missing"
            ),
            "n": critical["n"],
            "counts": [
                critical["counts"][
                    category
                ]
                for category
                in categories
            ],
        },
        "tv": tv,
    }


def build_discriminative_figure(
    profiles: dict,
) -> plt.Figure:
    """
    Visualise attribution profiles for broad
    evidence and critical-evidence-missing cases.

    No title is included because figure numbering
    and captioning are handled in LaTeX.
    """

    labels = profiles[
        "labels"
    ]

    broad_counts = profiles[
        "broad"
    ]["counts"]

    critical_counts = profiles[
        "critical"
    ]["counts"]

    n_broad = profiles[
        "broad"
    ]["n"]

    n_critical = profiles[
        "critical"
    ]["n"]

    fig, ax = plt.subplots(
        figsize=(9.2, 5.0)
    )

    broad_y = [
        7.0,
        6.0,
        5.0,
        4.0,
    ]

    critical_y = [
        2.5,
        1.5,
        0.5,
        -0.5,
    ]

    ax.barh(
        broad_y,
        broad_counts,
        height=0.58,
    )

    ax.barh(
        critical_y,
        critical_counts,
        height=0.58,
    )

    label_x = -0.35

    for y, label in zip(
        broad_y,
        labels,
    ):
        ax.text(
            label_x,
            y,
            label,
            ha="right",
            va="center",
            fontsize=10,
        )

    for y, label in zip(
        critical_y,
        labels,
    ):
        ax.text(
            label_x,
            y,
            label,
            ha="right",
            va="center",
            fontsize=10,
        )

    for y, count in zip(
        broad_y,
        broad_counts,
    ):
        ax.text(
            count + 0.25,
            y,
            str(count),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for y, count in zip(
        critical_y,
        critical_counts,
    ):
        ax.text(
            count + 0.25,
            y,
            str(count),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0,
        7.85,
        (
            "Broad evidence present"
            f"  (n = {n_broad})"
        ),
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    ax.text(
        0,
        3.35,
        (
            "Critical evidence missing"
            f"  (n = {n_critical})"
        ),
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    ax.axhline(
        3.55,
        linewidth=0.8,
    )

    max_count = max(
        broad_counts
        + critical_counts
    )

    ax.set_xlim(
        -0.5,
        max_count + 3.0,
    )

    ax.set_ylim(
        -1.2,
        8.35,
    )

    ax.set_xlabel(
        "Number of assessments",
        fontsize=10,
    )

    ax.set_yticks([])

    upper_tick = (
        (max_count // 5 + 1)
        * 5
    )

    ax.set_xticks(
        list(
            range(
                0,
                upper_tick + 1,
                5,
            )
        )
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
    )

    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        top=0.96,
        bottom=0.13,
    )

    return fig


# ============================================================
# Figure 2
# Hybrid empirical synthesis
# ============================================================

def extract_summary_results() -> dict:
    d = load_json(
        D_FILE
    )

    r = load_json(
        R_FILE
    )

    g = load_json(
        G_FILE
    )

    c = load_json(
        C_FILE
    )

    v = load_json(
        V_FILE
    )

    # --------------------------------------------------------
    # D — Discriminative capacity
    # --------------------------------------------------------

    tv_key = (
        "broad_evidence_present"
        "__vs__"
        "critical_evidence_missing"
    )

    tv = d[
        "pairwise_attribution_profile_tv_distance"
    ][tv_key]

    # --------------------------------------------------------
    # R — Epistemic restraint
    # --------------------------------------------------------

    restraint = r[
        "critical_evidence_missing"
    ]["overall"][
        "strong_over_attribution"
    ]

    if not restraint[
        "applicable"
    ]:
        raise RuntimeError(
            "Strong over-attribution "
            "diagnostic unexpectedly "
            "not applicable."
        )

    # --------------------------------------------------------
    # G — Evidential grounding
    # --------------------------------------------------------

    grounding = g[
        "aggregate"
    ]["grounding"]

    # --------------------------------------------------------
    # C — Consistency
    # --------------------------------------------------------

    consistency = c[
        "overall"
    ][
        "attribution_exact_agreement"
    ]

    consistency_n = c[
        "overall"
    ]["pairs"]

    # --------------------------------------------------------
    # V — Selective revisability
    # --------------------------------------------------------

    revisability = v[
        "paired_selective_revisability"
    ]["overall"][
        "both_expected"
    ]

    revisability_n = v[
        "paired_selective_revisability"
    ]["overall"][
        "pairs"
    ]

    return {
        "D": {
            "name": (
                "Discriminative\ncapacity"
            ),
            "symbol": "D",
            "value": (
                f"TV = {tv:.3f}"
            ),
            "percentage": None,
            "description": (
                "Broad evidence vs.\n"
                "critical evidence missing"
            ),
        },
        "R": {
            "name": (
                "Epistemic\nrestraint"
            ),
            "symbol": "R",
            "value": (
                f"{restraint['count']}"
                f" / {restraint['n']}"
            ),
            "percentage": (
                f"{100 * restraint['rate']:.1f}%"
            ),
            "description": (
                "Strong over-attribution\n"
                "with critical evidence missing"
            ),
        },
        "G": {
            "name": (
                "Evidential\ngrounding"
            ),
            "symbol": "G",
            "value": (
                f"{grounding['well_or_critically_grounded']}"
                f" / {grounding['n']}"
            ),
            "percentage": (
                f"{100 * grounding['well_or_critically_grounded_rate']:.1f}%"
            ),
            "description": (
                "Blinded human observations\n"
                "with grounding score ≥ 3"
            ),
        },
        "C": {
            "name": (
                "Consistency"
            ),
            "symbol": "C",
            "value": (
                f"{consistency['count']}"
                f" / {consistency_n}"
            ),
            "percentage": (
                f"{100 * consistency['rate']:.1f}%"
            ),
            "description": (
                "Exact attribution agreement\n"
                "across independent repetitions"
            ),
            "state": (
                "Justificatory basis\n"
                "preserved"
            ),
        },
        "V": {
            "name": (
                "Selective\nrevisability"
            ),
            "symbol": "V",
            "value": (
                f"{revisability['count']}"
                f" / {revisability_n}"
            ),
            "percentage": (
                f"{100 * revisability['rate']:.1f}%"
            ),
            "description": (
                "Paired material-defeater /\n"
                "non-material-control response"
            ),
            "state": (
                "Justificatory basis\n"
                "materially changed"
            ),
        },
    }


def build_summary_figure(
    results: dict,
) -> plt.Figure:
    """
    Empirical synthesis of the five epistemic
    properties.

    D, R and G are displayed as three direct
    observations of W.

    C and V remain distinct epistemic properties,
    but are visually grouped as complementary
    manifestations of dynamic epistemic behaviour:

        preserved justificatory basis -> stability
        materially changed basis      -> revision

    No title is included because title and caption
    are handled in LaTeX.
    """

    fig, ax = plt.subplots(
        figsize=(11.8, 7.0)
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis(
        "off"
    )

    # ========================================================
    # Top level: Evidence -> W
    # ========================================================

    evidence_x = 0.22
    process_x = 0.50
    top_y = 0.91

    ax.text(
        evidence_x,
        top_y,
        "Evidence\n$E$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(
            process_x - 0.085,
            top_y,
        ),
        xytext=(
            evidence_x + 0.055,
            top_y,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
        },
    )

    ax.text(
        process_x,
        top_y,
        "Evaluative process\n$W$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox={
            "boxstyle": (
                "round,pad=0.45"
            ),
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.1,
        },
    )

    # ========================================================
    # First three observations: D, R, G
    # ========================================================

    upper_branch_y = 0.79

    upper_x = {
        "D": 0.20,
        "R": 0.50,
        "G": 0.80,
    }

    ax.plot(
        [
            process_x,
            process_x,
        ],
        [
            top_y - 0.055,
            upper_branch_y,
        ],
        linewidth=1.2,
    )

    ax.plot(
        [
            upper_x["D"],
            upper_x["G"],
        ],
        [
            upper_branch_y,
            upper_branch_y,
        ],
        linewidth=1.2,
    )

    upper_property_y = 0.665

    for key in (
        "D",
        "R",
        "G",
    ):
        x = upper_x[key]
        result = results[key]

        ax.annotate(
            "",
            xy=(
                x,
                upper_property_y + 0.055,
            ),
            xytext=(
                x,
                upper_branch_y,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.1,
            },
        )

        ax.text(
            x,
            upper_property_y + 0.035,
            result["symbol"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            x,
            upper_property_y - 0.015,
            result["name"],
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.15,
        )

        ax.text(
            x,
            upper_property_y - 0.105,
            result["value"],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        if result[
            "percentage"
        ]:
            ax.text(
                x,
                upper_property_y - 0.145,
                result["percentage"],
                ha="center",
                va="center",
                fontsize=9.5,
            )

        ax.text(
            x,
            upper_property_y - 0.205,
            result["description"],
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.2,
        )

    # ========================================================
    # Dynamic epistemic behaviour
    # ========================================================

    dynamic_header_y = 0.365

    ax.text(
        0.50,
        dynamic_header_y,
        "Dynamic epistemic behaviour",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    # Small conceptual line linking the heading
    # to its two distinct epistemic properties.
    dynamic_branch_y = 0.325

    c_x = 0.34
    v_x = 0.66

    ax.plot(
        [
            0.50,
            0.50,
        ],
        [
            dynamic_header_y - 0.022,
            dynamic_branch_y,
        ],
        linewidth=1.1,
    )

    ax.plot(
        [
            c_x,
            v_x,
        ],
        [
            dynamic_branch_y,
            dynamic_branch_y,
        ],
        linewidth=1.1,
    )

    # ========================================================
    # C — preserved justificatory basis
    # ========================================================

    lower_state_y = 0.265
    lower_property_y = 0.155

    for key, x in (
        ("C", c_x),
        ("V", v_x),
    ):
        result = results[key]

        ax.annotate(
            "",
            xy=(
                x,
                lower_state_y + 0.018,
            ),
            xytext=(
                x,
                dynamic_branch_y,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.0,
            },
        )

        ax.text(
            x,
            lower_state_y,
            result["state"],
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            linespacing=1.15,
        )

        ax.annotate(
            "",
            xy=(
                x,
                lower_property_y + 0.052,
            ),
            xytext=(
                x,
                lower_state_y - 0.035,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.0,
            },
        )

        ax.text(
            x,
            lower_property_y + 0.035,
            result["symbol"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            x,
            lower_property_y - 0.005,
            result["name"],
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.15,
        )

        ax.text(
            x,
            lower_property_y - 0.075,
            result["value"],
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
        )

        ax.text(
            x,
            lower_property_y - 0.112,
            result["percentage"],
            ha="center",
            va="center",
            fontsize=9.5,
        )

    # ========================================================
    # Methodological reminder
    # ========================================================

    ax.text(
        0.50,
        0.012,
        (
            "Five distinct epistemic properties; "
            "no aggregate epistemic score."
        ),
        ha="center",
        va="center",
        fontsize=8.3,
        style="italic",
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.98,
        bottom=0.025,
    )

    return fig


# ============================================================
# Save
# ============================================================

def save_figure(
    fig: plt.Figure,
    pdf_path: Path,
    png_path: Path,
) -> None:
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Figure 1: discriminative capacity
    # --------------------------------------------------------

    profiles = (
        extract_discriminative_profiles()
    )

    d_figure = (
        build_discriminative_figure(
            profiles
        )
    )

    save_figure(
        d_figure,
        D_OUTPUT_PDF,
        D_OUTPUT_PNG,
    )

    # --------------------------------------------------------
    # Figure 2: empirical synthesis
    # --------------------------------------------------------

    summary_results = (
        extract_summary_results()
    )

    summary_figure = (
        build_summary_figure(
            summary_results
        )
    )

    save_figure(
        summary_figure,
        SUMMARY_OUTPUT_PDF,
        SUMMARY_OUTPUT_PNG,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "Epistemic result figures: OK"
    )

    print()
    print(
        "DISCRIMINATIVE CAPACITY"
    )

    print(
        "Broad evidence:",
        profiles[
            "broad"
        ]["counts"],
    )

    print(
        "Critical evidence missing:",
        profiles[
            "critical"
        ]["counts"],
    )

    print(
        "TV =",
        profiles["tv"],
    )

    print()
    print(
        "EMPIRICAL SYNTHESIS"
    )

    for key in (
        "D",
        "R",
        "G",
        "C",
        "V",
    ):
        value = (
            summary_results[key]
        )

        print(
            key,
            value["value"],
            value[
                "percentage"
            ] or "",
        )

    print()
    print(
        "Dynamic epistemic behaviour:"
    )

    print(
        "C ->",
        summary_results[
            "C"
        ]["state"].replace(
            "\n",
            " ",
        ),
    )

    print(
        "V ->",
        summary_results[
            "V"
        ]["state"].replace(
            "\n",
            " ",
        ),
    )

    print()
    print(
        "Generated:"
    )

    print(
        D_OUTPUT_PDF
    )

    print(
        D_OUTPUT_PNG
    )

    print(
        SUMMARY_OUTPUT_PDF
    )

    print(
        SUMMARY_OUTPUT_PNG
    )


if __name__ == "__main__":
    main()
