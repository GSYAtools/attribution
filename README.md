# Epistemic White Box

Empirical artefact for the controlled validation of an **Epistemic White
Box** approach to trustworthiness attribution in AI systems.

The project investigates whether an evaluative process can be
**epistemically audited without assuming a normative ground-truth
trustworthiness attribution**. Rather than asking whether an evaluator
produces the uniquely correct attribution, the experiment examines
whether the transformation from evidence to attribution exhibits
observable epistemic properties.

## Research question

> **Can a trustworthiness attribution process be epistemically audited
> without a normative ground-truth attribution?**

The evaluative process is represented abstractly as:

`E -> W -> A`

where **E** is the available evidential state, **W** is the evaluative
process, and **A** is the resulting structured attribution.

The empirical framework retains five observable epistemic properties:

-   **D --- Discriminative Capacity**
-   **R --- Epistemic Restraint**
-   **G --- Evidential Grounding**
-   **C --- Consistency**
-   **V --- Selective Revisability**

The results are narrated through four empirical observations.
Consistency and Selective Revisability remain distinct properties but
are treated as complementary manifestations of **dynamic epistemic
behaviour**: stability when the justificatory basis is preserved and
revision when it changes materially.

The five properties are deliberately **not aggregated into a single
epistemic or trustworthiness score**.

------------------------------------------------------------------------

## Experimental design

The experiment consists of two controlled stages plus a blinded human
evaluation.

### Stage 1 --- Attribution under controlled evidential conditions

Six synthetic vignettes are evaluated under three evaluative conditions,
using two evaluator-model realisations and two independent repetitions:

`6 x 3 x 2 x 2 = 72`

This produces **72 independent Stage 1 assessments**.

The cases include pre-specified evidential states:

-   `broad_evidence_present`
-   `critical_evidence_missing`
-   `procedural_evidence_incomplete`
-   `broad_procedural_evidence_present`

These states are fixed design characteristics of the vignettes. They are
**not normative ground-truth attribution labels**.

The three evaluative conditions preserve the underlying case evidence
while varying the evaluative framing. They should therefore not be
interpreted as a monotonic scale of evidential quality.

### Stage 2 --- Controlled revision

Two Stage 1 cases are subjected to two controlled update types:

-   **Material defeater (MD):** new information that materially changes
    the justificatory basis relevant to the attribution.
-   **Non-material control (NC):** new information that preserves the
    relevant justificatory basis.

Across three conditions, two evaluator-model realisations, two
repetitions and two update types:

`2 x 3 x 2 x 2 x 2 = 48`

This produces **48 revision assessments**, organised into **24 paired
MD/NC comparisons**.

The study-specific directional expectations are:

-   MD -\> `weaken` or `withdraw`
-   NC -\> `maintain`

These expectations belong to the controlled Stage 2 design and are not
proposed as universal normative rules for trustworthiness attribution.

### Human evaluation

The 72 Stage 1 assessments were independently inspected by three blinded
human assessors:

`72 x 3 = 216`

The human evaluation records:

-   grounding score;
-   limitation recognition;
-   unsupported evidence.

Human assessors are **not used to construct a normative ground truth or
consensus attribution**. Their role is to provide external observations
of the relationship between the available evidence and the generated
justification.

------------------------------------------------------------------------

## Empirical narrative

The empirical results are intended to answer four connected questions
about the same evaluative process.

### 1. Does the process distinguish evidential states?

This observation operationalises **Discriminative Capacity (D)**.

Cases with broad supporting evidence predominantly produce `supported`
and `partially_supported` attributions, whereas cases in which evidence
required by the evaluative question is deliberately absent predominantly
produce `insufficient_basis` and `not_supported`.

For the two clearest evidential profiles:

-   broad evidence present: 14 `supported`, 10 `partially_supported`, 0
    `insufficient_basis`, 0 `not_supported`;
-   critical evidence missing: 0 `supported`, 2 `partially_supported`,
    15 `insufficient_basis`, 7 `not_supported`.

The total variation distance between these attributional profiles is:

**TV = 0.917**

This value characterises separation between observed attribution
profiles. It is **not an accuracy measure** and does not assume a
normatively correct attribution.

Run:

``` bash
python -m src.analyze_discriminative_capacity
```

Output:

``` text
analysis/discriminative_capacity.json
```

### 2. Does the process moderate its commitment when evidence is insufficient?

This observation operationalises **Epistemic Restraint (R)**.

Restraint is characterised through multiple observable manifestations
rather than a single binary or aggregate score, including reduced
attributional commitment, `insufficient_basis`, `not_supported`,
`abstain`/`escalate`, and explicit limitations.

For the 24 assessments associated with `critical_evidence_missing`:

-   `supported`: **0/24**
-   `insufficient_basis`: **15/24**
-   `not_supported`: **7/24**
-   `partially_supported`: **2/24**
-   `abstain` or `escalate`: **18/24**
-   explicit limitations: **22/24**

For this controlled evidential state, strong over-attribution is
operationalised as:

`attribution == supported AND action == affirm`

Observed strong over-attribution:

**0/24**

This diagnostic is only applicable when critical evidence is
pre-specified as missing.

Run:

``` bash
python -m src.analyze_epistemic_retraint
```

Output:

``` text
analysis/epistemic_restraint.json
```

### 3. Does the justification remain grounded in the available evidence?

This observation operationalises **Evidential Grounding (G)**.

Across the 216 blinded human observations:

-   grounding score \>= 3: **180/216 (83.3%)**
-   limitation recognition: **202/216 (93.5%)**
-   unsupported evidence identified: **3/216 (1.4%)**

The analysis preserves assessor-level observations rather than
constructing a consensus ground-truth label.

Importantly, evidential sufficiency and evidential grounding are not
equivalent. Among observations associated with
`critical_evidence_missing` cases:

-   grounding score \>= 3: **56/72 (77.8%)**
-   limitation recognition: **67/72 (93.1%)**

An assessment can therefore remain grounded precisely because its
justification correctly represents the insufficiency of the available
evidence.

Run:

``` bash
python -m src.analyze_grounding_observability
```

Outputs:

``` text
analysis/human_evaluation/grounding_observability.json
analysis/human_evaluation/grounding_observability_items.csv
```

### 4. Does the process adapt structurally to the state of the evidence?

The final empirical movement combines two distinct but complementary
properties under **dynamic epistemic behaviour**.

#### C --- Consistency: justificatory basis preserved

Consistency examines stability across independent repetitions while
preserving case, evaluative condition and evaluator-model realisation.
Exact textual identity is not required.

Across 36 repetition pairs:

-   exact attribution agreement: **31/36 (86.1%)**
-   exact action agreement: **32/36 (88.9%)**
-   mean absolute confidence difference: **1.08**

Run:

``` bash
python -m src.analyze_round1_stability
```

Output:

``` text
analysis/consistency.json
```

#### V --- Selective Revisability: justificatory basis materially changed

Selective Revisability examines whether the process revises when the
justificatory basis changes materially while remaining stable under a
non-material control.

Results:

-   material defeaters following the expected direction: **20/24
    (83.3%)**
-   non-material controls following the expected direction: **24/24
    (100%)**
-   paired selective revisability: **20/24 (83.3%)**

Thus, Consistency and Selective Revisability are not treated as opposing
requirements. They describe complementary responses to whether the
justificatory basis is preserved or materially modified.

Run:

``` bash
python -m src.analyze_revisability
```

Output:

``` text
analysis/revisability.json
```

------------------------------------------------------------------------

## Revision intensity

Revision intensity is a **secondary diagnostic** of Selective
Revisability, not a sixth epistemic property.

It characterises:

-   whether the categorical attribution changes;
-   confidence change after an evidential update;
-   non-selective material-defeater responses.

Run:

``` bash
python -m src.analyze_revision_intensity
```

Output:

``` text
analysis/revision_intensity.json
```

Confidence represents commitment to the resulting attribution. A
material defeater therefore need not produce a negative confidence
delta: the evaluator may become more confident in a weakened
attribution.

------------------------------------------------------------------------

## Empirical figures

All figures are generated directly from the frozen JSON analysis
outputs. No reported result is hard-coded into the plotting script.

Run:

``` bash
python -m src.plot_epistemic_results
```

### Discriminative-capacity profiles

Outputs:

``` text
analysis/figures/discriminative_capacity_profiles.pdf
analysis/figures/discriminative_capacity_profiles.png
```

This figure visualises the attribution distributions for
`broad_evidence_present` and `critical_evidence_missing`. The TV
distance is reported separately in the analysis and manuscript.

### Empirical synthesis

Outputs:

``` text
analysis/figures/epistemic_white_box_results.pdf
analysis/figures/epistemic_white_box_results.png
```

The synthesis preserves all five epistemic properties while visually
grouping Consistency and Selective Revisability under **Dynamic
epistemic behaviour**:

``` text
D     R     G          Dynamic epistemic behaviour
                         /                  \
                        C                    V
              basis preserved      basis materially changed
```

The quantities displayed for D, R, G, C and V operationalise different
epistemic properties and **must not be interpreted as values on a common
scale or as components of an aggregate score**.

------------------------------------------------------------------------

## Repository structure

``` text
attribution/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── case_registry.json
├── src/
│   ├── analyze_discriminative_capacity.py
│   ├── analyze_epistemic_retraint.py
│   ├── analyze_grounding_observability.py
│   ├── analyze_round1_stability.py
│   ├── analyze_revisability.py
│   ├── analyze_revision_intensity.py
│   └── plot_epistemic_results.py
├── analysis/
│   ├── round1_dataset.csv
│   ├── round2_dataset.csv
│   ├── discriminative_capacity.json
│   ├── epistemic_restraint.json
│   ├── consistency.json
│   ├── revisability.json
│   ├── revision_intensity.json
│   ├── human_evaluation/
│   │   ├── blind_scores.csv
│   │   ├── grounding_observability.json
│   │   └── grounding_observability_items.csv
│   └── figures/
│       ├── discriminative_capacity_profiles.pdf
│       ├── discriminative_capacity_profiles.png
│       ├── epistemic_white_box_results.pdf
│       └── epistemic_white_box_results.png
└── runs/
```

The exact contents retained under `runs/` depend on the experimental
provenance and raw execution artefacts selected for release.

------------------------------------------------------------------------

## Reproducing the analyses

From the project root, activate the project environment and execute:

``` bash
python -m src.analyze_discriminative_capacity
python -m src.analyze_epistemic_retraint
python -m src.analyze_grounding_observability
python -m src.analyze_round1_stability
python -m src.analyze_revisability
python -m src.analyze_revision_intensity
python -m src.plot_epistemic_results
```

These analyses operate on the frozen Stage 1, Stage 2 and
human-evaluation datasets. Reproducing the reported analyses therefore
does **not** require re-running the evaluator models.

------------------------------------------------------------------------

## Core datasets

The principal frozen analysis datasets are:

``` text
analysis/round1_dataset.csv
analysis/round2_dataset.csv
analysis/human_evaluation/blind_scores.csv
```

-   `round1_dataset.csv` contains the **72 Stage 1 assessments**.
-   `round2_dataset.csv` contains the **48 controlled revision
    assessments**.
-   `blind_scores.csv` contains the **216 blinded human observations**
    associated with the 72 Stage 1 assessments.

------------------------------------------------------------------------

## Methodological interpretation

This artefact supports **epistemic auditability**, not normative
certification.

The experiment does not assume that there exists one uniquely correct
trustworthiness attribution for every evidential state. Instead, it asks
whether the transformation from evidence to attribution exhibits
empirically observable epistemic behaviour.

Accordingly:

-   no normative `A_gold` is constructed;
-   human assessors are not treated as an oracle;
-   no aggregate epistemic score is calculated;
-   the five epistemic properties remain distinct;
-   Consistency and Selective Revisability are grouped narratively, not
    collapsed into a single property;
-   differences between evaluator-model realisations are characterised
    rather than converted into an overall model ranking;
-   the experiment is a controlled empirical validation of
    observability, not a population-level benchmark;
-   inferential population-level claims are outside the scope of the
    study.

------------------------------------------------------------------------

## Analysis strategy

Results are reported descriptively using counts, proportions,
distributions, paired comparisons and agreement statistics where
appropriate.

The objective is to determine whether the epistemic patterns proposed by
the framework are observable under controlled evidential conditions
rather than to estimate population-level effects. The study therefore
relies on a deliberately constructed set of evaluation cases rather than
random sampling from a defined population.

The empirical observations follow the epistemic argument:

1.  **Distinguish** --- does the process differentiate evidential
    states?
2.  **Moderate** --- does it bound commitment under evidential
    insufficiency?
3.  **Ground** --- does the justification remain anchored in the
    evidence?
4.  **Adapt** --- does it remain stable when the justificatory basis is
    preserved and revise when that basis changes materially?

This four-movement narrative preserves the five formal epistemic
properties because the fourth movement contains the complementary
properties of Consistency and Selective Revisability.

------------------------------------------------------------------------

## Reproducibility and repository hygiene

The repository retains the frozen datasets and generated analysis
outputs used to obtain the reported results.

Analysis JSON files are versioned alongside source code so that reported
quantities can be traced to their analytical artefacts. Figures are
regenerated from those JSON files rather than from hard-coded values.

Environment-specific files, virtual environments, caches and credentials
should not be committed.

At minimum, exclude:

``` text
.env
.env.*
.venv/
venv/
env/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.vscode/
.idea/
.DS_Store
Thumbs.db
*.log
```

API credentials, tokens and other secrets must never be committed.

The `analysis/` directory is intentionally versioned because it contains
the frozen datasets and reproducible analytical outputs used in the
manuscript.

------------------------------------------------------------------------

## Status

The experimental validation is frozen.

The current artefact contains:

-   **72** Stage 1 assessments;
-   **48** Stage 2 revision assessments;
-   **216** blinded human observations;
-   reproducible analyses for **D, R, G, C and V**;
-   a secondary revision-intensity diagnostic;
-   a dedicated Discriminative Capacity figure;
-   a five-property empirical synthesis figure structured around the
    four-movement results narrative.

Further changes to the experimental design should be treated as a new
experimental version rather than silently modifying the frozen results.

