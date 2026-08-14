# SC5 Experimental Study — Scientific Design Notes

## 1. Study motivation

The study investigates a distinction between technical evidence about an AI system and the subsequent attribution of trustworthiness to that system.

Technical measurements can provide evidence about observable system behaviour. However, an attribution such as *fair*, *robust*, or *accountable* requires an evaluative judgement concerning whether the available evidence is sufficient and relevant for supporting that attribution in a given context.

The experiment therefore does not attempt to establish a unique ground-truth label for trustworthiness. Instead, it investigates whether making evaluative conditions explicit changes the way large language models reason from an identical evidential basis toward a trustworthiness attribution.

The study operationalises the Evaluative Governance component of SC5 by introducing explicit governance conditions between technical evidence and attribution.

## 2. Research question

The central research question is:

**RQ — How does the explicit introduction of evaluative criteria and governance conditions affect trustworthiness attribution from an otherwise identical technical evidential basis?**

The experiment focuses on changes in attribution behaviour rather than on establishing whether any individual trustworthiness judgement is objectively correct.

## 3. Experimental principle

The fundamental experimental structure is:

[
C_0 = E
]

[
C_1 = E + C
]

[
C_2 = E + C + G
]

where:

* (E) denotes the technical and contextual evidence provided in a vignette;
* (C) denotes externally grounded evaluation criteria relevant to the evaluated property;
* (G) denotes the Evaluative Governance intervention derived from SC5.

The evidence (E) remains identical across the three conditions.

Consequently:

[
C_1-C_0
]

captures the effect of making evaluation criteria explicit, whereas:

[
C_2-C_1
]

captures the incremental effect associated with the Evaluative Governance intervention.

The latter constitutes the primary contrast for evaluating SC5.

## 4. Experimental conditions

### C0 — Evidence only

The evaluator receives the case context, available evidence, evaluation request, and common output format.

No explicit evaluation criteria or governance procedure are supplied.

This condition represents attribution from technical evidence under the evaluator's implicit evaluative assumptions.

### C1 — Evidence plus explicit evaluation criteria

The evaluator receives exactly the same evidential basis as C0 together with externally grounded criteria relevant to the evaluated property.

The criteria identify dimensions that should be considered during evaluation but do not prescribe a case-specific conclusion.

In particular, the criteria do not contain rules of the form:

> If evidence X is missing, abstain.

The evaluator must still interpret the relevance and sufficiency of the evidence.

### C2 — Evidence, criteria, and Evaluative Governance

C2 contains exactly the same evidence and evaluation criteria as C1 and additionally introduces an Evaluative Governance procedure.

The intervention requires the relationship between evidence, criteria, and attribution to remain explicit; evidential limitations to remain visible; unresolved evaluative issues to be capable of referral for further review; and attributions to remain open to reconsideration when materially relevant evidence or contextual conditions change.

The intervention deliberately does not command the evaluator to abstain, escalate, or revise. These behaviours remain evaluative decisions.

## 5. Experimental domains

The design considers trustworthiness attribution across multiple domains rather than treating trustworthiness as a single scalar property.

The pilot includes cases concerning:

* fairness;
* operational robustness;
* accountable decision-making.

Operational monitoring is represented in the criteria framework and is reserved for extension in the main experimental study.

The use of multiple domains is intended to test whether the effect of explicit evaluative governance is specific to one trustworthiness property or observable across distinct evaluative contexts.

## 6. Vignette design

The pilot uses six synthetic but operationally plausible vignettes.

Cases are constructed to represent different evidential states, including:

* relatively broad evidential support;
* missing evidence relevant to attribution;
* incomplete procedural evidence;
* cases capable of receiving subsequent materially relevant evidence.

Vignettes contain factual and contextual information only.

Experimental annotations such as *broad evidence*, *critical evidence missing*, *defeater*, or *expected evidential state* are maintained separately and are never exposed to the evaluating model.

Neutral case identifiers are used to prevent leakage of experimental roles.

## 7. No trustworthiness ground truth

A central methodological choice is the deliberate absence of a predefined gold-standard trustworthiness attribution.

The experiment does not define:

[
Fair=True
]

or:

[
Robust=False.
]

Instead, internal annotations describe properties of the evidential package, such as the presence or absence of particular evidence.

Thus, the study distinguishes:

[
\text{Ground truth about evidence}
]

from:

[
\text{Ground truth about trustworthiness}.
]

Only the former is defined experimentally.

This distinction avoids embedding the researchers' normative judgement into the dependent variable that the experiment is intended to investigate.

## 8. Common attribution output

Every condition uses an identical structured output.

The evaluator provides:

* attribution;
* confidence in the assessment;
* justification;
* evidence used;
* limitations;
* proposed action.

Attribution belongs to one of four categories:

* supported;
* partially supported;
* not supported;
* insufficient basis for attribution.

The procedural action belongs to one of:

* affirm;
* abstain;
* escalate.

Attribution and action are deliberately represented separately.

The design therefore distinguishes what the evaluator concludes from what the evaluator proposes doing in response to that conclusion.

## 9. Epistemic restraint

A central outcome of interest is *epistemic restraint*.

Epistemic restraint concerns the evaluator's willingness to avoid making an attribution stronger than the available evidential basis supports.

It is not operationalised simply as a high abstention rate.

Universal abstention would itself constitute undesirable evaluative behaviour.

The experiment therefore examines restraint conditionally, particularly in cases whose evidential packages contain predefined material limitations.

One derived measure is strong over-attribution:

[
SOA =
\mathbb{1}
(
Attribution=Supported
\land
Action=Affirm
)
]

for cases whose experimental annotation indicates missing critical evidence.

The objective is to determine whether Evaluative Governance reduces unsupported strong attribution without inducing indiscriminate abstention.

## 10. Evidential grounding

Evidential grounding evaluates the quality of the relationship between evidence and attribution rather than whether the attribution matches a preferred answer.

A five-level ordinal rubric is used:

* 0 — ungrounded;
* 1 — superficially grounded;
* 2 — partially grounded;
* 3 — well grounded;
* 4 — critically grounded.

Higher scores require increasingly explicit relationships among evidence, interpretation, limitations, uncertainty, and the scope or strength of the resulting attribution.

Response length alone does not increase the score.

The scorer is blinded to:

* experimental condition;
* generating model;
* repetition;
* experimental role of the case;
* expected evidential state;
* experimental hypotheses.

The scorer receives the vignette, the evaluation material legitimately available to the evaluator, and the generated assessment.

## 11. Revisability

SC5 conceptualises trustworthiness attribution as revisable rather than permanent.

A second experimental round therefore introduces additional information after an initial attribution.

Two update classes are used in the pilot:

### Material defeater

New information that materially weakens part of the evidential basis supporting the previous attribution.

### Non-material control

New information that does not materially change the evidential basis of the previous attribution.

The evaluator is not told which type of update it receives.

The central construct is therefore not revision frequency but **selective revisability**:

[
P(Revision\mid Material)

>

P(Revision\mid NonMaterial).
]

This distinction prevents a generally unstable evaluator from appearing more revisable merely because it frequently changes its judgement.

The main experiment may additionally introduce positive resolving evidence to evaluate appropriate strengthening of previous attributions.

## 12. Model design

Two LLM families from different providers are used as experimental evaluators.

The pilot configuration currently specifies:

* OpenAI `gpt-5.4-mini-2026-03-17`;
* Google Gemini `gemini-3.5-flash`.

Model identity is retained explicitly as an experimental factor.

The models are not assumed to have equivalent capabilities.

Using different model families is intended to determine whether observed effects are specific to a particular model implementation or appear across distinct LLM architectures/providers.

Sampling parameters are intentionally left at the native defaults of each provider. In particular, temperature, top-p, and seed are not explicitly supplied to either API. This avoids imposing provider-specific sampling controls that do not have an equivalent or recommended interpretation across the evaluated model families. The experimental manipulation is therefore evaluated within each model under its native generation regime. A common maximum output length is applied to both models.

Repeated evaluations are retained to empirically characterise within-model output variability under these native generation conditions rather than artificially suppressing stochastic variation through provider-specific controls.

## 13. Pilot design

The primary pilot comprises:

[
6\ cases
\times
3\ conditions
\times
2\ models
\times
2\ repetitions
==============

72\ evaluations.
]

Revalidation runs are treated as a separate experimental round and are not included in these 72 primary evaluations.

The execution order is randomised using a fixed seed.

Each evaluation is independent and receives no previous model response or conversational history.

## 14. Purpose of the pilot

The pilot is an instrument-validation study.

It is not used for confirmatory hypothesis testing.

Its objectives are to examine:

* prompt-format compliance;
* vignette ambiguity;
* leakage of preferred conclusions;
* floor and ceiling effects;
* discriminative capacity of C0/C1/C2;
* applicability of the grounding rubric;
* epistemic restraint;
* selective revisability;
* intra-model stability;
* cross-model behaviour.

Pilot results will lead to one of three decisions:

**GO** — the instrument is sufficiently interpretable and discriminative for the main study.

**REVISE** — the construct appears measurable but parts of the instrument require modification.

**STOP/REDESIGN** — the manipulation does not produce an interpretable experimental phenomenon.

## 15. Avoiding circular validation

The study deliberately minimises dependence on LLM-as-a-Judge evaluation.

Primary outcomes are preferentially obtained from:

1. directly structured model outputs;
2. deterministic transformations of those outputs;
3. predefined annotations concerning the evidential packages;
4. blinded semantic assessment only where necessary.

LLM-based semantic scoring may subsequently be evaluated as a scalability mechanism, but it should not constitute the sole ground truth for validating LLM-generated trustworthiness judgements.

Human annotation of evidential grounding should therefore precede or calibrate automated semantic scoring.

## 16. Reproducibility and provenance

The experimental implementation maintains explicit provenance between:

[
Case
\rightarrow
Condition
\rightarrow
Prompt
\rightarrow
Prompt\ Hash
\rightarrow
Model
\rightarrow
Run
\rightarrow
Raw\ Output
\rightarrow
Parsed\ Output
\rightarrow
Score.
]

Prompts are generated deterministically.

A SHA-256 hash identifies each complete prompt.

The execution manifest contains the experimental combination of case, condition, model slot, repetition, prompt hash, and execution status.

Model identifiers, provider, token usage, timestamps, and generation parameters are recorded as run metadata.

The software environment is frozen using an explicit dependency lock file.

## 17. Separation of pilot and technical testing

Provider connectivity tests and software-development runs are explicitly separated from experimental runs.

Micro-runs:

* use dedicated technical prompts;
* are stored outside experimental run directories;
* do not consume experimental run identifiers;
* do not modify the experimental manifest.

Consequently, API validation cannot inadvertently become part of the pilot dataset.

## 18. Planned analysis

The primary inferential contrast for the main experiment is:

[
C_2-C_1,
]

representing the incremental effect of Evaluative Governance after controlling for explicit evaluation criteria.

Additional contrasts include:

[
C_1-C_0
]

and:

[
C_2-C_0.
]

Outcomes include:

* attribution distribution;
* epistemic restraint;
* strong over-attribution;
* abstention;
* escalation;
* evidential grounding;
* limitation recognition;
* unsupported evidence use;
* intra-model stability;
* inter-model agreement;
* selective revisability.

Model identity is treated explicitly in the analysis.

Vignettes constitute the substantive experimental units; repeated LLM generations are not treated as independent substantive cases.

This distinction is intended to avoid pseudoreplication.

## 19. Interpretation boundary

The experiment is designed to support a bounded claim.

It can provide evidence that explicit evaluative governance conditions affect the process through which technical evidence is converted into trustworthiness attribution.

It does not by itself demonstrate:

* that a particular trustworthiness attribution is objectively correct;
* that SC5 guarantees legitimate governance;
* that LLM evaluators should replace human institutional judgement;
* that the experimental governance procedure exhausts the requirements of real-world governance;
* that observed effects automatically generalise beyond the evaluated domains and models.

The intended contribution is narrower:

**to empirically examine whether the explicit institutionalisation of evaluative conditions changes the epistemic behaviour of trustworthiness attribution while holding the underlying technical evidence constant.**

## 20. Core methodological contribution

The experiment operationalises the conceptual transition:

[
\text{Technical Evidence}
\not\equiv
\text{Trustworthiness Attribution}
]

and tests whether an explicit evaluative layer affects that transition.

This provides an empirical complement to SC5 by moving the argument from a purely conceptual claim about the need for Evaluative Governance toward an experimentally observable question about how governance conditions affect attribution behaviour
## Pilot instrumentation revision v1.1

An initial smoke-test execution of pilot v1.0 identified a provider-specific
output-formatting problem. Three Gemini executions produced invalid JSON,
including truncated output and free-form text preceding the requested
structured response.

The issue was treated as an instrumentation failure rather than an
experimental outcome. No substantive interpretation of these runs was
performed.

Pilot v1.1 therefore introduces provider-level structured output enforcement
for both evaluated model families using the same experimental JSON schema.

For Gemini 3.5 Flash, the thinking level is additionally set to `low` to
prevent the default reasoning budget from interfering with completion of the
structured response. Sampling parameters remain at the native provider
defaults.

The revised configuration was validated through independent provider
micro-runs. Both OpenAI and Gemini produced complete JSON responses conforming
to the experimental output schema before pilot execution was restarted.

The six v1.0 smoke-test runs are retained as instrumentation diagnostics and
are excluded from the v1.1 experimental dataset.
