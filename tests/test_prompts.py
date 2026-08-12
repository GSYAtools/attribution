from src.build_prompts import (
    build_all_prompts,
    build_user_prompt,
    get_vignettes,
    load_experiment_materials,
)


def get_case(materials, case_id):
    for vignette in get_vignettes(materials):
        if vignette["case_id"] == case_id:
            return vignette

    raise ValueError(f"Case '{case_id}' not found.")


def test_expected_number_of_prompts():
    records = build_all_prompts()

    assert len(records) == 18


def test_all_prompt_hashes_are_unique():
    records = build_all_prompts()

    hashes = [record["prompt_hash"] for record in records]

    assert len(hashes) == len(set(hashes))


def test_all_cases_have_three_conditions():
    records = build_all_prompts()

    cases = {}

    for record in records:
        cases.setdefault(record["case_id"], set())
        cases[record["case_id"]].add(record["condition"])

    assert len(cases) == 6

    for conditions in cases.values():
        assert conditions == {"C0", "C1", "C2"}


def test_c0_does_not_include_criteria_or_governance():
    materials = load_experiment_materials()
    vignette = get_case(materials, "PX_1001")

    prompt = build_user_prompt(
        vignette,
        "C0",
        materials,
    )

    criteria_text = materials["criteria"]["cards"]["fairness"]["text"]
    governance_text = materials["governance"]

    assert criteria_text not in prompt
    assert governance_text not in prompt


def test_c1_adds_criteria_but_not_governance():
    materials = load_experiment_materials()
    vignette = get_case(materials, "PX_1001")

    prompt = build_user_prompt(
        vignette,
        "C1",
        materials,
    )

    criteria_text = materials["criteria"]["cards"]["fairness"]["text"]
    governance_text = materials["governance"]

    assert criteria_text in prompt
    assert governance_text not in prompt


def test_c2_adds_criteria_and_governance():
    materials = load_experiment_materials()
    vignette = get_case(materials, "PX_1001")

    prompt = build_user_prompt(
        vignette,
        "C2",
        materials,
    )

    criteria_text = materials["criteria"]["cards"]["fairness"]["text"]
    governance_text = materials["governance"]

    assert criteria_text in prompt
    assert governance_text in prompt


def test_vignette_is_identical_across_conditions():
    materials = load_experiment_materials()
    vignette = get_case(materials, "PX_1002")

    evidence_items = vignette["evidence"]

    for condition in ("C0", "C1", "C2"):
        prompt = build_user_prompt(
            vignette,
            condition,
            materials,
        )

        for evidence in evidence_items:
            assert evidence in prompt


def test_internal_metadata_is_not_leaked():
    records = build_all_prompts()

    forbidden_terms = [
        "critical_evidence_missing",
        "broad_evidence_present",
        "broad_procedural_evidence_present",
        "procedural_evidence_incomplete",
        "material_defeater",
        "non_material_control",
        "expected_evidential_state",
    ]

    for record in records:
        combined = (
            record["system_prompt"]
            + "\n"
            + record["user_prompt"]
        ).lower()

        for term in forbidden_terms:
            assert term.lower() not in combined
