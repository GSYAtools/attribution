from src.build_revision_prompts import (
    REVISION_CASES,
    REVISION_CONDITIONS,
    UPDATE_TYPES,
    build_revision_user_prompt,
    get_updates_for_case,
    get_vignette,
    load_revision_materials,
)


def test_expected_revision_cases_and_updates():
    assert REVISION_CASES == {
        "PX_1005",
        "PX_1006",
    }

    assert REVISION_CONDITIONS == {
        "C0",
        "C1",
        "C2",
    }

    assert UPDATE_TYPES == (
        "material_defeater",
        "non_material_control",
    )


def test_each_revision_case_has_two_updates():
    materials = load_revision_materials()

    for case_id in REVISION_CASES:
        updates = get_updates_for_case(
            materials,
            case_id,
        )

        assert updates["base_case"] == case_id
        assert "domain" in updates

        assert set(
            key
            for key in updates.keys()
            if key in {
                "material_defeater",
                "non_material_control",
            }
        ) == {
            "material_defeater",
            "non_material_control",
        }

        assert "text" in updates["material_defeater"]
        assert "text" in updates["non_material_control"]

def test_update_labels_are_not_exposed_in_prompt():
    materials = load_revision_materials()

    initial_assessment = {
        "attribution": "supported",
        "confidence": 80,
        "justification": "Initial assessment for testing.",
        "evidence_used": ["Evidence A."],
        "limitations": ["Limitation A."],
        "action": "affirm",
    }

    for case_id in REVISION_CASES:
        vignette = get_vignette(
            materials,
            case_id,
        )

        updates = get_updates_for_case(
            materials,
            case_id,
        )

        for condition in REVISION_CONDITIONS:
            for update_type in UPDATE_TYPES:

                prompt = build_revision_user_prompt(
                    vignette=vignette,
                    condition=condition,
                    initial_assessment=initial_assessment,
                    update_text=updates[update_type]["text"],
                    materials=materials,
                )

                prompt_lower = prompt.lower()

                assert "material_defeater" not in prompt_lower
                assert "non_material_control" not in prompt_lower
                assert "update_id" not in prompt_lower


def test_update_text_is_present():
    materials = load_revision_materials()

    initial_assessment = {
        "attribution": "partially_supported",
        "confidence": 60,
        "justification": "Initial assessment for testing.",
        "evidence_used": ["Evidence A."],
        "limitations": ["Limitation A."],
        "action": "abstain",
    }

    for case_id in REVISION_CASES:
        vignette = get_vignette(
            materials,
            case_id,
        )

        updates = get_updates_for_case(
            materials,
            case_id,
        )

        for condition in REVISION_CONDITIONS:
            for update_type in UPDATE_TYPES:

                update_text = updates[update_type]["text"]

                prompt = build_revision_user_prompt(
                    vignette=vignette,
                    condition=condition,
                    initial_assessment=initial_assessment,
                    update_text=update_text,
                    materials=materials,
                )

                assert update_text in prompt


def test_complete_initial_assessment_is_preserved():
    materials = load_revision_materials()

    initial_assessment = {
        "attribution": "partially_supported",
        "confidence": 63,
        "justification": (
            "The evidence supports part of the assessment "
            "but does not resolve all limitations."
        ),
        "evidence_used": [
            "Evidence A",
            "Evidence B",
        ],
        "limitations": [
            "Important evidence is incomplete.",
        ],
        "action": "escalate",
    }

    vignette = get_vignette(
        materials,
        "PX_1005",
    )

    updates = get_updates_for_case(
        materials,
        "PX_1005",
    )

    prompt = build_revision_user_prompt(
        vignette=vignette,
        condition="C2",
        initial_assessment=initial_assessment,
        update_text=updates["material_defeater"]["text"],
        materials=materials,
    )

    for key, value in initial_assessment.items():
        if isinstance(value, list):
            for item in value:
                assert item in prompt
        else:
            assert str(value) in prompt


def test_condition_controls_evaluation_context():
    materials = load_revision_materials()

    initial_assessment = {
        "attribution": "insufficient_basis",
        "confidence": 50,
        "justification": "Initial assessment.",
        "evidence_used": ["Evidence A."],
        "limitations": ["Limitation A."],
        "action": "abstain",
    }

    vignette = get_vignette(
        materials,
        "PX_1005",
    )

    updates = get_updates_for_case(
        materials,
        "PX_1005",
    )

    prompts = {}

    for condition in REVISION_CONDITIONS:
        prompts[condition] = build_revision_user_prompt(
            vignette=vignette,
            condition=condition,
            initial_assessment=initial_assessment,
            update_text=updates["material_defeater"]["text"],
            materials=materials,
        )

    assert prompts["C0"] != prompts["C1"]
    assert prompts["C1"] != prompts["C2"]

    criteria_text = materials["criteria"]["cards"][
        "robustness"
    ]["text"]

    governance_text = materials["governance"]

    assert criteria_text not in prompts["C0"]
    assert governance_text not in prompts["C0"]

    assert criteria_text in prompts["C1"]
    assert governance_text not in prompts["C1"]

    assert criteria_text in prompts["C2"]
    assert governance_text in prompts["C2"]
