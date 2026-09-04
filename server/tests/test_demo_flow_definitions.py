import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "deploy" / "ensure_default_flow.py"
SPEC = importlib.util.spec_from_file_location("ensure_default_flow", MODULE_PATH)
FLOW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FLOW)


def test_ten_step_flow_uses_the_three_supported_capabilities_in_order():
    steps = FLOW.ten_step_steps()

    assert len(steps) == 10
    assert [step["type"] for step in steps] == [
        "QR",
        "FASTENER_MARK",
        "METER",
        "FASTENER_MARK",
        "QR",
        "FASTENER_MARK",
        "METER",
        "FASTENER_MARK",
        "METER",
        "FASTENER_MARK",
    ]
    assert len({step["key"] for step in steps}) == 10


def test_every_demo_step_requires_evidence_and_manual_review_where_needed():
    for step in FLOW.default_steps() + FLOW.ten_step_steps():
        assert step["required"] is True
        assert step["require_evidence"] is True
        assert step["require_human_confirmation"] is (step["type"] != "QR")


def test_demo_assignments_bind_three_and_ten_step_templates():
    assert FLOW.demo_assignments() == [
        {"asset_code": "CRRC-DEMO-001", "template_code": "CRRC_THREE_STEP"},
        {"asset_code": "CRRC-DEMO-010", "template_code": "CRRC_TEN_STEP"},
    ]


def test_matching_template_and_active_assignment_are_reused(monkeypatch):
    definition = FLOW.template_definitions()[1]
    template = {"id": 20, "version": 1, **definition}
    assignment = {
        "id": 30,
        "asset_code": "CRRC-DEMO-010",
        "status": "pending",
        "template_id": 20,
    }

    def unexpected_request(*args, **kwargs):
        raise AssertionError("idempotent path must not create another record")

    monkeypatch.setattr(FLOW, "request_json", unexpected_request)

    assert FLOW.ensure_template("token", [template], definition) is template
    assert FLOW.ensure_assignment(
        "token",
        [assignment],
        FLOW.demo_assignments()[1],
        template,
        1,
    ) is assignment


def test_missing_assignment_is_created_for_latest_template(monkeypatch):
    recorded = {}

    def create_request(path, *, token=None, payload=None):
        recorded.update(path=path, token=token, payload=payload)
        return {"id": 31, "status": "pending", **payload}

    monkeypatch.setattr(FLOW, "request_json", create_request)
    created = FLOW.ensure_assignment(
        "secret-token",
        [],
        FLOW.demo_assignments()[1],
        {"id": 20},
        7,
    )

    assert created["id"] == 31
    assert recorded == {
        "path": "/api/v1/assignments",
        "token": "secret-token",
        "payload": {
            "template_id": 20,
            "assignee_id": 7,
            "asset_code": "CRRC-DEMO-010",
        },
    }
