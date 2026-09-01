import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        secret_key="test-secret-key-that-is-long-enough-123456",
        bootstrap_admin_username="admin",
        bootstrap_admin_password="correct-horse-battery-staple",
        database_path=tmp_path / "test.sqlite3",
        evidence_dir=tmp_path / "evidence",
    )
    return TestClient(create_app(settings))


def auth(c: TestClient, username="admin", password="correct-horse-battery-staple") -> dict:
    response = c.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def seed(c: TestClient):
    admin = auth(c)
    user = c.post(
        "/api/v1/users",
        headers=admin,
        json={"username": "worker1", "display_name": "巡检员一", "password": "worker-password-123", "role": "inspector"},
    )
    assert user.status_code == 201, user.text
    template = c.post(
        "/api/v1/sop/templates",
        headers=admin,
        json={
            "code": "CRRC_THREE_STEP",
            "title": "中车三步巡检",
            "steps": [
                {"key": "QR_CHECK", "type": "QR", "title": "二维码打卡", "instruction": "拍摄设备二维码"},
                {"key": "FASTENER_CHECK", "type": "FASTENER_MARK", "title": "防松标记", "instruction": "拍摄并确认防松标记", "require_human_confirmation": True},
                {"key": "METER_CHECK", "type": "METER", "title": "万用表读数", "instruction": "拍摄万用表屏幕"},
            ],
        },
    )
    assert template.status_code == 201, template.text
    assignment = c.post(
        "/api/v1/assignments",
        headers=admin,
        json={"template_id": template.json()["id"], "assignee_id": user.json()["id"], "asset_code": "TRAIN-CAR-001"},
    )
    assert assignment.status_code == 201, assignment.text
    return admin, user.json(), assignment.json()


def test_health_and_login(tmp_path):
    with client(tmp_path) as c:
        root = c.get("/", follow_redirects=False)
        assert root.status_code == 200
        assert "中车眼镜巡检后台" in root.text
        assert root.headers["cache-control"] == "no-store, max-age=0"
        admin_page = c.get("/admin")
        assert admin_page.status_code == 200
        assert "中车眼镜巡检后台" in admin_page.text
        assert c.get("/healthz").json()["status"] == "ok"
        assert c.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong-password"}).status_code == 401
        assert c.get("/api/v1/users/me", headers=auth(c)).json()["role"] == "admin"


def test_management_dashboard_and_role_guards(tmp_path):
    with client(tmp_path) as c:
        admin, user, _ = seed(c)
        dashboard = c.get("/api/v1/dashboard", headers=admin)
        assert dashboard.status_code == 200
        assert dashboard.json()["users"] == 2
        assert dashboard.json()["pending_assignments"] == 1

        users = c.get("/api/v1/users", headers=admin)
        assert users.status_code == 200
        assert {item["username"] for item in users.json()} == {"admin", "worker1"}
        assert all("password_hash" not in item for item in users.json())

        worker = auth(c, "worker1", "worker-password-123")
        assert c.get("/api/v1/dashboard", headers=worker).status_code == 403
        assert c.get("/api/v1/users", headers=worker).status_code == 403
        own_runs = c.get("/api/v1/runs", headers=worker)
        assert own_runs.status_code == 200
        assert own_runs.json() == []


def test_inspector_only_sees_own_assignments(tmp_path):
    with client(tmp_path) as c:
        _, _, assignment = seed(c)
        worker = auth(c, "worker1", "worker-password-123")
        response = c.get("/api/v1/assignments", headers=worker)
        assert response.status_code == 200
        assert [item["id"] for item in response.json()] == [assignment["id"]]
        assert [step["key"] for step in response.json()[0]["steps"]] == ["QR_CHECK", "FASTENER_CHECK", "METER_CHECK"]


def test_complete_sop_requires_steps_evidence_and_human_decision(tmp_path):
    with client(tmp_path) as c:
        admin, _, assignment = seed(c)
        worker = auth(c, "worker1", "worker-password-123")
        run = c.post("/api/v1/runs", headers=worker, json={"assignment_id": assignment["id"], "device": {"source": "PHONE"}})
        assert run.status_code == 201
        run_id = run.json()["id"]
        assert c.post(f"/api/v1/runs/{run_id}/submit", headers=worker).status_code == 409

        for index, key in enumerate(("QR_CHECK", "FASTENER_CHECK", "METER_CHECK")):
            payload = {
                "idempotency_key": f"device-run-step-{index}",
                "status": "succeeded",
                "value": {"value": "ok"},
                "confidence": 0.91,
                "requires_human_review": key == "FASTENER_CHECK",
                "human_decision": "confirmed_aligned" if key == "FASTENER_CHECK" else None,
                "analyzer_version": "test-v1",
                "captured_at": "2026-09-01T12:00:00+08:00",
            }
            first = c.put(f"/api/v1/runs/{run_id}/steps/{key}", headers=worker, json=payload)
            assert first.status_code == 200, first.text
            replay = c.put(f"/api/v1/runs/{run_id}/steps/{key}", headers=worker, json=payload)
            assert replay.status_code == 200
            evidence = c.post(
                f"/api/v1/runs/{run_id}/steps/{key}/evidence",
                headers=worker,
                files={"file": (f"{key}.jpg", io.BytesIO(b"fake-jpeg-evidence"), "image/jpeg")},
            )
            assert evidence.status_code == 201, evidence.text

        submitted = c.post(f"/api/v1/runs/{run_id}/submit", headers=worker)
        assert submitted.status_code == 200, submitted.text
        reviewed = c.post(f"/api/v1/runs/{run_id}/review", headers=admin, json={"decision": "reviewed", "note": "证据完整"})
        assert reviewed.status_code == 200, reviewed.text


def test_role_and_evidence_guards(tmp_path):
    with client(tmp_path) as c:
        _, _, assignment = seed(c)
        worker = auth(c, "worker1", "worker-password-123")
        assert c.post("/api/v1/users", headers=worker, json={"username": "bad", "display_name": "x", "password": "long-password-123", "role": "admin"}).status_code == 403
        run = c.post("/api/v1/runs", headers=worker, json={"assignment_id": assignment["id"]}).json()
        assert c.post(f"/api/v1/runs/{run['id']}/steps/QR_CHECK/evidence", headers=worker, files={"file": ("x.txt", b"x", "text/plain")}).status_code == 415
