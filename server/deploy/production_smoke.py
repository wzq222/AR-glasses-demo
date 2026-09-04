"""Authenticated production smoke without printing credentials or tokens."""

import json
import os
import urllib.request


base = os.environ.get("CRRC_SMOKE_BASE", "http://127.0.0.1:18081").rstrip("/")
password = open("/root/crrc-sop-admin-password.txt", encoding="utf-8").read().strip()


def request(path: str, *, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=10) as response:
        return response.status, json.load(response)


status, login = request("/api/v1/auth/login", payload={"username": "admin", "password": password})
assert status == 200 and login["token_type"] == "bearer"
token = login["access_token"]
status, me = request("/api/v1/users/me", token=token)
assert status == 200 and me["role"] == "admin"
status, templates = request("/api/v1/sop/templates", token=token)
assert status == 200
template_codes = {item["code"] for item in templates}
assert {"CRRC_THREE_STEP", "CRRC_TEN_STEP"}.issubset(template_codes)
status, assignments = request("/api/v1/assignments", token=token)
assert status == 200
active = {
    item["asset_code"]: item
    for item in assignments
    if item["status"] in {"pending", "in_progress"}
}
assert len(active["CRRC-DEMO-001"]["steps"]) == 3
ten_step = active["CRRC-DEMO-010"]["steps"]
assert [step["type"] for step in ten_step] == [
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
print("authenticated smoke passed: public API, 3-step task, 10-step task")
