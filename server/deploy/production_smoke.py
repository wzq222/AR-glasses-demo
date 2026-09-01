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
assert status == 200 and any(item["code"] == "CRRC_THREE_STEP" for item in templates)
print("authenticated smoke passed: login, current user, seeded three-step SOP")
