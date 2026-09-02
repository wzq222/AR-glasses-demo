"""Publish the enriched CRRC three-step workflow once on an existing deployment."""

import json
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:18081"
PASSWORD_FILE = Path("/root/crrc-sop-admin-password.txt")


def request_json(path: str, *, token: str | None = None, payload: dict | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    method = "POST" if payload is not None else "GET"
    with urllib.request.urlopen(
        urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method),
        timeout=15,
    ) as response:
        return json.load(response)


def default_steps() -> list[dict]:
    return [
        {
            "key": "QR_CHECK",
            "type": "QR",
            "title": "二维码点位确认",
            "instruction": "扫描设备二维码，自动解析点位并核对任务设备",
            "required": True,
            "require_evidence": True,
            "require_human_confirmation": False,
            "config": {"analyzer": "barcode-v1", "capture_source": "BOTH", "failure_action": "RETRY"},
        },
        {
            "key": "FASTENER_CHECK",
            "type": "FASTENER_MARK",
            "title": "防松标记检测",
            "instruction": "拍摄检查区域，检测带红黄防松标记的紧固点并逐项确认状态",
            "required": True,
            "require_evidence": True,
            "require_human_confirmation": True,
            "config": {
                "analyzer": "marked-point-v1",
                "capture_source": "BOTH",
                "allowedValues": ["ALIGNED", "SUSPECTED", "UNABLE_TO_JUDGE"],
                "failure_action": "MANUAL_REVIEW",
            },
        },
        {
            "key": "METER_CHECK",
            "type": "METER",
            "title": "万用表读数复核",
            "instruction": "拍摄万用表屏幕，识别数值与单位并由作业人员确认",
            "required": True,
            "require_evidence": True,
            "require_human_confirmation": True,
            "config": {"analyzer": "meter-ocr-v1", "capture_source": "BOTH", "failure_action": "MANUAL_REVIEW"},
        },
    ]


def main() -> None:
    password = PASSWORD_FILE.read_text(encoding="utf-8").strip()
    login = request_json("/api/v1/auth/login", payload={"username": "admin", "password": password})
    token = login["access_token"]
    templates = request_json("/api/v1/sop/templates", token=token)
    matching = [item for item in templates if item["code"] == "CRRC_THREE_STEP"]
    latest = max(matching, key=lambda item: item["version"]) if matching else None
    expected = default_steps()
    if latest and latest["steps"] == expected:
        print(f"default flow already ready: v{latest['version']}")
        return
    created = request_json(
        "/api/v1/sop/templates",
        token=token,
        payload={
            "code": "CRRC_THREE_STEP",
            "title": "中车三步巡检",
            "description": "二维码、防松标记、万用表固定巡检流程",
            "steps": expected,
        },
    )
    print(f"default flow published: v{created['version']}")


if __name__ == "__main__":
    main()
