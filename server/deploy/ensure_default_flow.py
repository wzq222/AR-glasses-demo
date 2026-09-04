"""Idempotently publish the CRRC three- and ten-step demo workflows."""

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


def qr_step(key: str, title: str, instruction: str) -> dict:
    return {
        "key": key,
        "type": "QR",
        "title": title,
        "instruction": instruction,
        "required": True,
        "require_evidence": True,
        "require_human_confirmation": False,
        "config": {
            "analyzer": "barcode-v1",
            "capture_source": "BOTH",
            "failure_action": "RETRY",
        },
    }


def fastener_step(key: str, title: str, instruction: str) -> dict:
    return {
        "key": key,
        "type": "FASTENER_MARK",
        "title": title,
        "instruction": instruction,
        "required": True,
        "require_evidence": True,
        "require_human_confirmation": True,
        "config": {
            "analyzer": "marked-point-v1",
            "capture_source": "BOTH",
            "allowedValues": ["ALIGNED", "SUSPECTED", "UNABLE_TO_JUDGE"],
            "failure_action": "MANUAL_REVIEW",
        },
    }


def meter_step(key: str, title: str, instruction: str) -> dict:
    return {
        "key": key,
        "type": "METER",
        "title": title,
        "instruction": instruction,
        "required": True,
        "require_evidence": True,
        "require_human_confirmation": True,
        "config": {
            "analyzer": "meter-ocr-v1",
            "capture_source": "BOTH",
            "failure_action": "MANUAL_REVIEW",
        },
    }


def default_steps() -> list[dict]:
    return [
        qr_step("QR_CHECK", "二维码点位确认", "扫描设备二维码，自动解析点位并核对任务设备"),
        fastener_step(
            "FASTENER_CHECK",
            "防松标记检测",
            "拍摄检查区域，检测带红黄防松标记的紧固点并逐项确认状态",
        ),
        meter_step("METER_CHECK", "万用表读数复核", "拍摄万用表屏幕，识别数值与单位并由作业人员确认"),
    ]


def ten_step_steps() -> list[dict]:
    return [
        qr_step("QR_VEHICLE_STATION", "车辆/工位确认", "扫描车辆或工位二维码，核对本次巡检位置"),
        fastener_step("FASTENER_A", "防松线检查点 A", "拍摄检查点 A，逐个审核已定位的防松线状态"),
        meter_step("METER_A", "仪表 A 读数", "拍摄仪表 A 屏幕，识别读数并人工复核"),
        fastener_step("FASTENER_B", "防松线检查点 B", "拍摄检查点 B，逐个审核已定位的防松线状态"),
        qr_step("QR_CABINET", "设备柜确认", "扫描设备柜二维码，核对柜体与任务是否一致"),
        fastener_step("FASTENER_C", "防松线检查点 C", "拍摄检查点 C，逐个审核已定位的防松线状态"),
        meter_step("METER_B", "仪表 B 读数", "拍摄仪表 B 屏幕，识别读数并人工复核"),
        fastener_step("FASTENER_D", "防松线检查点 D", "拍摄检查点 D，逐个审核已定位的防松线状态"),
        meter_step("METER_C", "仪表 C 读数", "拍摄仪表 C 屏幕，识别读数并人工复核"),
        fastener_step("FASTENER_E", "防松线检查点 E", "拍摄检查点 E，逐个审核已定位的防松线状态"),
    ]


def template_definitions() -> list[dict]:
    return [
        {
            "code": "CRRC_THREE_STEP",
            "title": "中车三步巡检",
            "description": "二维码、防松标记、万用表固定巡检流程",
            "steps": default_steps(),
        },
        {
            "code": "CRRC_TEN_STEP",
            "title": "中车十步巡检",
            "description": "二维码、防松标记与万用表交错执行的完整巡检流程",
            "steps": ten_step_steps(),
        },
    ]


def demo_assignments() -> list[dict]:
    return [
        {"asset_code": "CRRC-DEMO-001", "template_code": "CRRC_THREE_STEP"},
        {"asset_code": "CRRC-DEMO-010", "template_code": "CRRC_TEN_STEP"},
    ]


def ensure_template(token: str, templates: list[dict], definition: dict) -> dict:
    matching = [item for item in templates if item["code"] == definition["code"]]
    latest = max(matching, key=lambda item: item["version"]) if matching else None
    if latest and latest["steps"] == definition["steps"]:
        print(f"template ready: {definition['code']} v{latest['version']}")
        return latest
    created = request_json("/api/v1/sop/templates", token=token, payload=definition)
    templates.append(created)
    print(f"template published: {definition['code']} v{created['version']}")
    return created


def ensure_assignment(
    token: str,
    assignments: list[dict],
    assignment: dict,
    template: dict,
    assignee_id: int,
) -> dict:
    matching = [
        item
        for item in assignments
        if item["asset_code"] == assignment["asset_code"]
        and item["status"] in {"pending", "in_progress"}
    ]
    if matching:
        current = matching[0]
        print(f"assignment ready: {assignment['asset_code']} id={current['id']}")
        return current
    created = request_json(
        "/api/v1/assignments",
        token=token,
        payload={
            "template_id": template["id"],
            "assignee_id": assignee_id,
            "asset_code": assignment["asset_code"],
        },
    )
    assignments.append(created)
    print(f"assignment created: {assignment['asset_code']} id={created['id']}")
    return created


def main() -> None:
    password = PASSWORD_FILE.read_text(encoding="utf-8").strip()
    login = request_json(
        "/api/v1/auth/login", payload={"username": "admin", "password": password}
    )
    token = login["access_token"]
    users = request_json("/api/v1/users", token=token)
    admin = next(user for user in users if user["username"] == "admin" and user["active"])
    templates = request_json("/api/v1/sop/templates", token=token)
    ensured_templates = {
        definition["code"]: ensure_template(token, templates, definition)
        for definition in template_definitions()
    }
    assignments = request_json("/api/v1/assignments", token=token)
    for assignment in demo_assignments():
        ensure_assignment(
            token,
            assignments,
            assignment,
            ensured_templates[assignment["template_code"]],
            admin["id"],
        )


if __name__ == "__main__":
    main()
