import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .auth import create_token, current_user_dependency, hash_password, verify_password
from .database import Database, now_iso
from .schemas import (
    AssignmentCreate,
    LoginRequest,
    ReviewRequest,
    RunCreate,
    StepResultUpsert,
    TemplateCreate,
    UserCreate,
)
from .settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    database = Database(settings.database_path)
    current_user = current_user_dependency(settings, database)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.migrate()
        settings.evidence_dir.mkdir(parents=True, exist_ok=True)
        if settings.bootstrap_admin_password:
            with database.connect() as db:
                admin = db.execute("SELECT id FROM users WHERE username=?", (settings.bootstrap_admin_username,)).fetchone()
                if not admin:
                    cursor = db.execute(
                        "INSERT INTO users(username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                        (settings.bootstrap_admin_username, "系统管理员", hash_password(settings.bootstrap_admin_password), "admin", now_iso()),
                    )
                    admin_id = cursor.lastrowid
                else:
                    admin_id = admin["id"]
                if not db.execute("SELECT id FROM sop_templates WHERE code='CRRC_THREE_STEP'").fetchone():
                    default_steps = [
                        {"key": "QR_CHECK", "type": "QR", "title": "二维码点位确认", "instruction": "扫描设备二维码，自动解析点位并核对任务设备", "required": True, "require_evidence": True, "require_human_confirmation": False, "config": {"analyzer": "barcode-v1", "capture_source": "BOTH", "failure_action": "RETRY"}},
                        {"key": "FASTENER_CHECK", "type": "FASTENER_MARK", "title": "防松标记检测", "instruction": "拍摄检查区域，检测带红黄防松标记的紧固点并逐项确认状态", "required": True, "require_evidence": True, "require_human_confirmation": True, "config": {"analyzer": "marked-point-v1", "capture_source": "BOTH", "allowedValues": ["ALIGNED", "SUSPECTED", "UNABLE_TO_JUDGE"], "failure_action": "MANUAL_REVIEW"}},
                        {"key": "METER_CHECK", "type": "METER", "title": "万用表读数复核", "instruction": "拍摄万用表屏幕，识别数值与单位并由作业人员确认", "required": True, "require_evidence": True, "require_human_confirmation": True, "config": {"analyzer": "meter-ocr-v1", "capture_source": "BOTH", "failure_action": "MANUAL_REVIEW"}},
                    ]
                    db.execute(
                        "INSERT INTO sop_templates(code,version,title,description,steps_json,created_by,created_at) VALUES('CRRC_THREE_STEP',1,?,?,?,?,?)",
                        ("中车三步巡检", "二维码、防松标记、万用表固定演示流程", Database.json(default_steps), admin_id, now_iso()),
                    )
        yield

    app = FastAPI(
        title="中车眼镜 SOP Server",
        version="0.2.0",
        root_path=settings.root_path,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database

    def require(user: dict, *roles: str) -> None:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="insufficient role")

    def audit(db, actor_id: int, action: str, entity_type: str, entity_id: str, detail: dict | None = None):
        db.execute(
            "INSERT INTO audit_events(actor_id,action,entity_type,entity_id,detail_json,created_at) VALUES(?,?,?,?,?,?)",
            (actor_id, action, entity_type, entity_id, Database.json(detail or {}), now_iso()),
        )

    @app.get("/healthz")
    def healthz():
        with database.connect() as db:
            db.execute("SELECT 1").fetchone()
        return {"status": "ok", "service": "crrc-sop", "version": app.version}

    def admin_page() -> HTMLResponse:
        page = Path(__file__).with_name("static") / "admin.html"
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root():
        return admin_page()

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_console():
        return admin_page()

    @app.post("/api/v1/auth/login")
    def login(request: LoginRequest):
        with database.connect() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (request.username,)).fetchone()
        user = Database.row(row)
        if not user or not user["active"] or not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid credentials")
        return {"access_token": create_token(settings, user), "token_type": "bearer", "user": {k: user[k] for k in ("id", "username", "display_name", "role")}}

    @app.get("/api/v1/users/me")
    def me(user: Annotated[dict, Depends(current_user)]):
        return user

    @app.post("/api/v1/users", status_code=201)
    def create_user(request: UserCreate, user: Annotated[dict, Depends(current_user)]):
        require(user, "admin")
        try:
            with database.connect() as db:
                cursor = db.execute(
                    "INSERT INTO users(username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (request.username, request.display_name, hash_password(request.password), request.role, now_iso()),
                )
                audit(db, user["id"], "create", "user", str(cursor.lastrowid), {"role": request.role})
                user_id = cursor.lastrowid
        except Exception as exc:
            if "UNIQUE" in str(exc):
                raise HTTPException(status_code=409, detail="username already exists") from None
            raise
        return {"id": user_id, **request.model_dump(exclude={"password"}), "active": True}

    @app.get("/api/v1/users")
    def list_users(user: Annotated[dict, Depends(current_user)]):
        require(user, "admin", "reviewer")
        with database.connect() as db:
            rows = db.execute(
                "SELECT id,username,display_name,role,active,created_at FROM users ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    @app.post("/api/v1/sop/templates", status_code=201)
    def create_template(request: TemplateCreate, user: Annotated[dict, Depends(current_user)]):
        require(user, "admin")
        with database.connect() as db:
            latest = db.execute("SELECT COALESCE(MAX(version),0) FROM sop_templates WHERE code=?", (request.code,)).fetchone()[0]
            version = latest + 1
            cursor = db.execute(
                "INSERT INTO sop_templates(code,version,title,description,steps_json,created_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (request.code, version, request.title, request.description, Database.json([s.model_dump() for s in request.steps]), user["id"], now_iso()),
            )
            audit(db, user["id"], "create", "sop_template", str(cursor.lastrowid), {"code": request.code, "version": version})
            template_id = cursor.lastrowid
        return {"id": template_id, "version": version, **request.model_dump()}

    @app.get("/api/v1/sop/templates")
    def list_templates(user: Annotated[dict, Depends(current_user)]):
        with database.connect() as db:
            rows = db.execute("SELECT * FROM sop_templates WHERE active=1 ORDER BY code,version DESC").fetchall()
        return [{**dict(row), "steps": json.loads(row["steps_json"])} for row in rows]

    @app.post("/api/v1/assignments", status_code=201)
    def create_assignment(request: AssignmentCreate, user: Annotated[dict, Depends(current_user)]):
        require(user, "admin", "reviewer")
        with database.connect() as db:
            if not db.execute("SELECT id FROM sop_templates WHERE id=? AND active=1", (request.template_id,)).fetchone():
                raise HTTPException(status_code=404, detail="template not found")
            assignee = db.execute("SELECT id FROM users WHERE id=? AND active=1", (request.assignee_id,)).fetchone()
            if not assignee:
                raise HTTPException(status_code=404, detail="assignee not found")
            cursor = db.execute(
                "INSERT INTO assignments(template_id,assignee_id,asset_code,status,due_at,created_by,created_at) VALUES(?,?,?,'pending',?,?,?)",
                (request.template_id, request.assignee_id, request.asset_code, request.due_at.isoformat() if request.due_at else None, user["id"], now_iso()),
            )
            audit(db, user["id"], "create", "assignment", str(cursor.lastrowid))
            assignment_id = cursor.lastrowid
        return {"id": assignment_id, "status": "pending", **request.model_dump()}

    @app.get("/api/v1/assignments")
    def list_assignments(user: Annotated[dict, Depends(current_user)]):
        sql = """SELECT a.*,t.code,t.version,t.title,t.steps_json,
                        (SELECT r.status FROM runs r WHERE r.assignment_id=a.id
                         ORDER BY r.started_at DESC LIMIT 1) AS latest_run_status
                 FROM assignments a
                 JOIN sop_templates t ON t.id=a.template_id"""
        params: tuple = ()
        if user["role"] == "inspector":
            sql += " WHERE a.assignee_id=?"
            params = (user["id"],)
        sql += " ORDER BY a.created_at DESC"
        with database.connect() as db:
            rows = db.execute(sql, params).fetchall()
        response = []
        for row in rows:
            item = dict(row)
            if item.pop("latest_run_status", None) == "submitted":
                item["status"] = "submitted"
            item["steps"] = json.loads(row["steps_json"])
            response.append(item)
        return response

    @app.post("/api/v1/runs", status_code=201)
    def start_run(request: RunCreate, user: Annotated[dict, Depends(current_user)]):
        with database.connect() as db:
            assignment = db.execute("SELECT * FROM assignments WHERE id=?", (request.assignment_id,)).fetchone()
            if not assignment:
                raise HTTPException(status_code=404, detail="assignment not found")
            if user["role"] == "inspector" and assignment["assignee_id"] != user["id"]:
                raise HTTPException(status_code=403, detail="assignment belongs to another user")
            existing = db.execute("SELECT * FROM runs WHERE assignment_id=? AND status='in_progress'", (request.assignment_id,)).fetchone()
            if existing:
                return dict(existing)
            submitted = db.execute("SELECT id FROM runs WHERE assignment_id=? AND status='submitted'", (request.assignment_id,)).fetchone()
            if submitted:
                raise HTTPException(status_code=409, detail="assignment is awaiting review")
            if assignment["status"] not in ("pending", "in_progress"):
                raise HTTPException(status_code=409, detail="assignment is not executable")
            run_id = str(uuid.uuid4())
            started_at = now_iso()
            db.execute("INSERT INTO runs(id,assignment_id,operator_id,status,device_json,started_at) VALUES(?,?,?,'in_progress',?,?)", (run_id, request.assignment_id, user["id"], Database.json(request.device), started_at))
            db.execute("UPDATE assignments SET status='in_progress' WHERE id=?", (request.assignment_id,))
            audit(db, user["id"], "start", "run", run_id)
        return {"id": run_id, "assignment_id": request.assignment_id, "status": "in_progress", "started_at": started_at, "device": request.device}

    @app.get("/api/v1/runs")
    def list_runs(user: Annotated[dict, Depends(current_user)]):
        sql = """SELECT r.id,r.assignment_id,r.operator_id,r.status,r.started_at,r.submitted_at,
                        r.reviewed_at,r.review_note,a.asset_code,u.display_name AS operator_name,
                        t.title AS sop_title,
                        (SELECT COUNT(*) FROM step_results s WHERE s.run_id=r.id) AS step_count
                 FROM runs r JOIN assignments a ON a.id=r.assignment_id
                 JOIN users u ON u.id=r.operator_id JOIN sop_templates t ON t.id=a.template_id"""
        params: tuple = ()
        if user["role"] == "inspector":
            sql += " WHERE r.operator_id=?"
            params = (user["id"],)
        sql += " ORDER BY r.started_at DESC LIMIT 200"
        with database.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    @app.get("/api/v1/dashboard")
    def dashboard(user: Annotated[dict, Depends(current_user)]):
        require(user, "admin", "reviewer")
        with database.connect() as db:
            return {
                "users": db.execute("SELECT COUNT(*) FROM users WHERE active=1").fetchone()[0],
                "active_templates": db.execute("SELECT COUNT(*) FROM sop_templates WHERE active=1").fetchone()[0],
                "pending_assignments": db.execute(
                    """SELECT COUNT(*) FROM assignments a
                       WHERE a.status IN ('pending','in_progress')
                       AND NOT EXISTS (
                           SELECT 1 FROM runs r
                           WHERE r.assignment_id=a.id AND r.status='submitted'
                       )"""
                ).fetchone()[0],
                "awaiting_review": db.execute("SELECT COUNT(*) FROM runs WHERE status='submitted'").fetchone()[0],
                "completed_assignments": db.execute("SELECT COUNT(*) FROM assignments WHERE status='completed'").fetchone()[0],
            }

    def authorized_run(db, run_id: str, user: dict):
        row = db.execute("SELECT r.*,a.assignee_id,t.steps_json FROM runs r JOIN assignments a ON a.id=r.assignment_id JOIN sop_templates t ON t.id=a.template_id WHERE r.id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="run not found")
        if user["role"] == "inspector" and row["operator_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="run belongs to another user")
        return row

    @app.put("/api/v1/runs/{run_id}/steps/{step_key}")
    def upsert_step(run_id: str, step_key: str, request: StepResultUpsert, user: Annotated[dict, Depends(current_user)]):
        with database.connect() as db:
            run = authorized_run(db, run_id, user)
            if run["status"] != "in_progress":
                raise HTTPException(status_code=409, detail="run is not editable")
            step_keys = {step["key"] for step in json.loads(run["steps_json"])}
            if step_key not in step_keys:
                raise HTTPException(status_code=404, detail="step not in SOP")
            existing_key = db.execute("SELECT * FROM step_results WHERE idempotency_key=?", (request.idempotency_key,)).fetchone()
            if existing_key:
                return dict(existing_key)
            existing = db.execute("SELECT id FROM step_results WHERE run_id=? AND step_key=?", (run_id, step_key)).fetchone()
            values = (request.status, Database.json(request.value), request.confidence, int(request.requires_human_review), request.human_decision, request.analyzer_version, request.error_code, request.captured_at.isoformat(), now_iso())
            if existing:
                db.execute("UPDATE step_results SET idempotency_key=?,status=?,value_json=?,confidence=?,requires_human_review=?,human_decision=?,analyzer_version=?,error_code=?,captured_at=?,created_at=? WHERE id=?", (request.idempotency_key, *values, existing["id"]))
                result_id = existing["id"]
            else:
                cursor = db.execute("INSERT INTO step_results(run_id,step_key,idempotency_key,status,value_json,confidence,requires_human_review,human_decision,analyzer_version,error_code,captured_at,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, step_key, request.idempotency_key, *values))
                result_id = cursor.lastrowid
            audit(db, user["id"], "upsert", "step_result", str(result_id), {"run_id": run_id, "step_key": step_key})
            row = db.execute("SELECT * FROM step_results WHERE id=?", (result_id,)).fetchone()
        return dict(row)

    @app.post("/api/v1/runs/{run_id}/steps/{step_key}/evidence", status_code=201)
    async def upload_evidence(run_id: str, step_key: str, user: Annotated[dict, Depends(current_user)], file: UploadFile = File(...)):
        allowed = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
        if file.content_type not in allowed:
            raise HTTPException(status_code=415, detail="unsupported evidence type")
        content = await file.read(15 * 1024 * 1024 + 1)
        if not content or len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="evidence must be 1 byte to 15 MB")
        digest = hashlib.sha256(content).hexdigest()
        storage_name = f"{uuid.uuid4().hex}{allowed[file.content_type]}"
        with database.connect() as db:
            authorized_run(db, run_id, user)
            result = db.execute("SELECT id FROM step_results WHERE run_id=? AND step_key=?", (run_id, step_key)).fetchone()
            if not result:
                raise HTTPException(status_code=409, detail="submit step result before evidence")
            target = settings.evidence_dir / storage_name
            target.write_bytes(content)
            cursor = db.execute("INSERT INTO evidence(step_result_id,storage_name,original_name,media_type,sha256,size_bytes,created_at) VALUES(?,?,?,?,?,?,?)", (result["id"], storage_name, Path(file.filename or "evidence").name, file.content_type, digest, len(content), now_iso()))
            audit(db, user["id"], "upload", "evidence", str(cursor.lastrowid), {"sha256": digest})
            evidence_id = cursor.lastrowid
        return {"id": evidence_id, "sha256": digest, "size_bytes": len(content), "media_type": file.content_type}

    @app.get("/api/v1/evidence/{evidence_id}")
    def get_evidence(evidence_id: int, user: Annotated[dict, Depends(current_user)]):
        with database.connect() as db:
            row = db.execute("SELECT e.*,r.operator_id FROM evidence e JOIN step_results s ON s.id=e.step_result_id JOIN runs r ON r.id=s.run_id WHERE e.id=?", (evidence_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="evidence not found")
        if user["role"] == "inspector" and row["operator_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="evidence belongs to another user")
        return FileResponse(settings.evidence_dir / row["storage_name"], media_type=row["media_type"], filename=row["original_name"])

    @app.post("/api/v1/runs/{run_id}/submit")
    def submit_run(run_id: str, user: Annotated[dict, Depends(current_user)]):
        with database.connect() as db:
            run = authorized_run(db, run_id, user)
            definitions = json.loads(run["steps_json"])
            results = {row["step_key"]: row for row in db.execute("SELECT * FROM step_results WHERE run_id=?", (run_id,)).fetchall()}
            missing = [step["key"] for step in definitions if step.get("required", True) and step["key"] not in results]
            missing_evidence = []
            for step in definitions:
                result = results.get(step["key"])
                if result and step.get("require_evidence", True) and not db.execute("SELECT id FROM evidence WHERE step_result_id=?", (result["id"],)).fetchone():
                    missing_evidence.append(step["key"])
            unresolved = [
                step["key"] for step in definitions
                if step.get("require_human_confirmation", False)
                and step["key"] in results
                and not results[step["key"]]["human_decision"]
            ]
            if missing or missing_evidence or unresolved:
                raise HTTPException(status_code=409, detail={"missing_steps": missing, "missing_evidence": missing_evidence, "unresolved_review": unresolved})
            submitted_at = now_iso()
            db.execute("UPDATE runs SET status='submitted',submitted_at=? WHERE id=?", (submitted_at, run_id))
            audit(db, user["id"], "submit", "run", run_id)
        return {"id": run_id, "status": "submitted", "submitted_at": submitted_at}

    @app.post("/api/v1/runs/{run_id}/review")
    def review_run(run_id: str, request: ReviewRequest, user: Annotated[dict, Depends(current_user)]):
        require(user, "admin", "reviewer")
        with database.connect() as db:
            run = authorized_run(db, run_id, user)
            if run["status"] != "submitted":
                raise HTTPException(status_code=409, detail="only submitted runs can be reviewed")
            reviewed_at = now_iso()
            db.execute("UPDATE runs SET status=?,reviewed_by=?,reviewed_at=?,review_note=? WHERE id=?", (request.decision, user["id"], reviewed_at, request.note, run_id))
            if request.decision == "reviewed":
                db.execute("UPDATE assignments SET status='completed' WHERE id=?", (run["assignment_id"],))
            else:
                db.execute("UPDATE assignments SET status='pending' WHERE id=?", (run["assignment_id"],))
            audit(db, user["id"], request.decision, "run", run_id)
        return {"id": run_id, "status": request.decision, "reviewed_at": reviewed_at}

    return app


app = create_app()
