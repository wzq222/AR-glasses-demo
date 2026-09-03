from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .database import Database
from .settings import Settings


password_hasher = PasswordHasher()
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_token(settings: Settings, user: dict) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def current_user_dependency(settings: Settings, database: Database):
    def current_user(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> dict:
        if credentials is None:
            raise HTTPException(status_code=401, detail="missing token")
        try:
            payload = jwt.decode(credentials.credentials, settings.secret_key, algorithms=["HS256"])
            user_id = int(payload["sub"])
        except (jwt.PyJWTError, KeyError, ValueError):
            raise HTTPException(status_code=401, detail="invalid token") from None
        with database.connect() as db:
            row = db.execute(
                "SELECT id,username,display_name,role,active,created_at FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        user = Database.row(row)
        if not user or not user["active"]:
            raise HTTPException(status_code=401, detail="inactive user")
        return user

    return current_user


def require_roles(*roles: str):
    def dependency(user: Annotated[dict, Depends(lambda: None)]) -> dict:  # replaced in main
        return user
    dependency.allowed_roles = set(roles)  # type: ignore[attr-defined]
    return dependency
