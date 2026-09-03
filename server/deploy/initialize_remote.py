"""Initialize production secrets without printing them to stdout."""

import os
import secrets
from pathlib import Path


deploy_dir = Path("/opt/crrc-sop")
env_file = deploy_dir / ".env"
password_file = Path("/root/crrc-sop-admin-password.txt")

if env_file.exists() and "CRRC_SECRET_KEY=" in env_file.read_text():
    values = dict(
        line.split("=", 1)
        for line in env_file.read_text().splitlines()
        if line and "=" in line
    )
    existing_secret = values.get("CRRC_SECRET_KEY", "")
    existing_password = values.get("CRRC_BOOTSTRAP_ADMIN_PASSWORD", "")
else:
    existing_secret = ""
    existing_password = ""

secret = existing_secret if len(existing_secret) >= 32 else secrets.token_hex(32)
password = existing_password if len(existing_password) >= 12 else secrets.token_urlsafe(18)
env_file.write_text(
    "\n".join(
        [
            f"CRRC_SECRET_KEY={secret}",
            "CRRC_BOOTSTRAP_ADMIN_USERNAME=admin",
            f"CRRC_BOOTSTRAP_ADMIN_PASSWORD={password}",
            "CRRC_DATABASE_PATH=/data/crrc-sop.sqlite3",
            "CRRC_EVIDENCE_DIR=/data/evidence",
            "CRRC_ACCESS_TOKEN_MINUTES=720",
            "CRRC_ROOT_PATH=/crrc-sop",
            "",
        ]
    )
)
password_file.write_text(password + "\n")
os.chmod(env_file, 0o600)
os.chmod(password_file, 0o600)
print("initialized /opt/crrc-sop/.env and protected admin credential file")
