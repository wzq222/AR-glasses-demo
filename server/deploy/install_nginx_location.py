"""Add the isolated CRRC SOP reverse-proxy location to the existing TLS vhost."""

from pathlib import Path
import shutil


config = Path("/etc/nginx/sites-enabled/finbot.ifix.xin")
backup = Path("/root/finbot.ifix.xin.pre-crrc-sop")
text = config.read_text()
marker = "location /crrc-sop/"
if marker in text:
    print("CRRC SOP nginx location already present")
    raise SystemExit(0)

needle = "    location / {\n"
if needle not in text:
    raise RuntimeError("expected root location not found; refusing to edit nginx")

location = """    location /crrc-sop/ {
        proxy_pass http://127.0.0.1:18081/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

"""
if not backup.exists():
    shutil.copy2(config, backup)
config.write_text(text.replace(needle, location + needle, 1))
print("installed CRRC SOP nginx location")
