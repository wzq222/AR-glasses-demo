# 中车眼镜 SOP Server

最小后台闭环：用户/角色、版本化 SOP、巡检任务、执行会话、步骤结果、证据图片、人工复核和审计日志。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
$env:CRRC_BOOTSTRAP_ADMIN_PASSWORD='本地强密码'
$env:CRRC_SECRET_KEY='至少32位随机字符串'
.\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8080
```

API 文档：`http://127.0.0.1:8080/docs`。

## 生产部署

复制 `.env.example` 为 `.env`，设置随机 secret 和初始管理员密码后：

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:18081/healthz
```

容器仅绑定主机回环地址 `127.0.0.1:18081`。公网访问应由主机现有 nginx/Caddy 反代并配置 HTTPS，
不要直接暴露 SQLite 服务或容器端口。
