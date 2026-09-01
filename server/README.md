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

管理后台：`http://127.0.0.1:8080/admin`。API 文档：`http://127.0.0.1:8080/docs`。

## 生产部署

复制 `.env.example` 为 `.env`，设置随机 secret 和初始管理员密码后：

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:18081/healthz
```

容器仅绑定主机回环地址 `127.0.0.1:18081`。公网访问应由主机现有 nginx/Caddy 反代并配置 HTTPS，
不要直接暴露 SQLite 服务或容器端口。

当前部署：`https://crrc-glasses.ifix.xin/`，根入口会跳转至中文 Web 管理后台；OpenAPI 文档位于
`https://crrc-glasses.ifix.xin/docs`。管理后台支持仪表盘、用户、SOP模板、任务和巡检记录管理，
巡检员账号只能通过API/移动端访问自己的任务与记录。旧的`https://finbot.ifix.xin/crrc-sop/`暂保留兼容。
服务器代码和数据分别位于`/opt/crrc-sop`与Docker卷
`crrc-sop_crrc_sop_data`；管理员初始密码文件仅root可读，使用后应通过受控运维流程轮换。
