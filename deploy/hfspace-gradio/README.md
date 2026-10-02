---
title: Agentic RAG - Indian Health Insurance
emoji: 📑
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Agentic RAG over Indian health insurance policy wordings

**Day-1 deploy proving the path. The engine is a stub and abstains on every question.**

No Dockerfile needed: HF installs `requirements.txt` and runs `app.py`. The FastAPI app is
mounted under the Gradio server, so the JSON API and a browsable UI share one process and
one port.

- UI at `/`
- `GET /health`, `GET /status`
- `POST /ask` with `{"question": "..."}`

`GET /status` reports exactly what is and is not wired, so a live URL cannot imply a
working system.
