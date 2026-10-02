---
title: Agentic RAG - Indian Health Insurance
emoji: 📑
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Agentic RAG over Indian health insurance policy wordings

**Day-1 deploy. The engine is a stub and abstains on every question by design.**

This Space exists to prove the deployment path while there is time to fix it, not to
demonstrate a working system. `GET /status` reports exactly what is and is not wired.

- `GET /health` — liveness
- `GET /status` — what is wired, and that the eval gate has no labelled data yet
- `POST /ask` — `{"question": "..."}` → a real `AnswerResult`, currently always abstaining

Source: the `arag` package. Corpus documents are never redistributed; the repo ships a
manifest of URLs and SHA-256 hashes only.
