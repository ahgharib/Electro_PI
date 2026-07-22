# Electro Pi — AI Engineer Technical Test — Submission

**Role:** AI Engineer, Mid-Level (3+ yrs) · **Skills assessed:** LiveKit ·
LLMs · LangChain · Quantization · Model Deployment

This repo contains all four sections of the take-home test, each in its
own folder with its own `README.md` (setup/run instructions), `NOTES.md`
(the required half-page write-up), and any supporting evidence files
(transcripts, comparison tables, test question banks).

---

## Repo structure

```
.
├── README.md                          <- you are here
├── section1_livekit_agents/
│   ├── README.md                      Setup + run instructions
│   ├── NOTES.md                       Write-up: barge-in, second-tool safety, Task 1.2 bonus
│   ├── SWAPPING_PROVIDERS.md          Task 1.2 detail: swapping vs. adding an STT/TTS provider
│   ├── CONVERSATIONS.md               4 real live-run transcripts (readable)
│   └── transcripts/README.md          How to generate the real evidence transcript
├── section2_langchain_rag/
│   ├── README.md                      Setup + run instructions, architecture, docs used
│   ├── NOTES.md                       Write-up + 3 example Q&A
│   └── QUESTIONS_PDF_SET.md           15-question test bank against the 3 source PDFs (edge-case tests)
├── section3_quantization/
│   ├── README.md                      Setup + run instructions
│   ├── NOTES.md                       Write-up: results, qualitative comparison, GPTQ/AWQ/GGUF reasoning
│   └── results/model_comparison.md    Full per-prompt raw output backing NOTES.md's claims
└── section4_deployment/
    ├── README.md                      Setup + run instructions (Docker + native)
    └── NOTES.md                       Write-up: load test results, scaling to 50 users
```

---

## Documentation index

| Section | Task | README (setup/run) | Write-up (NOTES.md) | Supporting evidence |
|---|---|---|---|---|
| 1 — LiveKit Agents | 1.1 (required) + 1.2 (bonus) | [`section1_livekit_agents/README.md`](section-1-livekit-agents/README.md) | [`section1_livekit_agents/NOTES.md`](section-1-livekit-agents/NOTES.md) | [`CONVERSATIONS.md`](section-1-livekit-agents/CONVERSATIONS.md) · [`SWAPPING_PROVIDERS.md`](./section1_livekit_agents/SWAPPING_PROVIDERS.md) · [`transcripts/README.md`](./section1_livekit_agents/transcripts/README.md) |
| 2 — LangChain / RAG | 2.1 | [`section2_langchain_rag/README.md`](./section2_langchain_rag/README.md) | `section2_langchain_rag/NOTES.md` | [`QUESTIONS_PDF_SET.md`](./section2_langchain_rag/QUESTIONS_PDF_SET.md) |
| 3 — Quantization | 3.1 | [`section3_quantization/README.md`](./section3_quantization/README.md) | [`section3_quantization/NOTES.md`](./section3_quantization/NOTES.md) | [`results/model_comparison.md`](./section3_quantization/results/model_comparison.md) |
| 4 — Model Deployment | 4.1 | [`section4_deployment/README.md`](./section4_deployment/README.md) | [`section4_deployment/NOTES.md`](./section4_deployment/NOTES.md) | — |

---

## Quickstart (each section runnable within ~10 minutes)

```bash
# Section 1 — LiveKit voice agent (mock STT/TTS, real Groq/Cerebras LLM)
cd section1_livekit_agents && pip install -r requirements.txt && pytest tests/ -v && python src/run_demo.py

# Section 2 — LangChain/LangGraph RAG over 3 PDFs (Gemini free tier)
cd section2_langchain_rag && pip install -r requirements.txt && python scripts/ingest.py && python scripts/ask.py "What is VAERS?"

# Section 3 — Quantization benchmark (Qwen2.5-3B, fp16 vs bitsandbytes 4-bit)
cd section3_quantization && pip install -r requirements.txt && python scripts/check_environment.py && pytest tests/ -v

# Section 4 — FastAPI deployment (Docker, CPU default)
cd section4_deployment && docker build -t section4-api . && docker run -p 8000:8000 section4-api
```

Each section's own `README.md` has the full setup (API keys, env vars,
troubleshooting) — this is just the fast path.

---

## Known repo-wide limitations

- **Mocked STT/TTS (Section 1)** — explicitly allowed by the task brief; the LLM and tool-calling logic are real (Groq, with Cerebras fallback). Real audio via Deepgram/Cartesia is wired and tested at the construction level (`tests/test_providers.py`) but not run live end-to-end — needs a human with a microphone and a free-tier key.
- **OpenAI fallback path (Section 2)** — implemented but only exercised against fakes in tests, not a live OpenAI account.
- **4-bit quantization (Section 3)** — requires a CUDA GPU; no CPU kernel exists for bitsandbytes.
- **Docker deployment (Section 4)** — defaults to CPU + the smaller 1.5B model for guaranteed portability; the 3B/GPU path is native-only and optional.
