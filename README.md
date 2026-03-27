# Voice Agent with RAG for Customer Service

A production-ready voice agent that handles customer-service calls end-to-end:

| Stage | Technology |
|---|---|
| **Speech-to-Text** | OpenAI Whisper (`whisper-1`) |
| **Intent Detection** | OpenAI GPT (`gpt-4o-mini`) |
| **Knowledge Retrieval** | FAISS vector store + LangChain RAG |
| **Response Generation** | OpenAI GPT (`gpt-4o-mini`) |
| **Text-to-Speech** | OpenAI TTS (`tts-1`, voice: alloy) |
| **API Server** | FastAPI |

The agent handles **FAQs, order status queries, appointment scheduling/cancellation, product information, billing questions**, and more — reducing workload and improving customer experience.

---

## Architecture

```
Caller audio
    │
    ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐
│  Whisper    │───▶│ Intent       │    │  FAISS Vector Store    │
│  STT        │    │ Detection    │    │  (knowledge base)      │
└─────────────┘    │  (GPT LLM)  │───▶│                        │
                   └──────────────┘    └───────────┬────────────┘
                                                   │ top-k docs
                                                   ▼
                                       ┌────────────────────────┐
                                       │ Response Generation    │
                                       │ (GPT LLM + RAG context)│
                                       └───────────┬────────────┘
                                                   │
                                                   ▼
                                       ┌────────────────────────┐
                                       │  OpenAI TTS            │
                                       │  (MP3 audio reply)     │
                                       └────────────────────────┘
```

---

## Project Structure

```
.
├── app.py                  # FastAPI application (API server)
├── requirements.txt
├── .env.example            # Environment variable template
├── voice_agent/
│   ├── __init__.py
│   ├── config.py           # Settings (loaded from .env)
│   ├── stt.py              # Speech-to-Text (OpenAI Whisper)
│   ├── tts.py              # Text-to-Speech (OpenAI TTS)
│   ├── rag.py              # RAG pipeline (FAISS + LangChain)
│   ├── llm.py              # LLM client (intent + response)
│   └── agent.py            # End-to-end orchestration
├── data/                   # Knowledge-base files
│   ├── faqs.json           # General FAQs
│   ├── orders.json         # Order status information
│   ├── appointments.json   # Appointment Q&A
│   ├── products.json       # Product information
│   └── billing.txt         # Billing policies (plain text)
└── tests/
    ├── test_agent.py
    ├── test_app.py
    ├── test_llm.py
    ├── test_rag.py
    └── test_stt_tts.py
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/surya3141/Voice-Agent-with-RAG-for-Customer-Service.git
cd Voice-Agent-with-RAG-for-Customer-Service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the Server

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On first start the server embeds the knowledge-base files and saves a FAISS index to `faiss_index/`. Subsequent starts load the index from disk (much faster).

### 4. Interactive API Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/voice` | Upload audio → receive MP3 reply |
| `POST` | `/voice/text` | Send text → receive JSON reply |
| `POST` | `/voice/text/audio` | Send text → receive MP3 reply |

### Example: text query

```bash
curl -s -X POST http://localhost:8000/voice/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Where is my order #ORD-1001?"}' | python -m json.tool
```

```json
{
  "transcript": "Where is my order #ORD-1001?",
  "intent": "order_status",
  "reply_text": "Order #ORD-1001 is currently in transit and is estimated to arrive on March 28, 2026. You can track it with tracking number 1Z999AA10123456784.",
  "context_used": "Q: What is the status of order #ORD-1001?\nA: ..."
}
```

### Example: voice call

```bash
curl -X POST http://localhost:8000/voice \
  -F "audio=@recording.wav" \
  --output reply.mp3
```

The `X-Transcript` and `X-Intent` response headers carry the STT transcript and detected intent respectively.

---

## Adding to the Knowledge Base

Drop `.json` or `.txt` files into the `data/` directory and delete the `faiss_index/` directory to trigger a rebuild on next start.

**JSON format** (FAQ style):
```json
[
  {"question": "Do you offer gift wrapping?", "answer": "Yes, for $5 per item."}
]
```

**Plain text** (policies, manuals, etc.):
```text
RETURNS POLICY
Items must be returned within 30 days in original packaging ...
```

---

## Supported Intents

| Intent | Example trigger |
|--------|-----------------|
| `faq` | "What are your hours?" |
| `order_status` | "Where is my order?" |
| `schedule_appointment` | "I'd like to book a technician." |
| `cancel_appointment` | "Cancel my appointment for Friday." |
| `product_info` | "Tell me about the ProMax 3000." |
| `complaint` | "My package arrived damaged." |
| `billing` | "I have a question about my invoice." |
| `general` | Everything else |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocks — no OpenAI API key required.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for intent & responses |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `STT_MODEL` | `whisper-1` | Whisper model for transcription |
| `TTS_MODEL` | `tts-1` | TTS model |
| `TTS_VOICE` | `alloy` | TTS voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| `FAISS_INDEX_PATH` | `faiss_index` | Where to save/load the vector index |
| `DATA_DIR` | `data` | Knowledge-base directory |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
