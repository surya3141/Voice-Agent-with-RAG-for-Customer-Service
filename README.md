# Voice Agent with RAG for Customer Service

An AI-powered voice agent that automates customer service calls using Speech-to-Text, Retrieval-Augmented Generation (RAG), and Text-to-Speech. Built for **TechPulse Electronics** as a demonstration of end-to-end voice AI pipeline.

## Features

- **Speech-to-Text** — Deepgram Nova-2 for accurate transcription
- **Intent Detection** — Automatic classification of customer queries (FAQ, order status, appointments)
- **RAG Pipeline** — ChromaDB vector store with semantic search for contextual answers
- **LLM Integration** — Llama 3.3 70B on Groq Cloud for natural response generation
- **Text-to-Speech** — ElevenLabs for natural speech output
- **Telephony** — Twilio integration for inbound/outbound calls
- **Security** — PII data masking (credit cards, SSN, email, phone) and RBAC
- **Monitoring** — Real-time cost tracking with ROI analysis and threshold alerts
- **Dashboard** — Streamlit app for monitoring, call simulation, and knowledge base browsing

## Architecture

```
Caller → Twilio → Deepgram STT → Data Masking → RAG Pipeline → Groq LLM → ElevenLabs TTS → Caller
                                                      ↑
                                              ChromaDB Vector Store
                                           (FAQs, Orders, Appointments)
```

## Project Structure

```
├── src/
│   ├── stt/              # Speech-to-Text (Deepgram)
│   ├── tts/              # Text-to-Speech (ElevenLabs)
│   ├── rag/              # Vector store + RAG pipeline (ChromaDB)
│   ├── llm/              # LLM integration (Groq/Llama)
│   ├── agent/            # Voice agent orchestrator
│   ├── telephony/        # Twilio integration
│   ├── security/         # Data masking + RBAC
│   ├── monitoring/       # Cost tracking + alerts
│   └── config.py         # Centralized configuration
├── data/                 # Sample datasets (FAQs, orders, appointments)
├── dashboard/            # Streamlit monitoring dashboard
├── tests/                # Unit tests (41 tests)
├── evaluation/           # Evaluation pipeline
├── deployment/           # Docker configuration
├── docs/                 # Cheatsheet and blog templates
└── .github/workflows/    # CI/CD pipeline
```

## Quick Start

### Prerequisites
- Python 3.11+
- API keys for: Deepgram, Groq, ElevenLabs, Twilio (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/surya3141/Voice-Agent-with-RAG-for-Customer-Service.git
cd Voice-Agent-with-RAG-for-Customer-Service

# Create and configure environment
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py

# Run tests
python -m pytest tests/ -v
```

### Docker Deployment

```bash
cd deployment
docker-compose up -d
```

## Usage

### Programmatic Usage

```python
from src.config import get_config
from src.agent.voice_agent import VoiceAgent

# Initialize the agent
config = get_config()
agent = VoiceAgent(config)
agent.initialize()

# Process a text query
result = agent.process_text("What is your return policy?")
print(result["response_text"])
print(result["intent"])  # "faq"

# Process audio
result = agent.process_audio(audio_bytes, mimetype="audio/wav")
print(result["transcript"])
print(result["response_text"])
```

### Dashboard

The Streamlit dashboard provides:
- **Overview** — Key metrics and call logs
- **Call Simulator** — Test queries against the RAG pipeline
- **Cost Analysis** — API cost breakdown and ROI reports (admin only)
- **Knowledge Base** — Searchable FAQ browser

### Evaluation

```bash
# Run the evaluation pipeline
python -m evaluation.eval_pipeline
```

## API Cost Estimates

| Service | Pricing |
|---------|---------|
| Deepgram STT | $0.0043/minute |
| Groq LLM | $0.0003/1K tokens |
| ElevenLabs TTS | $0.30/1K characters |
| Twilio | $0.0085/minute |

**Estimated cost per call: ~$0.02** (vs ~$5.00 for human agent)

## Security

- **PII Masking** — Automatic detection and masking of credit cards, SSNs, emails, and phone numbers
- **RBAC** — Three roles (Admin, Support, Viewer) with HMAC-SHA256 token authentication
- **No Secrets in Code** — All credentials loaded from environment variables

## Testing

```bash
# Run all 41 tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_security.py -v
python -m pytest tests/test_rag.py -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| STT | Deepgram Nova-2 |
| LLM | Llama 3.3 70B (Groq Cloud) |
| Vector DB | ChromaDB |
| TTS | ElevenLabs |
| Telephony | Twilio |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## License

This project is for demonstration and educational purposes.