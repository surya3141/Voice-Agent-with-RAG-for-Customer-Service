# Voice Agent with RAG — Quick Reference Card

## 🏗️ Architecture
- **STT**: Deepgram Nova-2 → transcription
- **LLM**: Llama 3.3 70B on Groq Cloud → understanding & response generation
- **RAG**: ChromaDB vector store → contextual knowledge retrieval
- **TTS**: ElevenLabs → natural speech output
- **Telephony**: Twilio → inbound/outbound calls
- **Dashboard**: Streamlit → monitoring & simulation

## 🚀 Quick Start
```bash
# Clone and setup
git clone <repo-url>
cd Voice-Agent-with-RAG-for-Customer-Service
cp .env.example .env
# Fill in your API keys in .env

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v

# Docker deployment
cd deployment && docker-compose up -d
```

## 📁 Project Structure
```
src/
├── stt/          # Speech-to-Text (Deepgram)
├── tts/          # Text-to-Speech (ElevenLabs)
├── rag/          # Vector store + RAG pipeline
├── llm/          # LLM integration (Groq/Llama)
├── agent/        # Voice agent orchestrator
├── telephony/    # Twilio integration
├── security/     # Data masking + RBAC
├── monitoring/   # Cost tracking + alerts
└── config.py     # Centralized configuration
```

## 🔑 Key Components
| Component | Class | Module |
|-----------|-------|--------|
| STT | `DeepgramSTT` | `src.stt.deepgram_stt` |
| TTS | `ElevenLabsTTS` | `src.tts.elevenlabs_tts` |
| Vector Store | `KnowledgeStore` | `src.rag.vector_store` |
| RAG | `RAGPipeline` | `src.rag.pipeline` |
| LLM | `GroqLLM` | `src.llm.groq_llm` |
| Agent | `VoiceAgent` | `src.agent.voice_agent` |
| Telephony | `TwilioHandler` | `src.telephony.twilio_handler` |
| Security | `DataMasker`, `RBACManager` | `src.security.*` |
| Monitoring | `CostTracker` | `src.monitoring.cost_tracker` |

## 🔒 Security Features
- PII masking (credit cards, SSN, email, phone)
- RBAC with 3 roles: Admin, Support, Viewer
- HMAC-SHA256 token authentication
- No secrets in source code (.env based)

## 💰 Cost Tracking
- Deepgram: $0.0043/min
- Groq: $0.0003/1K tokens
- ElevenLabs: $0.30/1K chars
- Twilio: $0.0085/min

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_security.py -v

# Run evaluation pipeline
python -m evaluation.eval_pipeline
```
