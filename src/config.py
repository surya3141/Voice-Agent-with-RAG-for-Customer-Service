"""Configuration module for Voice Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_data"


@dataclass
class DeepgramConfig:
    """Deepgram STT configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    model: str = "nova-2"
    language: str = "en-US"


@dataclass
class GroqConfig:
    """Groq LLM configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class ElevenLabsConfig:
    """ElevenLabs TTS configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default "Rachel" voice
    model_id: str = "eleven_monolingual_v1"


@dataclass
class TwilioConfig:
    """Twilio telephony configuration."""
    account_sid: str = field(default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    phone_number: str = field(default_factory=lambda: os.getenv("TWILIO_PHONE_NUMBER", ""))


@dataclass
class SecurityConfig:
    """Security configuration."""
    admin_secret_key: str = field(default_factory=lambda: os.getenv("ADMIN_SECRET_KEY", ""))
    session_timeout_minutes: int = 30


@dataclass
class AppConfig:
    """Main application configuration."""
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    log_level: str = "INFO"
    data_dir: Path = DATA_DIR
    chroma_dir: Path = CHROMA_DIR


def get_config() -> AppConfig:
    """Get application configuration."""
    return AppConfig()
