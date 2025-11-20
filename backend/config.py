import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

# Get project root directory (parent of backend)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    # Model Configuration - use absolute paths based on project root
    MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models")
    LORA_ADAPTERS_PATH: str = os.path.join(PROJECT_ROOT, "lora_adapters")
    VOICE_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "vibevoice")
    
    # API Configuration  
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Model Configuration
    BASE_MODEL: str = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
    DEVICE: str = os.getenv("DEVICE", "cuda")
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    
    # Audio Configuration
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "22050"))
    AUDIO_FORMAT: str = os.getenv("AUDIO_FORMAT", "wav")
    ENABLE_VOICE: bool = os.getenv("ENABLE_VOICE", "False").lower() == "true"  # Disabled by default for easier deployment
    
    # Character Configuration
    DEFAULT_CHARACTER: str = os.getenv("DEFAULT_CHARACTER", "moses")
    
    @property
    def AVAILABLE_CHARACTERS(self) -> list:
        characters_str = os.getenv("AVAILABLE_CHARACTERS", "moses,samsung_employee,jinx")
        return [char.strip() for char in characters_str.split(",")]

settings = Settings()