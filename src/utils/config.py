import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage application settings."""

    # API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    # GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY")

    # Qdrant Configuration
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Application settings
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "textbook_content")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))  # Cohere multilingual dimensions
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # Validation
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        # At least one LLM API key is required (OpenAI or Google)
        if not cls.OPENROUTER_API_KEY :
            raise ValueError("Either OPENROUTER_API_KEY is required")
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is required")
        return True