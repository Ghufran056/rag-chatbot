import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppException(Exception):
    """Base exception class for the application."""
    pass

class ConfigurationError(AppException):
    """Raised when there's a configuration issue."""
    pass

class ValidationError(AppException):
    """Raised when there's a validation issue."""
    pass

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up and return a logger with the specified level."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    return logger

def validate_api_keys(config) -> bool:
    """Validate that required API keys are present."""
    if not config.OPENAI_API_KEY:
        raise ConfigurationError("OPENAI_API_KEY is required")
    if not config.COHERE_API_KEY:
        raise ConfigurationError("COHERE_API_KEY is required")
    return True

def safe_get_nested(data: Dict[str, Any], *keys) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length and add ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_sources(sources: List[str]) -> str:
    """Format a list of sources into a readable string."""
    if not sources:
        return ""
    return "\n\nSources: " + "\n".join([f"- {source}" for source in sources])

def handle_error(error: Exception, context: str = "") -> None:
    """Log an error with context and re-raise it."""
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    raise error