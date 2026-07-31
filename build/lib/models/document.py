from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class Document(BaseModel):
    """
    Represents a piece of content from the textbook with its metadata.

    Attributes:
        content_id: Unique identifier for the document
        source_url: URL where the content was sourced from
        text_content: The actual text content
        embedding_vector: Vector representation of the content
        metadata: Additional metadata about the document
        created_at: Timestamp when the document was created
    """
    content_id: str
    source_url: str
    text_content: str
    embedding_vector: Optional[list] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Allow arbitrary types for embedding vector
        arbitrary_types_allowed = True