from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class Query(BaseModel):
    """
    Represents a user's question with its embedding representation.

    Attributes:
        query_id: Unique identifier for the query
        query_text: The original text of the user's question
        query_embedding: Vector representation of the query
        created_at: Timestamp when the query was created
        top_k: Number of results to retrieve
    """
    query_id: str
    query_text: str
    query_embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    top_k: int = 5  # Default number of results to retrieve

    class Config:
        # Allow arbitrary types for embedding vector
        arbitrary_types_allowed = True