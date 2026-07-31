from pydantic import BaseModel
from typing import List, Optional


class RetrievedChunk(BaseModel):
    """
    Represents a text chunk retrieved from Qdrant with its metadata.

    Attributes:
        chunk_text: The text content of the retrieved chunk
        source_url: URL where the chunk originated from
        relevance_score: Score indicating how relevant the chunk is to the query
        metadata: Additional metadata about the chunk
    """
    chunk_text: str
    source_url: str
    relevance_score: float
    metadata: dict = {}

    @property
    def formatted_source(self) -> str:
        """Return a formatted representation of the source."""
        return f"{self.source_url}"