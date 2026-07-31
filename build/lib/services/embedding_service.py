import cohere
from typing import List, Union
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using Cohere API.
    """

    def __init__(self):
        """Initialize the embedding service with Cohere client."""
        self.config = Config()
        self.config.validate()
        self.client = cohere.Client(self.config.COHERE_API_KEY)
        self.model = "embed-multilingual-v3.0"  # Using multilingual model for broader language support

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"  # Using search_document for content indexing
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for texts: {str(e)}")
            raise e

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (using search_query input type).

        Args:
            query: The query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"  # Using search_query for search queries
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise e