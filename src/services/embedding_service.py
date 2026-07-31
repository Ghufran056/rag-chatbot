import cohere
from typing import List
import logging
import time

from src.utils.config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Cohere API.
    """

    # Cohere Embed supports up to 96 text inputs per request.
    BATCH_SIZE = 96

    # Only retry transient rate limits a small number of times.
    MAX_RETRIES = 3

    def __init__(self):
        """Initialize the embedding service with Cohere client."""
        self.config = Config()
        self.config.validate()

        self.client = cohere.Client(self.config.COHERE_API_KEY)

        self.model = "embed-multilingual-v3.0"

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check whether an exception appears to be a rate-limit error."""
        message = str(error).lower()

        return (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        )

    @staticmethod
    def _is_monthly_quota_error(error: Exception) -> bool:
        """
        Check whether Cohere reports that the trial/monthly quota
        has been exhausted.

        We specifically avoid retrying this because waiting a few seconds
        will not restore a monthly quota.
        """
        message = str(error).lower()

        return (
            "1000 api calls / month" in message
            or "1000 api calls per month" in message
            or "monthly" in message and "trial key" in message
        )

    def _embed_batch(
        self,
        batch: List[str],
        input_type: str
    ) -> List[List[float]]:
        """
        Embed one batch of texts.

        Args:
            batch: Texts to embed.
            input_type: Cohere input type:
                        'search_document' or 'search_query'.

        Returns:
            Embedding vectors.
        """

        attempt = 0

        while True:
            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type
                )

                embeddings = response.embeddings

                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"Cohere returned {len(embeddings)} embeddings "
                        f"for {len(batch)} inputs."
                    )

                return embeddings

            except Exception as e:
                attempt += 1

                # Monthly quota exhaustion is not recoverable by retrying.
                if self._is_monthly_quota_error(e):
                    logger.error(
                        "Cohere monthly API quota has been exhausted. "
                        "Not retrying."
                    )
                    raise RuntimeError(
                        "Cohere monthly trial API quota has been exhausted. "
                        "Please wait for the quota reset or use another API key."
                    ) from e

                # Retry only transient rate-limit errors.
                if self._is_rate_limit_error(e) and attempt <= self.MAX_RETRIES:
                    wait_time = attempt * 3

                    logger.warning(
                        "Cohere rate limit hit. "
                        "Retrying attempt %d/%d in %d seconds.",
                        attempt,
                        self.MAX_RETRIES,
                        wait_time
                    )

                    time.sleep(wait_time)
                    continue

                logger.error(
                    "Error generating Cohere embeddings: %s",
                    e
                )
                raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single document text.

        This method remains available for compatibility, but the
        ingestion pipeline should prefer embed_texts() so that multiple
        chunks are embedded in batches.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        embeddings = self._embed_batch(
            [text],
            input_type="search_document"
        )

        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple document texts.

        Texts are automatically split into batches of up to 96.

        Args:
            texts: List of document chunks.

        Returns:
            List of embedding vectors in the same order as the input texts.
        """

        if not texts:
            return []

        # Remove accidental empty strings.
        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        all_embeddings: List[List[float]] = []

        for start in range(0, len(cleaned_texts), self.BATCH_SIZE):
            batch = cleaned_texts[start:start + self.BATCH_SIZE]

            logger.info(
                "Embedding batch %d-%d of %d chunks...",
                start + 1,
                start + len(batch),
                len(cleaned_texts)
            )

            batch_embeddings = self._embed_batch(
                batch,
                input_type="search_document"
            )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a user's search query.

        Only ONE Cohere API call is made per unique query invocation.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            response = self.client.embed(
                texts=[query.strip()],
                model=self.model,
                input_type="search_query"
            )

            return response.embeddings[0]

        except Exception as e:
            if self._is_monthly_quota_error(e):
                logger.error(
                    "Cohere monthly API quota has been exhausted."
                )

                raise RuntimeError(
                    "Cohere monthly trial API quota has been exhausted. "
                    "The textbook cannot be searched until the quota resets "
                    "or a different API key is configured."
                ) from e

            logger.error(
                "Error generating query embedding: %s",
                e
            )

            raise