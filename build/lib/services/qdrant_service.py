import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.models.document import Document
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk
from src.utils.config import Config

logger = logging.getLogger(__name__)

class QdrantService:
    """
    Service for managing vector storage and retrieval using Qdrant.
    """

    def __init__(self):
        """Initialize the Qdrant service."""
        self.config = Config()
        self.config.validate()

        # Initialize Qdrant client
        if self.config.QDRANT_URL:
            # Using Qdrant Cloud
            self.client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY
            )
        else:
            # Using local Qdrant instance
            self.client = QdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT
            )

        self.collection_name = self.config.COLLECTION_NAME

    def setup_collection(self) -> bool:
        """
        Set up the Qdrant collection with the proper schema for textbook content.

        The collection will store:
        - Vector embeddings of text chunks
        - Payload with source URL and other metadata
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if collection_exists:
                logger.info(f"Collection {self.collection_name} already exists")
                return True

            # Create collection with specified vector size (Cohere multilingual embeddings are 1024-dim)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.EMBEDDING_DIMENSION,  # Cohere multilingual embeddings are 1024-dim
                    distance=Distance.COSINE
                )
            )

            logger.info(f"Created collection {self.collection_name} with {self.config.EMBEDDING_DIMENSION}-dimension vectors")

            # Create payload index for faster filtering by source_url
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            return True
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise e

    def store_document(self, document: Document, embedding: List[float]) -> bool:
        """
        Store a document with its embedding in Qdrant.

        Args:
            document: The document to store
            embedding: The embedding vector for the document

        Returns:
            True if successful
        """
        try:
            # Prepare the point to insert
            point = PointStruct(
                id=document.content_id,
                vector=embedding,
                payload={
                    "source_url": document.source_url,
                    "text_content": document.text_content,
                    "metadata": document.metadata,
                    "created_at": document.created_at.isoformat()
                }
            )

            # Insert the point into the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Stored document {document.content_id} in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error storing document {document.content_id}: {str(e)}")
            raise e

    def store_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        """
        Store multiple documents with their embeddings in Qdrant.

        Args:
            documents: List of documents to store
            embeddings: List of embedding vectors corresponding to the documents

        Returns:
            True if successful
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        try:
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = PointStruct(
                    id=doc.content_id,
                    vector=embedding,
                    payload={
                        "source_url": doc.source_url,
                        "text_content": doc.text_content,
                        "metadata": doc.metadata,
                        "created_at": doc.created_at.isoformat()
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.debug(f"Stored {len(documents)} documents in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise e

    def retrieve_similar(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve similar content from Qdrant based on the query embedding.

        Args:
            query_embedding: The embedding vector to search for similar content
            top_k: Number of results to return

        Returns:
            List of RetrievedChunk objects containing the similar content
        """
        try:
            # Search for similar vectors
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )

            retrieved_chunks = []
            for result in results:
                if result.payload:
                    chunk = RetrievedChunk(
                        chunk_text=result.payload.get("text_content", ""),
                        source_url=result.payload.get("source_url", ""),
                        relevance_score=result.score,
                        metadata=result.payload.get("metadata", {})
                    )
                    retrieved_chunks.append(chunk)

            logger.debug(f"Retrieved {len(retrieved_chunks)} similar chunks from Qdrant")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error retrieving similar content: {str(e)}")
            raise e

    def search_by_source(self, source_url: str) -> List[RetrievedChunk]:
        """
        Search for all content from a specific source URL.

        Args:
            source_url: The source URL to search for

        Returns:
            List of RetrievedChunk objects from that source
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url",
                            match=models.MatchValue(value=source_url)
                        )
                    ]
                ),
                with_payload=True,
                limit=1000  # Limit to prevent too many results
            )

            retrieved_chunks = []
            for point, _ in results:
                if point.payload:
                    chunk = RetrievedChunk(
                        chunk_text=point.payload.get("text_content", ""),
                        source_url=point.payload.get("source_url", ""),
                        relevance_score=1.0,  # Score is not meaningful in this context
                        metadata=point.payload.get("metadata", {})
                    )
                    retrieved_chunks.append(chunk)

            logger.debug(f"Found {len(retrieved_chunks)} chunks from source {source_url}")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error searching by source {source_url}: {str(e)}")
            raise e

    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution!).

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise e