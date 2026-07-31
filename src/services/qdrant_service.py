import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.models.document import Document
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

        if self.config.QDRANT_URL:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
                check_compatibility=False
            )
        else:
            # Local Qdrant
            self.client = QdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT,
                check_compatibility=False
            )

        self.collection_name = self.config.COLLECTION_NAME

    def setup_collection(self) -> bool:
        """
        Create the collection if it does not exist.

        Also ensures required payload indexes exist even when
        the collection already exists.
        """

        try:
            collections = self.client.get_collections()

            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            # ---------------------------------------------------------
            # Create collection if it does not exist
            # ---------------------------------------------------------

            if not collection_exists:

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )

                logger.info(
                    "Created collection %s with %d-dimensional vectors",
                    self.collection_name,
                    self.config.EMBEDDING_DIMENSION
                )

            else:

                logger.info(
                    "Collection %s already exists",
                    self.collection_name
                )

            # ---------------------------------------------------------
            # Ensure source_url index exists
            # ---------------------------------------------------------

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            # ---------------------------------------------------------
            # Ensure chapter index exists
            # ---------------------------------------------------------

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chapter",
                field_schema=models.PayloadSchemaType.INTEGER
            )

            logger.info(
                "Payload indexes are ready for source_url and chapter"
            )

            return True

        except Exception as e:
            logger.error(
                "Error setting up collection: %s",
                e
            )
            raise

    def store_document(
        self,
        document: Document,
        embedding: List[float]
    ) -> bool:
        """Store one document and its vector in Qdrant."""

        try:
            point = PointStruct(
                id=document.content_id,
                vector=embedding,
                payload={
                    "source_url": document.source_url,
                    "text_content": document.text_content,
                    "chapter": document.metadata.get("chapter"),
                    "metadata": document.metadata,
                    "created_at": document.created_at.isoformat()
                }
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(
                "Stored document %s",
                document.content_id
            )

            return True

        except Exception as e:
            logger.error(
                "Error storing document %s: %s",
                document.content_id,
                e
            )
            raise

    def store_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Store multiple documents and embeddings in one Qdrant upsert.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        if not documents:
            return True

        try:
            points = []

            for document, embedding in zip(
                documents,
                embeddings
            ):
                points.append(
                    PointStruct(
                        id=document.content_id,
                        vector=embedding,
                        payload={
                            "source_url": document.source_url,
                            "text_content": document.text_content,
                            "chapter": document.metadata.get("chapter"),
                            "metadata": document.metadata,
                            "created_at": document.created_at.isoformat()
                        }
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(
                "Stored %d documents in Qdrant",
                len(points)
            )

            return True

        except Exception as e:
            logger.error(
                "Error storing documents: %s",
                e
            )
            raise

    def retrieve_chapter_chunks(self, chapter_number: int) -> List[RetrievedChunk]:
        """
        Retrieve all stored chunks belonging to a specific chapter.
        """

        try:
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chapter",
                            match=models.MatchValue(
                                value=chapter_number
                            )
                        )
                    ]
                ),
                with_payload=True,
                limit=1000
            )
    
            retrieved_chunks = []
    
            for point in points:
            
                if not point.payload:
                    continue
                
                metadata = point.payload.get(
                    "metadata",
                    {}
                )
    
                chunk = RetrievedChunk(
                    chunk_text=point.payload.get(
                        "text_content",
                        ""
                    ),
                    source_url=point.payload.get(
                        "source_url",
                        ""
                    ),
                    relevance_score=1.0,
                    metadata=metadata
                )
    
                retrieved_chunks.append(chunk)
    
            retrieved_chunks.sort(
                key=lambda chunk: (
                    chunk.source_url,
                    chunk.metadata.get(
                        "chunk_index",
                        0
                    )
                )
            )
    
            logger.info(
                "Retrieved %d chunks from chapter %d",
                len(retrieved_chunks),
                chapter_number
            )
    
            return retrieved_chunks
    
        except Exception as e:
            logger.error(
                "Error retrieving chapter %d: %s",
                chapter_number,
                e
            )
            raise

    def retrieve_similar(
        self,
        query_embedding: List[float],
        top_k: int = 8
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most semantically relevant textbook chunks.
        """

        try:

            if hasattr(self.client, "query_points"):

                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True
                )

                results = getattr(
                    response,
                    "points",
                    []
                )

            else:

                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True
                )

            retrieved_chunks: List[RetrievedChunk] = []

            for result in results:

                payload = None
                score = 0.0

                if (
                    hasattr(result, "payload")
                    and result.payload is not None
                ):
                    payload = result.payload
                    score = getattr(
                        result,
                        "score",
                        0.0
                    )

                elif isinstance(result, dict):

                    payload = result.get(
                        "payload",
                        {}
                    )

                    score = result.get(
                        "score",
                        0.0
                    )

                if not payload:
                    continue

                metadata = payload.get(
                    "metadata",
                    {}
                )

                metadata = dict(metadata)

                metadata["relevance_score"] = score

                chunk = RetrievedChunk(
                    chunk_text=payload.get(
                        "text_content",
                        ""
                    ),
                    source_url=payload.get(
                        "source_url",
                        ""
                    ),
                    relevance_score=score,
                    metadata=metadata
                )

                retrieved_chunks.append(chunk)

            logger.info(
                "Retrieved %d relevant chunks from Qdrant",
                len(retrieved_chunks)
            )

            return retrieved_chunks

        except Exception as e:

            logger.error(
                "Error retrieving similar content: %s",
                e
            )

            raise

    def search_by_source(
        self,
        source_url: str
    ) -> List[RetrievedChunk]:
        """Retrieve all chunks associated with one source URL."""

        try:

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url",
                            match=models.MatchValue(
                                value=source_url
                            )
                        )
                    ]
                ),
                with_payload=True,
                limit=1000
            )

            retrieved_chunks = []

            for point, _ in results:

                if not point.payload:
                    continue

                chunk = RetrievedChunk(
                    chunk_text=point.payload.get(
                        "text_content",
                        ""
                    ),
                    source_url=point.payload.get(
                        "source_url",
                        ""
                    ),
                    relevance_score=1.0,
                    metadata=point.payload.get(
                        "metadata",
                        {}
                    )
                )

                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:

            logger.error(
                "Error searching by source %s: %s",
                source_url,
                e
            )

            raise

    def delete_collection(self) -> bool:
        """
        Delete the entire Qdrant collection.
        """

        try:

            self.client.delete_collection(
                collection_name=self.collection_name
            )

            logger.info(
                "Deleted collection %s",
                self.collection_name
            )

            return True

        except Exception as e:

            logger.error(
                "Error deleting collection: %s",
                e
            )

            raise

    def reset_collection(self) -> bool:
        """
        Delete and recreate the textbook collection.

        Use this only when explicitly ingesting/rebuilding
        the textbook index.
        """

        try:

            collections = self.client.get_collections()

            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if collection_exists:
                self.delete_collection()

            return self.setup_collection()

        except Exception as e:

            logger.error(
                "Error resetting collection: %s",
                e
            )

            raise