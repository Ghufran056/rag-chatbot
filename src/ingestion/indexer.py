import uuid
from typing import List, Dict, Any
import logging
import re

from src.services.qdrant_service import QdrantService
from src.services.embedding_service import EmbeddingService
from src.models.document import Document
from src.utils.config import Config


def _is_placeholder_content(content: str) -> bool:
    """Return True for generic placeholder text that should not be indexed."""

    lowered = content.lower()

    placeholders = [
        "lorem ipsum",
        "this is the summary of a very long blog post",
        "use a `<!--",
        "one min read",
        "first blog post",
        "long blog post",
        "skip to main content",
    ]

    return any(token in lowered for token in placeholders)


def _is_content_relevant(
    url: str,
    content: str,
    title: str = ""
) -> bool:
    """Prefer real docs/chapter content over generic blog placeholders."""

    lowered = (url + " " + title + " " + content).lower()

    if _is_placeholder_content(content):
        return False

    if "/docs/" in url or "/chapter" in url or "chapter" in lowered:
        return True

    if (
        "embodied intelligence" in lowered
        or "learning objectives" in lowered
    ):
        return True

    return False


logger = logging.getLogger(__name__)


class Indexer:
    """
    Indexes textbook content by:

    1. Extracting content
    2. Chunking text
    3. Batch-generating embeddings
    4. Storing vectors in Qdrant
    """

    def __init__(self):
        """Initialize the indexer with required services."""

        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.config = Config()

        # Only creates the collection if it does not already exist.
        self.qdrant_service.setup_collection()

    def _extract_chapter_number(self, url: str) -> int | None:
        """
        Extract chapter number from a chapter URL.

        Example:
            /docs/chapter5/5.3-something
            -> 5
        """

        match = re.search(
            r"/docs/chapter(10|[1-9])/",
            url
        )

        if match:
            return int(match.group(1))

        return None

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split.
            chunk_size: Maximum chunk size in characters.
            overlap: Number of overlapping characters.

        Returns:
            List of text chunks.
        """

        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            # Avoid breaking a word when possible.
            if end < len(text) and not text[end].isspace():

                last_space = chunk.rfind(" ")

                if last_space != -1 and last_space > chunk_size // 2:
                    chunk = text[start:start + last_space + 1]
                    end = start + last_space + 1

            chunk = chunk.strip()

            if len(chunk) > 50:
                chunks.append(chunk)

            # Prevent infinite loops.
            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

            if start >= len(text):
                break

        return chunks

    def index_content(
        self,
        url: str,
        content: str,
        title: str = ""
    ) -> bool:
        """
        Chunk, batch-embed, and store one document/page.

        IMPORTANT:
        All chunks are embedded in batches rather than making one
        Cohere API call per chunk.
        """

        try:

            # ---------------------------------------------------------
            # 1. Chunk content
            # ---------------------------------------------------------

            chunks = self.chunk_text(
                content,
                chunk_size=1000,
                overlap=200
            )

            if not chunks:
                logger.warning(
                    "No valid chunks created from content at %s",
                    url
                )
                return False

            # ---------------------------------------------------------
            # 2. Create Document objects
            # ---------------------------------------------------------

            documents: List[Document] = []

            for i, chunk in enumerate(chunks):

                # Deterministic ID based on URL + chunk index.
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{url}#{i}"
                    )
                )

                chapter_number = self._extract_chapter_number(url)

                metadata = {
                    "source_url": url,
                    "chapter": chapter_number,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "title": title,
                    "original_content_length": len(content)
                }

                document = Document(
                    content_id=chunk_id,
                    source_url=url,
                    text_content=chunk,
                    metadata=metadata
                )

                documents.append(document)

            # ---------------------------------------------------------
            # 3. Extract all text chunks
            # ---------------------------------------------------------

            chunk_texts = [
                document.text_content
                for document in documents
            ]

            logger.info(
                "Generating embeddings for %d chunks from %s",
                len(chunk_texts),
                url
            )

            # ---------------------------------------------------------
            # 4. Batch embedding
            #
            # Cohere will internally split this into batches of up to
            # 96 texts per API call.
            # ---------------------------------------------------------

            embeddings = self.embedding_service.embed_texts(
                chunk_texts
            )

            if len(embeddings) != len(documents):
                raise RuntimeError(
                    f"Embedding count mismatch for {url}: "
                    f"{len(embeddings)} embeddings for "
                    f"{len(documents)} documents."
                )

            # ---------------------------------------------------------
            # 5. Store vectors in Qdrant
            # ---------------------------------------------------------

            success = self.qdrant_service.store_documents(
                documents,
                embeddings
            )

            if success:

                logger.info(
                    "Successfully indexed %d chunks from %s",
                    len(chunks),
                    url
                )

                return True

            logger.error(
                "Failed to store documents from %s in Qdrant",
                url
            )

            return False

        except Exception as e:

            logger.error(
                "Error indexing content from %s: %s",
                url,
                e
            )

            return False

    def index_multiple_contents(
        self,
        contents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index multiple content items.

        Each page is embedded in batches.
        """

        results = {
            "successful": 0,
            "failed": 0,
            "total": len(contents),
            "failed_items": []
        }

        for content_data in contents:

            url = content_data.get("url", "")
            content = content_data.get("content", "")
            title = content_data.get("title", "")

            if not content:

                logger.warning(
                    "Skipping %s - no content",
                    url
                )

                results["failed"] += 1

                results["failed_items"].append({
                    "url": url,
                    "reason": "No content"
                })

                continue

            if not _is_content_relevant(
                url,
                content,
                title
            ):

                logger.info(
                    "Skipping non-relevant content from %s",
                    url
                )

                results["failed"] += 1

                results["failed_items"].append({
                    "url": url,
                    "reason": "Irrelevant content"
                })

                continue

            success = self.index_content(
                url=url,
                content=content,
                title=title
            )

            if success:

                results["successful"] += 1

            else:

                results["failed"] += 1

                results["failed_items"].append({
                    "url": url,
                    "reason": "Indexing failed"
                })

        logger.info(
            "Indexing complete: %d successful, %d failed",
            results["successful"],
            results["failed"]
        )

        return results

    def index_from_sitemap(
        self,
        sitemap_url: str
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline.

        This function is called ONLY when the user explicitly
        enters the ingest command.
        """

        from src.ingestion.sitemap_parser import SitemapParser
        from src.ingestion.content_extractor import ContentExtractor

        sitemap_parser = SitemapParser()
        content_extractor = ContentExtractor()

        # ---------------------------------------------------------
        # 1. Parse sitemap
        # ---------------------------------------------------------

        logger.info(
            "Parsing sitemap: %s",
            sitemap_url
        )

        urls = sitemap_parser.parse_sitemap(sitemap_url)

        chapter_urls = sitemap_parser.filter_chapter_urls(urls)

        logger.info(
            f"Found {len(chapter_urls)} chapter URLs to process"
)

        if not chapter_urls:

            logger.warning(
                "No /docs/ URLs found in sitemap"
            )

            return {
                "successful": 0,
                "failed": 0,
                "total": 0,
                "failed_items": [],
                "message": "No /docs/ URLs found in sitemap"
            }

        # ---------------------------------------------------------
        # 2. Extract content
        # ---------------------------------------------------------

        logger.info(
            "Starting content extraction..."
        )

        extraction_results = (
            content_extractor.extract_multiple_contents(
                chapter_urls,
                delay=0.5
            )
        )

        successful_extractions = []

        for result in extraction_results:

            if (
                result
                and result.get("status") == "success"
            ):

                successful_extractions.append({
                    "url": result["url"],
                    "content": result["content"],
                    "title": result.get("title", "")
                })

        logger.info(
            "Successfully extracted content from %d URLs",
            len(successful_extractions)
        )

        # ---------------------------------------------------------
        # 3. Index content
        # ---------------------------------------------------------

        logger.info(
            "Starting indexing process..."
        )

        indexing_results = (
            self.index_multiple_contents(
                successful_extractions
            )
        )

        return indexing_results