import uuid
from typing import List, Dict, Any, Optional
import logging
from src.services.qdrant_service import QdrantService
from src.services.embedding_service import EmbeddingService
from src.models.document import Document
from src.utils.config import Config
import re

logger = logging.getLogger(__name__)

class Indexer:
    """
    Indexes content by chunking text, generating embeddings, and storing in Qdrant.
    """

    def __init__(self):
        """Initialize the indexer with required services."""
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.config = Config()

        # Setup collection if it doesn't exist
        self.qdrant_service.setup_collection()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks to preserve context.

        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Ensure we don't break in the middle of a word if possible
            if end < len(text) and not text[end].isspace():
                # Find the last space before the end to avoid breaking words
                last_space = chunk.rfind(' ')
                if last_space != -1 and last_space > chunk_size // 2:
                    chunk = text[start:start + last_space + 1]
                    end = start + last_space + 1

            chunks.append(chunk.strip())

            # Move start to create overlap
            start = end - overlap if end - overlap > start else end
            if start >= len(text):
                break

        # Remove any chunks that are too short
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

        return chunks

    def index_content(self, url: str, content: str, title: str = "") -> bool:
        """
        Index content by chunking, embedding, and storing in Qdrant.

        Args:
            url: Source URL of the content
            content: The content to index
            title: Title of the content (optional)

        Returns:
            True if indexing was successful
        """
        try:
            # Chunk the content with overlap
            chunks = self.chunk_text(content, chunk_size=1000, overlap=200)

            if not chunks:
                logger.warning(f"No valid chunks created from content at {url}")
                return False

            documents = []
            embeddings = []

            for i, chunk in enumerate(chunks):
                # Create a unique ID for this chunk
                chunk_id = f"{url}#{i}"

                # Create document metadata
                metadata = {
                    "source_url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "title": title,
                    "original_content_length": len(content)
                }

                # Create document object
                document = Document(
                    content_id=chunk_id,
                    source_url=url,
                    text_content=chunk,
                    metadata=metadata
                )

                # Generate embedding for the chunk
                embedding = self.embedding_service.embed_text(chunk)

                documents.append(document)
                embeddings.append(embedding)

            # Store all documents in Qdrant
            success = self.qdrant_service.store_documents(documents, embeddings)

            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks from {url} into Qdrant")
                return True
            else:
                logger.error(f"Failed to store documents from {url} in Qdrant")
                return False

        except Exception as e:
            logger.error(f"Error indexing content from {url}: {str(e)}")
            return False

    def index_multiple_contents(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index multiple content items.

        Args:
            contents: List of content dictionaries with 'url', 'content', and 'title' keys

        Returns:
            Dictionary with indexing results
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
                logger.warning(f"Skipping {url} - no content to index")
                results["failed"] += 1
                results["failed_items"].append({"url": url, "reason": "No content"})
                continue

            success = self.index_content(url, content, title)

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["failed_items"].append({"url": url, "reason": "Indexing failed"})

        logger.info(f"Indexing complete: {results['successful']} successful, {results['failed']} failed")
        return results

    def index_from_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """
        Full ingestion pipeline: parse sitemap, extract content, and index.

        Args:
            sitemap_url: URL to the sitemap.xml file

        Returns:
            Dictionary with ingestion results
        """
        from src.ingestion.sitemap_parser import SitemapParser
        from src.ingestion.content_extractor import ContentExtractor

        sitemap_parser = SitemapParser()
        content_extractor = ContentExtractor()

        # Parse sitemap to get URLs
        logger.info(f"Parsing sitemap: {sitemap_url}")
        urls = sitemap_parser.parse_sitemap(sitemap_url)
        chapter_urls = sitemap_parser.filter_chapter_urls(urls)

        logger.info(f"Found {len(chapter_urls)} /docs/ URLs to process")

        if not chapter_urls:
            logger.warning("No /docs/ URLs found in sitemap")
            return {
                "successful": 0,
                "failed": 0,
                "total": 0,
                "failed_items": [],
                "message": "No /docs/ URLs found in sitemap"
            }

        # Extract content from URLs
        logger.info("Starting content extraction...")
        extraction_results = content_extractor.extract_multiple_contents(chapter_urls, delay=0.5)

        # Filter successful extractions
        successful_extractions = []
        for result in extraction_results:
            if result and result.get("status") == "success":
                successful_extractions.append({
                    "url": result["url"],
                    "content": result["content"],
                    "title": result.get("title", "")
                })

        logger.info(f"Successfully extracted content from {len(successful_extractions)} URLs")

        # Index the extracted content
        logger.info("Starting indexing process...")
        indexing_results = self.index_multiple_contents(successful_extractions)

        return indexing_results