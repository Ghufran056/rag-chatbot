import requests
import trafilatura
from typing import Optional, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)

class ContentExtractor:
    """
    Extracts readable text content from HTML pages using trafilatura.
    """

    def __init__(self):
        """Initialize the content extractor."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Chatbot-Ingestion-Bot/1.0'
        })

    def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract readable text content from a URL.

        Args:
            url: URL to extract content from

        Returns:
            Dictionary with content and metadata, or None if extraction fails
        """
        try:
            # Fetch the page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Use trafilatura to extract the main content
            extracted_content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                no_fallback=False
            )

            if not extracted_content or len(extracted_content.strip()) < 50:
                logger.warning(f"Could not extract meaningful content from {url}")
                return None

            # Get additional metadata
            metadata = trafilatura.extract_metadata(response.text)

            title = metadata.title() if metadata else ""
            if not title:
                # Fallback to extracting title from HTML if trafilatura didn't get it
                import re
                title_match = re.search(r'<title[^>]*>(.*?)</title>', response.text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else ""

            return {
                "url": url,
                "title": title,
                "content": extracted_content.strip(),
                "status": "success",
                "content_length": len(extracted_content)
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "error": f"Request error: {str(e)}",
                "status": "request_error"
            }
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "status": "extraction_error"
            }

    def extract_content_from_html(self, html_content: str, url: str = "") -> Optional[str]:
        """
        Extract readable text content from raw HTML string.

        Args:
            html_content: Raw HTML string to extract content from
            url: Optional URL for context

        Returns:
            Extracted text content, or None if extraction fails
        """
        try:
            extracted_content = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                no_fallback=False
            )

            if not extracted_content or len(extracted_content.strip()) < 10:
                logger.warning(f"Could not extract meaningful content from {'HTML string' if not url else url}")
                return None

            return extracted_content.strip()

        except Exception as e:
            logger.error(f"Error extracting content from {'HTML string' if not url else url}: {str(e)}")
            return None

    def extract_multiple_contents(self, urls: list, delay: float = 0.5) -> list:
        """
        Extract content from multiple URLs with a delay between requests.

        Args:
            urls: List of URLs to extract content from
            delay: Delay in seconds between requests (to be respectful to servers)

        Returns:
            List of extraction results
        """
        results = []

        for i, url in enumerate(urls):
            logger.info(f"Extracting content {i+1}/{len(urls)}: {url}")

            result = self.extract_content(url)
            results.append(result)

            # Add delay between requests to be respectful to servers
            if i < len(urls) - 1:  # Don't delay after the last request
                time.sleep(delay)

        return results

    def validate_content(self, content: str, min_length: int = 50) -> bool:
        """
        Validate that extracted content meets minimum quality requirements.

        Args:
            content: Content to validate
            min_length: Minimum length required

        Returns:
            True if content is valid, False otherwise
        """
        if not content:
            return False

        # Check minimum length
        if len(content.strip()) < min_length:
            return False

        # Check for common non-content indicators
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in [
            'not found', '404 error', 'page not found', 'error 404'
        ]):
            return False

        return True