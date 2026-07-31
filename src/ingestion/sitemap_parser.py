import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests


logger = logging.getLogger(__name__)


class SitemapParser:
    """
    Parser for sitemap.xml files.

    Supports:
    - Regular sitemap.xml files
    - Sitemap index files
    - Nested sitemaps
    - Filtering URLs to only textbook chapter pages
    """

    def __init__(self):
        """Initialize the sitemap parser."""

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "RAG-Chatbot-Ingestion-Bot/1.0"
        })

    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap.xml file and extract valid URLs.

        Args:
            sitemap_url: URL to sitemap.xml

        Returns:
            List of URLs discovered in the sitemap.
        """

        try:
            response = self.session.get(
                sitemap_url,
                timeout=30
            )

            response.raise_for_status()

            root = ET.fromstring(
                response.content
            )

            # Sitemap index
            if self._is_sitemap_index(root):
                return self._parse_sitemap_index(
                    root,
                    sitemap_url
                )

            # Regular sitemap
            return self._parse_regular_sitemap(
                root
            )

        except Exception as e:

            logger.error(
                "Error parsing sitemap %s: %s",
                sitemap_url,
                e
            )

            raise

    def _is_sitemap_index(
        self,
        root: ET.Element
    ) -> bool:
        """
        Determine whether the XML is a sitemap index.
        """

        namespaces = [
            "http://www.sitemaps.org/schemas/sitemap/0.9",
            "http://www.google.com/schemas/sitemap/0.9",
            "http://www.w3.org/1999/xhtml",
            "http://www.google.com/schemas/sitemap-image/1.1",
            "http://www.google.com/schemas/sitemap-video/1.1",
        ]

        for uri in namespaces:

            if root.findall(
                f".//{{{uri}}}sitemap"
            ):
                return True

        # Handle sitemap XML without namespace.
        if root.findall(".//sitemap"):
            return True

        return False

    def _parse_sitemap_index(
        self,
        root: ET.Element,
        base_url: str
    ) -> List[str]:
        """
        Parse a sitemap index and recursively parse nested sitemaps.
        """

        urls: List[str] = []

        sitemap_namespaces = [
            "http://www.sitemaps.org/schemas/sitemap/0.9",
            "http://www.google.com/schemas/sitemap/0.9",
        ]

        for namespace in sitemap_namespaces:

            sitemap_tag = f"{{{namespace}}}sitemap"
            loc_tag = f"{{{namespace}}}loc"

            for sitemap_elem in root.findall(
                f".//{sitemap_tag}"
            ):

                loc_elem = sitemap_elem.find(
                    loc_tag
                )

                if (
                    loc_elem is None
                    or not loc_elem.text
                ):
                    continue

                nested_sitemap_url = (
                    loc_elem.text.strip()
                )

                if nested_sitemap_url.startswith("/"):
                    nested_sitemap_url = urljoin(
                        base_url,
                        nested_sitemap_url
                    )

                logger.info(
                    "Parsing nested sitemap: %s",
                    nested_sitemap_url
                )

                try:

                    nested_urls = (
                        self.parse_sitemap(
                            nested_sitemap_url
                        )
                    )

                    urls.extend(
                        nested_urls
                    )

                except Exception as e:

                    logger.warning(
                        "Failed to parse nested sitemap %s: %s",
                        nested_sitemap_url,
                        e
                    )

        return list(
            dict.fromkeys(urls)
        )

    def _parse_regular_sitemap(
        self,
        root: ET.Element
    ) -> List[str]:
        """
        Parse a regular sitemap and extract URLs.
        """

        urls: List[str] = []

        sitemap_namespaces = [
            "http://www.sitemaps.org/schemas/sitemap/0.9",
            "http://www.google.com/schemas/sitemap/0.9",
        ]

        for namespace in sitemap_namespaces:

            url_tag = f"{{{namespace}}}url"
            loc_tag = f"{{{namespace}}}loc"

            for url_elem in root.findall(
                f".//{url_tag}"
            ):

                loc_elem = url_elem.find(
                    loc_tag
                )

                if (
                    loc_elem is None
                    or not loc_elem.text
                ):
                    continue

                url = loc_elem.text.strip()

                if self._is_content_url(url):
                    urls.append(url)

        # Remove duplicates while preserving order.
        return list(
            dict.fromkeys(urls)
        )

    def _is_content_url(
        self,
        url: str
    ) -> bool:
        """
        Basic validation for URLs.

        This method does NOT decide whether a URL is a textbook
        chapter. That is handled separately by filter_chapter_urls().
        """

        if not url:
            return False

        parsed = urlparse(url)
        path = parsed.path.lower()

        # Ignore homepage.
        if not path or path == "/":
            return False

        excluded_segments = (
            "/api/",
            "/admin/",
            "/login",
            "/logout",
            "/signup",
        )

        if any(
            path.startswith(segment)
            for segment in excluded_segments
        ):
            return False

        return True

    def filter_docs_urls(
        self,
        urls: List[str]
    ) -> List[str]:
        """
        Filter URLs to documentation-like pages.

        This method is kept for compatibility with the existing
        codebase.
        """

        return [
            url
            for url in urls
            if self._is_content_url(url)
        ]

    def filter_chapter_urls(
        self,
        urls: List[str]
    ) -> List[str]:
        """
        Keep ONLY textbook chapter URLs.

        Expected Docusaurus structure:

            /docs/chapter1/...
            /docs/chapter2/...
            ...
            /docs/chapter10/...

        Examples that WILL match:

            /docs/chapter1/1.1-what-is-physical-ai
            /docs/chapter1/1.2-embodied-intelligence
            /docs/chapter5/5.3-something
            /docs/chapter10/10.1-something

        Examples that will NOT match:

            /
            /docs/intro
            /docs/tutorial/...
            /blog/...
            /about
        """

        chapter_urls: List[str] = []

        chapter_pattern = re.compile(
            r"/docs/chapter(10|[1-9])/",
            re.IGNORECASE
        )

        for url in urls:

            if chapter_pattern.search(url):
                chapter_urls.append(url)

        # Remove duplicates while preserving order.
        chapter_urls = list(
            dict.fromkeys(chapter_urls)
        )

        logger.info(
            "Filtered %d chapter URLs from %d total URLs",
            len(chapter_urls),
            len(urls)
        )

        return chapter_urls

    def extract_chapter_number(
        self,
        url: str
    ) -> Optional[int]:
        """
        Extract the chapter number from a chapter URL.

        Example:

            /docs/chapter5/5.3-something
            -> 5

        Returns:
            Chapter number or None if the URL is not a chapter URL.
        """

        match = re.search(
            r"/docs/chapter(10|[1-9])/",
            url,
            re.IGNORECASE
        )

        if not match:
            return None

        return int(
            match.group(1)
        )

    def get_sitemap_urls(
        self,
        base_url: str
    ) -> List[str]:
        """
        Try common sitemap locations.

        Args:
            base_url: Base website URL.

        Returns:
            URLs found in the sitemap.
        """

        sitemap_locations = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap/sitemap.xml",
            "/sitemap-index.xml",
            "/sitemap.xml.gz",
            "/sitemap_index.xml.gz",
        ]

        for location in sitemap_locations:

            sitemap_url = urljoin(
                base_url,
                location
            )

            try:

                logger.info(
                    "Trying sitemap location: %s",
                    sitemap_url
                )

                urls = self.parse_sitemap(
                    sitemap_url
                )

                if urls:

                    logger.info(
                        "Found %d URLs in sitemap at %s",
                        len(urls),
                        sitemap_url
                    )

                    return urls

            except Exception as e:

                logger.debug(
                    "Sitemap not found at %s: %s",
                    sitemap_url,
                    e
                )

        logger.warning(
            "No sitemap found at common locations for %s",
            base_url
        )

        return []