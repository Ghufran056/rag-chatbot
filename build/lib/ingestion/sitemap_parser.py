import requests
import xml.etree.ElementTree as ET
from typing import List
import re
import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class SitemapParser:
    """
    Parser for sitemap.xml files to extract URLs for content ingestion.
    Only processes /docs/ URLs as per requirements.
    """

    def __init__(self):
        """Initialize the sitemap parser."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Chatbot-Ingestion-Bot/1.0'
        })

    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap.xml file and extract /docs/ URLs.

        Args:
            sitemap_url: URL to the sitemap.xml file

        Returns:
            List of URLs that match the /docs/ pattern
        """
        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()

            # Parse the XML content
            root = ET.fromstring(response.content)

            # Handle both regular sitemap and sitemap index
            if self._is_sitemap_index(root):
                return self._parse_sitemap_index(root, sitemap_url)
            else:
                return self._parse_regular_sitemap(root)
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {str(e)}")
            raise e

    def _is_sitemap_index(self, root: ET.Element) -> bool:
        """
        Check if the XML root represents a sitemap index (contains sitemap elements)
        rather than a regular sitemap (contains url elements).
        """
        # Common namespaces for sitemaps
        namespaces = {
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'xhtml': 'http://www.w3.org/1999/xhtml',
            'image': 'http://www.google.com/schemas/sitemap-image/1.1',
            'video': 'http://www.google.com/schemas/sitemap-video/1.1'
        }

        # Check for sitemap elements (indicating a sitemap index)
        for prefix, uri in namespaces.items():
            if root.findall(f'.//{{{uri}}}sitemap'):
                return True
        if root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
            return True
        if root.findall('.//sitemap'):
            return True

        return False

    def _parse_sitemap_index(self, root: ET.Element, base_url: str) -> List[str]:
        """
        Parse a sitemap index file and extract URLs from all included sitemaps.
        """
        urls = []
        namespaces = {
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }

        # Find all sitemap locations
        for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
            loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_elem is not None and loc_elem.text:
                sitemap_url = loc_elem.text.strip()
                # Resolve relative URLs
                if sitemap_url.startswith('/'):
                    sitemap_url = urljoin(base_url, sitemap_url)

                logger.info(f"Parsing nested sitemap: {sitemap_url}")
                try:
                    nested_urls = self.parse_sitemap(sitemap_url)
                    urls.extend(nested_urls)
                except Exception as e:
                    logger.warning(f"Failed to parse nested sitemap {sitemap_url}: {str(e)}")
                    continue

        return urls

    def _parse_regular_sitemap(self, root: ET.Element) -> List[str]:
        """
        Parse a regular sitemap file and extract /docs/ URLs.
        """
        urls = []
        namespaces = {
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }

        # Find all URL elements
        for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_elem is not None and loc_elem.text:
                url = loc_elem.text.strip()

                # Only include URLs that match the /docs/ pattern
                if self._is_docs_url(url):
                    urls.append(url)

        return urls

    def _is_docs_url(self, url: str) -> bool:
        """
        Check if a URL is a /docs/ URL as required by the specification.
        """
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Check if path starts with /docs/
        return path.startswith('/docs/')

    def filter_chapter_urls(self, urls: List[str]) -> List[str]:
        """
        Filter a list of URLs to only include /docs/ URLs.

        Args:
            urls: List of URLs to filter

        Returns:
            List of URLs that match the /docs/ pattern
        """
        return [url for url in urls if self._is_docs_url(url)]

    def get_sitemap_urls(self, base_url: str) -> List[str]:
        """
        Try to find and parse the sitemap at common locations relative to a base URL.

        Args:
            base_url: Base URL to look for sitemap

        Returns:
            List of /docs/ URLs found in sitemap
        """
        # Common sitemap locations
        sitemap_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap/sitemap.xml',
            '/sitemap-index.xml'
        ]

        for location in sitemap_locations:
            sitemap_url = urljoin(base_url, location)
            try:
                logger.info(f"Trying sitemap location: {sitemap_url}")
                urls = self.parse_sitemap(sitemap_url)
                if urls:
                    logger.info(f"Found {len(urls)} URLs in sitemap at {sitemap_url}")
                    return urls
            except Exception as e:
                logger.debug(f"Sitemap not found at {sitemap_url}: {str(e)}")
                continue

        logger.warning(f"No sitemap found at common locations for {base_url}")
        return []