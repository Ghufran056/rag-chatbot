#!/usr/bin/env python3
"""
Basic test to verify the RAG Chatbot components are working.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_config():
    """Test that configuration is properly loaded."""
    from src.utils.config import Config

    print("Testing configuration...")
    try:
        config = Config()
        # Skip validation to avoid requiring API keys for basic testing
        print("+ Configuration module loaded successfully")
        return True
    except Exception as e:
        print(f"X Configuration error: {e}")
        return False

def test_models():
    """Test that data models work."""
    print("\nTesting data models...")
    try:
        from src.models.document import Document
        from src.models.query import Query
        from src.models.retrieved_chunk import RetrievedChunk

        # Test Document model
        doc = Document(
            content_id="test123",
            source_url="https://example.com/test",
            text_content="This is test content"
        )

        # Test Query model
        query = Query(
            query_id="query123",
            query_text="What is this about?"
        )

        # Test RetrievedChunk model
        chunk = RetrievedChunk(
            chunk_text="This is a retrieved chunk",
            source_url="https://example.com/test",
            relevance_score=0.9
        )

        print("+ Data models work correctly")
        return True
    except Exception as e:
        print(f"X Data models error: {e}")
        return False

def test_services_structure():
    """Test that service modules can be imported without errors."""
    print("\nTesting service structure...")
    try:
        from src.services.embedding_service import EmbeddingService
        from src.services.qdrant_service import QdrantService

        # Don't initialize with API keys to avoid errors
        print("+ Service modules can be imported")
        return True
    except Exception as e:
        print(f"X Service structure error: {e}")
        return False

def test_agents_structure():
    """Test that agent modules can be imported without errors."""
    print("\nTesting agent structure...")
    try:
        from src.agents.textbook_agent import TextbookAgent, retrieve_content

        print("+ Agent modules can be imported")
        return True
    except Exception as e:
        print(f"X Agent structure error: {e}")
        return False

def test_ingestion_structure():
    """Test that ingestion modules can be imported without errors."""
    print("\nTesting ingestion structure...")
    try:
        from src.ingestion.sitemap_parser import SitemapParser
        from src.ingestion.content_extractor import ContentExtractor
        from src.ingestion.indexer import Indexer

        print("+ Ingestion modules can be imported")
        return True
    except Exception as e:
        print(f"X Ingestion structure error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running basic functionality tests for RAG Chatbot...\n")

    tests = [
        test_config,
        test_models,
        test_services_structure,
        test_agents_structure,
        test_ingestion_structure
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("All structural tests passed! The basic components are properly structured.")
        return 0
    else:
        print("Some structural tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())