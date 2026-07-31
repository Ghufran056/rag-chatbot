# Quickstart Guide: RAG Chatbot

This guide will help you get the RAG Chatbot up and running quickly.

## Prerequisites

- Python 3.13+
- OpenAI API key
- Cohere API key
- Qdrant vector database (local or cloud)

## Setup

1. **Clone and setup the project:**
   ```bash
   cd rag-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

## Basic Usage

Once the chatbot is running:

1. **Ask questions:** Simply type your question and press Enter
2. **Ingest content:** Use the command `ingest <sitemap_url>` to add content from a sitemap
3. **Exit:** Type `quit`, `exit`, or `bye` to exit the chat

Example:
```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...
```

## Ingestion Example

To ingest content from a sitemap:
```
You: ingest https://example.com/sitemap.xml
Assistant: Starting ingestion from: https://example.com/sitemap.xml
Ingestion completed:
  - Successful: 25 URLs
  - Failed: 0 URLs
  - Total processed: 25 URLs
```

## Architecture Overview

The RAG Chatbot consists of:

- **Agents**: Textbook agent using OpenAI Agent SDK with retrieval tool
- **Services**: Qdrant service for vector storage, Cohere for embeddings
- **Ingestion**: Sitemap parser, content extractor, and indexer
- **Models**: Document, Query, and RetrievedChunk data models

## Troubleshooting

- **API Key Issues**: Ensure your OPENAI_API_KEY and COHERE_API_KEY are set correctly
- **Qdrant Connection**: Check that Qdrant is running and accessible
- **Empty Results**: Verify content has been ingested before asking questions