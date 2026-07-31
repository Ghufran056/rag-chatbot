# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with OpenAI Agent SDK that answers questions based on indexed textbook content.

## Features

- Query textbook content with AI-powered responses
- Sitemap ingestion workflow for content indexing
- Vector search using Qdrant and Cohere embeddings
- CLI interface for interactive conversations

## Requirements

- Python 3.13+
- OpenAI API key (or Google API key for Gemini)
- Cohere API key
- Qdrant vector database

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
python main.py
```