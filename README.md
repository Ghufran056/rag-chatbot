# 🤖 RAG Chatbot Backend

A FastAPI-based Retrieval-Augmented Generation (RAG) backend built using the OpenAI Agents SDK.

The chatbot retrieves relevant textbook content from Qdrant using Cohere embeddings before generating responses.

---

# Live Demo

https://ghufran056.github.io/AI-spec-driven-book

# 🤖 Backend API: 
https://rag-chatbot-l6uo.onrender.com

---

## 🚀 Features

- OpenAI Agents SDK
- FastAPI REST API
- Qdrant vector search
- Cohere embeddings
- Semantic retrieval
- Context-aware answers
- Modular architecture
- Ready for frontend integration

---

## 🛠️ Tech Stack

- Python
- FastAPI
- OpenAI Agents SDK
- Qdrant
- Cohere
- uv
- Pydantic

---

## ⚙️ API Endpoint

### POST

/api/chat

---

## 🔍 Retrieval Pipeline

```text
User Question
      │
      ▼
FastAPI API
      │
      ▼
OpenAI Agent
      │
      ▼
Generate Embedding (Cohere)
      │
      ▼
Qdrant Vector Search
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Generate Final Answer
      │
      ▼
Return Response
```

---

## 📌 Notes

- The chatbot answers questions using indexed textbook content.
- Semantic search is powered by Cohere embeddings and Qdrant vector search.
- The backend is designed to be consumed by any frontend via a REST API.

---
