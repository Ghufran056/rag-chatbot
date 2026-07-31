from typing import Dict, Any
import re
import logging
import uuid
from datetime import datetime

from agents import Agent, Runner

from src.services.qdrant_service import QdrantService
from src.services.embedding_service import EmbeddingService
from src.models.query import Query
from src.utils.config import Config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Textbook answer agent
# ---------------------------------------------------------

def _detect_chapter_request(query: str) -> int | None:
    """
    Detect whether the user is asking about a specific chapter.

    Examples:
        "summarize chapter 5" -> 5
        "write summary of chapter 3" -> 3
        "explain chapter 7" -> 7

    Returns:
        Chapter number, or None if this is not a chapter request.
    """

    match = re.search(
        r"\bchapter\s*(10|[1-9])\b",
        query.lower()
    )

    if not match:
        return None

    return int(match.group(1))

def _get_answer_agent() -> Agent:
    """
    Create the LLM agent responsible for generating the final
    natural-language answer from retrieved textbook context.
    """

    try:
        from src.chatbot_agents.configuration import model
    except ImportError:
        model = None

    instructions = """
You are a helpful textbook question-answering assistant.

Your job is to answer the user's question using ONLY the
retrieved textbook context provided in the input.

Rules:

1. Stay grounded in the textbook context.
2. Do not invent facts that are not supported by the context.
3. Answer naturally, like a knowledgeable tutor speaking to a student.
4. Do not mention embeddings, vectors, Qdrant, retrieval, RAG,
   API calls, or internal system details unless the user explicitly
   asks about them.
5. Do not simply copy the retrieved passages word-for-word.
   Understand them and explain them naturally.
6. Keep the answer concise but complete.
7. Use simple language unless the textbook requires technical terminology.
8. When useful, give a short example or explanation from the context.
9. If the retrieved context does not contain enough information
   to answer the question, say:
   "I couldn't find enough information about that in the textbook."
10. Never pretend that information is in the textbook when it is not.

Answer format:

- Give the direct answer first.
- Add a brief explanation when helpful.
- Use bullets only when they improve clarity.
"""

    if model is not None:
        return Agent(
            name="Textbook Q&A Assistant",
            instructions=instructions,
            model=model
        )

    # If the project's configuration does not provide a model,
    # let the Agents SDK use its configured/default model.
    return Agent(
        name="Textbook Q&A Assistant",
        instructions=instructions
    )


def retrieve_content(query: str) -> str:
    """
    Retrieve textbook content.

    Normal questions:
        semantic search using top 8 results.

    Chapter questions:
        retrieve all chunks belonging to the requested chapter.
    """

    try:

        config = Config()
        config.validate()

        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        qdrant_service.setup_collection()

        query_obj = Query(
            query_id=str(uuid.uuid4()),
            query_text=query,
            created_at=datetime.utcnow()
        )

        # ---------------------------------------------------------
        # Detect chapter-level question
        # ---------------------------------------------------------

        chapter_number = _detect_chapter_request(query)

        # ---------------------------------------------------------
        # Chapter-specific retrieval
        # ---------------------------------------------------------

        if chapter_number is not None:

            logger.info(
                "Detected chapter-level request for chapter %d",
                chapter_number
            )

            retrieved_chunks = (
                qdrant_service.retrieve_chapter_chunks(
                    chapter_number
                )
            )

            if not retrieved_chunks:

                return (
                    f"NO_RELEVANT_CONTENT: "
                    f"No content from chapter {chapter_number} "
                    f"was found in the textbook."
                )

            context_parts = [
                f"RETRIEVED TEXTBOOK CONTENT "
                f"FROM CHAPTER {chapter_number}:\n"
            ]

            for index, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                context_parts.append(
                    f"""
--- Chapter {chapter_number} Passage {index} ---
Source: {chunk.source_url}

{chunk.chunk_text}
"""
                )

            return "\n".join(context_parts)

        # ---------------------------------------------------------
        # Normal semantic retrieval
        # ---------------------------------------------------------

        try:

            query_embedding = (
                embedding_service.embed_query(
                    query
                )
            )

            query_obj.query_embedding = query_embedding

        except Exception as exc:

            logger.error(
                "Query embedding failed: %s",
                exc
            )

            return (
                "ERROR: I cannot search the textbook right now "
                "because the embedding service is unavailable "
                "or its API quota has been exhausted."
            )

        try:

            retrieved_chunks = (
                qdrant_service.retrieve_similar(
                    query_embedding=query_embedding,
                    top_k=8
                )
            )

        except Exception as exc:

            logger.error(
                "Qdrant retrieval failed: %s",
                exc
            )

            return (
                "ERROR: I could not retrieve the textbook "
                "content from Qdrant."
            )

        if not retrieved_chunks:

            return (
                "NO_RELEVANT_CONTENT: "
                "No relevant textbook content was found."
            )

        context_parts = [
            "RETRIEVED TEXTBOOK CONTENT:\n"
        ]

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            context_parts.append(
                f"""
--- Textbook Passage {index} ---
Source: {chunk.source_url}
Relevance score: {chunk.relevance_score:.4f}

{chunk.chunk_text}
"""
            )

        return "\n".join(context_parts)

    except Exception as exc:

        logger.exception(
            "Error retrieving textbook content: %s",
            exc
        )

        return (
            f"ERROR: Error retrieving textbook content: {exc}"
        )
    """
    Retrieve relevant textbook content for a user query.

    IMPORTANT:
    This function embeds ONLY the user's query.
    The textbook embeddings already exist in Qdrant.
    """

    try:

        # ---------------------------------------------------------
        # Initialize services
        # ---------------------------------------------------------

        config = Config()
        config.validate()

        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        # Only creates the collection when missing.
        qdrant_service.setup_collection()

        # ---------------------------------------------------------
        # Create query object
        # ---------------------------------------------------------

        query_obj = Query(
            query_id=str(uuid.uuid4()),
            query_text=query,
            created_at=datetime.utcnow()
        )

        # ---------------------------------------------------------
        # Embed ONLY the user query
        # ---------------------------------------------------------

        try:

            query_embedding = (
                embedding_service.embed_query(
                    query
                )
            )

            query_obj.query_embedding = query_embedding

        except Exception as exc:

            logger.error(
                "Query embedding failed: %s",
                exc
            )

            return (
                "ERROR: I cannot search the textbook right now "
                "because the embedding service is unavailable "
                "or its API quota has been exhausted."
            )

        # ---------------------------------------------------------
        # Retrieve top 8 chunks
        # ---------------------------------------------------------

        try:

            retrieved_chunks = (
                qdrant_service.retrieve_similar(
                    query_embedding=query_embedding,
                    top_k=8
                )
            )

        except Exception as exc:

            logger.error(
                "Qdrant retrieval failed: %s",
                exc
            )

            return (
                "ERROR: I could not retrieve the textbook "
                "content from the vector database."
            )

        # ---------------------------------------------------------
        # No results
        # ---------------------------------------------------------

        if not retrieved_chunks:

            return (
                "NO_RELEVANT_CONTENT: "
                "No relevant textbook content was found."
            )

        # ---------------------------------------------------------
        # Build context
        # ---------------------------------------------------------

        context_parts = [
            "RETRIEVED TEXTBOOK CONTEXT:\n"
        ]

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            score = chunk.relevance_score

            context_parts.append(
                f"""
--- Textbook Passage {index} ---
Source: {chunk.source_url}
Relevance score: {score:.4f}

{chunk.chunk_text}
"""
            )

        return "\n".join(context_parts)

    except Exception as exc:

        logger.exception(
            "Error retrieving textbook content: %s",
            exc
        )

        return (
            f"ERROR: Error retrieving textbook content: {exc}"
        )


def _generate_grounded_answer(
    question: str,
    context: str
) -> str:
    """
    Use the OpenAI Agent to generate a natural, grounded answer.
    """

    # ---------------------------------------------------------
    # Handle retrieval errors
    # ---------------------------------------------------------

    if not context:
        return (
            "I couldn't find enough information about that "
            "in the textbook."
        )

    if context.startswith("ERROR:"):

        return context.replace(
            "ERROR:",
            ""
        ).strip()

    if context.startswith(
        "NO_RELEVANT_CONTENT:"
    ):

        return (
            "I couldn't find enough information about that "
            "in the textbook."
        )

    # ---------------------------------------------------------
    # Build final input for the LLM
    # ---------------------------------------------------------

    user_input = f"""
User question:
{question}

{context}

Please answer the user's question using only the textbook
passages above.
"""

    try:

        agent = _get_answer_agent()

        result = Runner.run_sync(
            agent,
            user_input
        )

        answer = str(
            result.final_output
        ).strip()

        if not answer:

            return (
                "I couldn't generate an answer from the "
                "retrieved textbook content."
            )

        return answer

    except Exception as exc:

        logger.exception(
            "Error generating grounded answer: %s",
            exc
        )

        return (
            "I retrieved the relevant textbook content, "
            "but I couldn't generate the final answer right now."
        )


def ask(question: str) -> str:
    """
    Ask a question to the textbook assistant.
    """

    if not question or not question.strip():

        return (
            "Please enter a question."
        )

    try:

        retrieved_context = retrieve_content(
            question.strip()
        )

        return _generate_grounded_answer(
            question.strip(),
            retrieved_context
        )
       

    except Exception as exc:

        logger.exception(
            "Error in agent ask: %s",
            exc
        )

        return (
            "Sorry, I encountered an error "
            "processing your request."
        )


def get_assistant_info() -> Dict[str, Any]:
    """Get information about the assistant."""

    return {
        "model": "grounded-textbook-answer",
        "name": "Textbook Q&A Assistant"
    }