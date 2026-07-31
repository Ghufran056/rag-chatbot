from agents import Agent, Runner, function_tool
from typing import Dict, Any, List
from src.services.qdrant_service import QdrantService
from src.services.embedding_service import EmbeddingService
from src.models.query import Query
from src.utils.config import Config
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

@function_tool
def retrieve_content(query: str) -> str:
    """
    Retrieve relevant content from the textbook based on the user's query.
    This tool must be called before answering any question.
    """
    try:
        # Initialize services
        config = Config()
        config.validate()
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        # Setup collection if it doesn't exist
        qdrant_service.setup_collection()

        # Create a Query object
        query_obj = Query(
            query_id=str(uuid.uuid4()),
            query_text=query,
            created_at=datetime.utcnow()
        )

        # Generate embedding for the query
        query_embedding = embedding_service.embed_query(query)
        query_obj.query_embedding = query_embedding

        # Retrieve similar content from Qdrant
        retrieved_chunks = qdrant_service.retrieve_similar(
            query_embedding=query_embedding,
            top_k=5  # Retrieve top 5 most relevant chunks
        )

        # Format the results
        if not retrieved_chunks:
            return "No relevant content found in the textbook. This topic is not covered in the textbook."

        # Prepare the response
        formatted_content = "RETRIEVED TEXTBOOK CONTENT:\n\n"
        sources = set()

        for i, chunk in enumerate(retrieved_chunks, 1):
            formatted_content += f"Source {i} ({chunk.source_url}):\n"
            formatted_content += f"{chunk.chunk_text}\n\n"
            sources.add(chunk.source_url)

        formatted_content += f"\nSources: {', '.join(sources)}"
        return formatted_content

    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        return f"Error retrieving content from the textbook: {str(e)}"


class TextbookAgent:
    """
    An agent that answers questions based on textbook content using OpenAI Agents SDK.
    The agent always retrieves content from Qdrant before answering.
    """

    def __init__(self, model: str):
        """Initialize the textbook agent with required services."""
        self.config = Config()
        self.config.validate()

        self.model = model

        # Create the agent using the OpenAI Agents SDK
        self.agent = Agent(
            name="Textbook Q&A Assistant",
            instructions=self._get_system_prompt(),
            tools=[retrieve_content],
            model=model
        )

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt with safety rules and instructions.
        """
        return """
        You are a textbook Q&A assistant. Your purpose is to answer questions based only on the content provided in the textbook.

        IMPORTANT RULES:
        1. ALWAYS call the retrieve_content tool before answering any question
        2. ONLY use information from the retrieved content to answer questions
        3. If the retrieved content does not contain information to answer the question, respond with: "This topic is not covered in the textbook."
        4. Do not use any external knowledge or make up information
        5. Attribute sources when providing information from the textbook
        6. Keep responses concise and focused on the question
        7. If the retrieved content is empty or irrelevant, respond with: "This topic is not covered in the textbook."
        """

    def ask(self, question: str) -> str:
        """
        Ask a question to the agent and get a response.

        Args:
            question: The question to ask

        Returns:
            The agent's response
        """
        try:
            # Run the agent with the provided question
            result = Runner.run_sync(
                starting_agent=self.agent,
                input=question
            )

            # Return the final output
            return result.final_output

        except Exception as e:
            logger.error(f"Error in agent ask: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def get_assistant_info(self) -> Dict[str, Any]:
        """Get information about the assistant."""
        return {
            "model": self.model,
            "name": self.agent.name
        }