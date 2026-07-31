#!/usr/bin/env python3
"""
Main entry point for the RAG Chatbot CLI application.
"""

import sys
import signal
import logging

from src.chatbot_agents.textbook_agent import ask
from src.ingestion.indexer import Indexer
from src.utils.config import Config


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""

    print("\n\nShutting down the chatbot... Goodbye!")
    sys.exit(0)


def main():
    """Main function to run the CLI chatbot."""

    signal.signal(
        signal.SIGINT,
        signal_handler
    )

    print("RAG Chatbot with OpenAI Agent SDK")
    print("=" * 50)

    print("Initializing system...")

    try:

        # ---------------------------------------------------------
        # Validate configuration
        # ---------------------------------------------------------

        config = Config()
        config.validate()

        print(
            "\nChatbot is ready! "
            "You can start asking questions about "
            "the textbook content."
        )

        print(
            "Type 'quit', 'exit', or 'bye' to exit."
        )

        print(
            "Type 'ingest <sitemap_url>' to ingest "
            "textbook content."
        )

        print("-" * 50)

        while True:

            try:

                user_input = input(
                    "\nYou: "
                ).strip()

                if not user_input:
                    continue

                # -------------------------------------------------
                # Exit
                # -------------------------------------------------

                if user_input.lower() in [
                    "quit",
                    "exit",
                    "bye"
                ]:

                    print(
                        "Assistant: Goodbye! "
                        "Thanks for using the textbook chatbot."
                    )

                    break

                # -------------------------------------------------
                # Ingestion
                # -------------------------------------------------

                if user_input.lower().startswith(
                    "ingest "
                ):

                    sitemap_url = (
                        user_input[7:].strip()
                    )

                    if not sitemap_url:

                        print(
                            "Please provide a sitemap URL.\n"
                            "Usage: ingest <sitemap_url>"
                        )

                        continue

                    print(
                        f"Starting ingestion from: "
                        f"{sitemap_url}"
                    )

                    try:

                        indexer = Indexer()

                        results = (
                            indexer.index_from_sitemap(
                                sitemap_url
                            )
                        )

                        print(
                            "\nIngestion completed:"
                        )

                        print(
                            f"  - Successful: "
                            f"{results['successful']} URLs"
                        )

                        print(
                            f"  - Failed: "
                            f"{results['failed']} URLs"
                        )

                        print(
                            f"  - Total processed: "
                            f"{results['total']} URLs"
                        )

                        if results["failed_items"]:

                            print(
                                "\nFailed URLs:"
                            )

                            for item in (
                                results["failed_items"][:5]
                            ):

                                print(
                                    f"  - {item['url']}: "
                                    f"{item['reason']}"
                                )

                            if len(
                                results["failed_items"]
                            ) > 5:

                                print(
                                    f"  ... and "
                                    f"{len(results['failed_items']) - 5} "
                                    f"more"
                                )

                    except Exception as ingest_error:

                        logger.exception(
                            "Ingestion failed: %s",
                            ingest_error
                        )

                        print(
                            f"Assistant: Ingestion failed: "
                            f"{ingest_error}"
                        )

                    continue

                # -------------------------------------------------
                # Normal question
                # -------------------------------------------------

                print(
                    "Assistant: Thinking...",
                    end="",
                    flush=True
                )

                response = ask(
                    user_input
                )

                # Clear "Thinking..."
                sys.stdout.write(
                    "\r"
                )

                sys.stdout.flush()

                print(
                    f"Assistant: {response}"
                )

            except KeyboardInterrupt:

                print(
                    "\n\nShutting down the chatbot... Goodbye!"
                )

                break

            except Exception as e:

                logger.exception(
                    "Error in chat loop: %s",
                    e
                )

                print(
                    f"Assistant: Sorry, "
                    f"I encountered an error: {e}"
                )

    except Exception as e:

        logger.exception(
            "Failed to initialize chatbot: %s",
            e
        )

        print(
            f"Error: Failed to initialize chatbot: {e}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()