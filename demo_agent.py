#!/usr/bin/env python3
"""
Demo script to show the new OpenAI Agents SDK implementation
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demo_agent():
    """Demonstrate the new OpenAI Agents SDK implementation."""
    print("Demonstrating the new OpenAI Agents SDK implementation")
    print("=" * 60)

    try:
        from src.agents.textbook_agent import TextbookAgent

        print("+ TextbookAgent imported successfully")
        print("+ Using OpenAI Agents SDK with function_tool and Runner")

        # Show the agent structure
        agent = TextbookAgent()
        info = agent.get_assistant_info()

        print(f"+ Agent created with model: {info['model']}")
        print(f"+ Agent name: {info['name']}")

        print("\n[INFO] The agent is configured with:")
        print("   - System prompt for textbook Q&A")
        print("   - retrieve_content tool using @function_tool decorator")
        print("   - OpenAI Agents SDK Runner for execution")
        print("   - Proper RAG flow: retrieve before answering")

        print("\n[FEATURES] Key features of the new implementation:")
        print("   - Uses official OpenAI Agents SDK")
        print("   - Function tools with @function_tool decorator")
        print("   - Agent definition with instructions and tools")
        print("   - Runner.run_sync for synchronous execution")
        print("   - Maintains RAG flow with Qdrant and Cohere")
        print("   - Preserves all original functionality")

        print("\n[GUIDELINES] The agent follows these rules:")
        print("   1. Always call retrieve_content tool before answering")
        print("   2. Only use retrieved content to answer questions")
        print("   3. Respond 'This topic is not covered in the textbook' if no content found")
        print("   4. Attribute sources when providing information")

        print("\n[READY] Ready to process questions with the new agent architecture!")

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("[INFO] Make sure to install openai-agents: pip install openai-agents")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_agent()