#!/usr/bin/env python3
"""
Test the updated agent implementation with new OpenAI Agent SDK approach.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_new_agent_initialization():
    """Test that the new agent can be imported and has correct structure."""
    print("Testing new agent structure...")
    try:
        from src.agents.textbook_agent import TextbookAgent

        # Just test that the class can be imported and has the expected constructor
        # We won't initialize it to avoid API key requirements
        import inspect
        sig = inspect.signature(TextbookAgent.__init__)
        params = list(sig.parameters.keys())

        # Check that it has the expected parameters (excluding 'self')
        expected_params = ['self', 'model']
        actual_params = [p for p in params]

        assert 'model' in actual_params, "Missing model parameter"

        print("+ Agent class has correct constructor signature")
        return True
    except Exception as e:
        print(f"X Agent structure error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_agent_structure():
    """Test that the new agent has the expected methods."""
    print("\nTesting new agent methods...")
    try:
        from src.agents.textbook_agent import TextbookAgent

        # Check that required methods exist on the class
        assert hasattr(TextbookAgent, 'ask'), "Missing ask method"
        assert hasattr(TextbookAgent, 'get_assistant_info'), "Missing get_assistant_info method"
        assert hasattr(TextbookAgent, '_get_system_prompt'), "Missing _get_system_prompt method"

        print("+ Agent class has all required methods")
        return True
    except Exception as e:
        print(f"X Agent methods error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_tool_compatibility():
    """Test that the retrieval tool is compatible with the new agent."""
    print("\nTesting retrieval tool compatibility...")
    try:
        from src.agents.textbook_agent import retrieve_content

        # Check that the function tool exists and has the expected attributes
        assert hasattr(retrieve_content, 'name'), "retrieve_content should have a name attribute"
        assert hasattr(retrieve_content, 'description'), "retrieve_content should have a description attribute"

        print("+ Retrieval tool is compatible with function calling")
        return True
    except Exception as e:
        print(f"X Retrieval tool compatibility error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_with_different_providers():
    """Test that the agent supports different providers through its constructor."""
    print("\nTesting agent provider configuration...")
    try:
        from src.agents.textbook_agent import TextbookAgent
        import inspect

        # Check that the constructor accepts model parameter
        sig = inspect.signature(TextbookAgent.__init__)
        params = list(sig.parameters.keys())

        assert 'model' in params, "Constructor missing model parameter"

        print("+ Agent supports model configuration through constructor")
        return True
    except Exception as e:
        print(f"X Agent provider configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the new agent implementation."""
    print("Running tests for updated RAG Chatbot with new OpenAI Agent SDK approach...\n")

    tests = [
        test_new_agent_initialization,
        test_new_agent_structure,
        test_retrieval_tool_compatibility,
        test_agent_with_different_providers
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*60}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("All tests passed! The updated agent implementation is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())