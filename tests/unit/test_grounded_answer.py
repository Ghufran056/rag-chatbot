import pytest

from src.chatbot_agents import textbook_agent


def test_ask_uses_retrieved_text(monkeypatch):
    monkeypatch.setattr(
        textbook_agent,
        "retrieve_content",
        lambda query: "The book explains embodied intelligence as a system that learns from interaction.",
    )

    captured = {}

    def fake_generate(question, context):
        captured["question"] = question
        captured["context"] = context
        return "According to the textbook, embodied intelligence is learned through interaction."

    monkeypatch.setattr(textbook_agent, "_generate_grounded_answer", fake_generate)

    response = textbook_agent.ask("What is embodied intelligence?")

    assert response == "According to the textbook, embodied intelligence is learned through interaction."
    assert captured["question"] == "What is embodied intelligence?"
    assert "embodied intelligence" in captured["context"].lower()
