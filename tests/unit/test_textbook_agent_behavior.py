import importlib


class FakeChunk:
    def __init__(self, text, source_url="https://example.com/docs/chapter-1"):
        self.chunk_text = text
        self.source_url = source_url
        self.relevance_score = 1.0
        self.metadata = {}


class FakeEmbeddingService:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return [0.1] * 1024


class FakeQdrantService:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def setup_collection(self):
        self.calls.append("setup_collection")

    def search_all_chunks(self):
        self.calls.append("search_all_chunks")
        return [FakeChunk("Chapter 1 introduces embodied intelligence.")]


class FakeConfig:
    def validate(self):
        return True


def test_regular_questions_do_not_reembed(monkeypatch):
    import src.chatbot_agents.textbook_agent as textbook_agent

    monkeypatch.setattr(textbook_agent, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(textbook_agent, "QdrantService", FakeQdrantService)
    monkeypatch.setattr(textbook_agent, "Config", FakeConfig)
    monkeypatch.setattr(textbook_agent, "_should_rebuild_index", lambda: False)

    imported_module = importlib.reload(textbook_agent)
    response = imported_module.retrieve_content("What does chapter 1 cover?")

    assert "RETRIEVED TEXTBOOK CONTENT" in response
    assert "embodied intelligence" in response.lower()
