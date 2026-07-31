from src.services import qdrant_service


class FakeResult:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeQueryPointsClient:
    def __init__(self):
        self.calls = []

    def query_points(self, collection_name, query, limit, with_payload):
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        return type("QueryResponse", (), {"points": [FakeResult({"text_content": "hello", "source_url": "https://example.com", "metadata": {}}, 0.95)]})()


def test_retrieve_similar_uses_query_points_when_search_is_unavailable():
    service = qdrant_service.QdrantService.__new__(qdrant_service.QdrantService)
    fake_client = FakeQueryPointsClient()
    service.client = fake_client
    service.collection_name = "test_collection"

    chunks = service.retrieve_similar([0.1, 0.2], top_k=1)

    assert len(chunks) == 1
    assert chunks[0].chunk_text == "hello"
    assert chunks[0].source_url == "https://example.com"
    assert fake_client.calls[0]["collection_name"] == "test_collection"
    assert fake_client.calls[0]["query"] == [0.1, 0.2]
