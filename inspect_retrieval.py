from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService

query = 'chapter count'
emb = EmbeddingService().embed_query(query)
svc = QdrantService()
chunks = svc.retrieve_similar(query_embedding=emb, top_k=10)
print('chunk-count', len(chunks))
for i, c in enumerate(chunks, 1):
    print('--- chunk', i, '---')
    print(c.source_url)
    print(c.chunk_text[:600])
    print()
