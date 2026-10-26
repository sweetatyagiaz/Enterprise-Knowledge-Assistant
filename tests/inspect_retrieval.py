
from ai_core.embeddings import EmbeddingService
from ai_core.retrieval.ranking import hybrid_rank
from ai_core.vector_store import VectorStore


questions = [
    {
        "question": "What are the main components of this project's RAG architecture?",
        "keywords": [
            "Document Loader",
            "Text Chunking",
            "OpenAI Embeddings",
            "SQLite Vector Store",
            "Similarity Search",
            "Retrieved Context",
            "OpenAI Chat Model",
            "FastAPI",
        ],
    },
    {
        "question": "Which embedding model does the project use by default?",
        "keywords": ["text-embedding-3-small", "1536"],
    },
        {
        "question": "What is the configured text chunk size?",
        "keywords": ["900", "CHUNK_SIZE"],
    },
    {
        "question": "How much overlap is configured between text chunks?",
        "keywords": ["150", "CHUNK_OVERLAP"],
    },
    {
        "question": "What is the maximum document upload size?",
        "keywords": ["15", "MAX_UPLOAD_MB"],
    },
]

embeddings = EmbeddingService()
store = VectorStore()

for item in questions:
    question = item["question"]
    keywords = item["keywords"]

    query_vector = embeddings.embed_query(question)
    candidates = store.search(query_vector, top_k=10)
    results = hybrid_rank(question, candidates)[:5]

    print(f"\nQUESTION: {question}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        text = result["text"]
        matched = [
            keyword
            for keyword in keywords
            if keyword.casefold() in text.casefold()
        ]

        print(f"\nRank: {rank}")
        print(f"Semantic score: {result['semantic_score']:.3f}")
        print(f"Lexical score:  {result['lexical_score']:.3f}")
        print(f"Hybrid score:   {result['hybrid_score']:.3f}")
        print(f"Keyword coverage: {len(matched)}/{len(keywords)}")
        print(f"Matches: {', '.join(matched) if matched else 'None'}")
        print("-" * 60)
        print(text)

    best_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if all(
                keyword.casefold() in result["text"].casefold()
                for keyword in keywords
            )
        ),
        None,
    )
    print(f"\nRank containing all expected keywords: {best_rank}")
