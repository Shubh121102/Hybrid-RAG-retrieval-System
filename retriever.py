from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.retrievers import BM25Retriever

URL = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\chroma_langchain_db"


def generate_embeddings():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
    return embeddings


def create_vector_store(embeddings, documents: list[Document]):
    vector_store = Chroma(collection_name = "example_collection1", embedding_function = embeddings) #persist_directory = URL
    vector_store.add_documents(documents = documents)
    return vector_store


def create_bm25_retriever(documents: list[Document]):
    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3)
    return bm25_retriever

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""
    doc_scores = {}  # page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]



def test_query(query, name, retriever):
    if retriever != hybrid_retrieve:
        results = retriever.invoke(query)
    else:
        results = retriever(query = query, retrievers=[vector_store,bm25_retriever],weights=[0.5,0.5] )

    print(f'\\n{name} - Query: \"{query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'{i+1}. {preview}')
    return results



# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    # Sample documents
    documents = [
        Document(page_content="The cat sat on the mat", metadata={"id": "doc1"}),
        Document(page_content="The dog played with the ball", metadata={"id": "doc2"}),
        Document(page_content="Cats and dogs are great pets", metadata={"id": "doc3"}),
        Document(page_content="Machine learning is fascinating", metadata={"id": "doc4"}),
        Document(page_content="Deep learning models use neural networks", metadata={"id": "doc5"})
    ]
    
    # Initialize embeddings and vector store
    embeddings = generate_embeddings()
    vectordb = create_vector_store(embeddings, documents)
    vector_store = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    
    # Create BM25 index
    bm25_retriever = create_bm25_retriever(documents)

    query = "dog food delivery"
    # hybrid_retriever = hybrid_retrieve(query = query,retrievers=[vector_store,bm25_retriever],weights=[0.5,0.5])

    vector_result = test_query(query, "VECTOR", vector_store)
    print("="*50)
    bm25_result = test_query(query, "BM25", bm25_retriever)
    print("="*50)
    hybrid_result = test_query(query, "HYBRID", hybrid_retrieve)
    print("="*50)
    
