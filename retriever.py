from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain.retrievers import EnsembleRetriever 
from rank_bm25 import BM25Okapi
import numpy as np


def generate_embeddings():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
    return embeddings


def create_vector_store(embeddings, documents: list[Document]):
    vector_store = Chroma(collection_name = "example_collection1", embedding_function = embeddings, persist_directory = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\chroma_langchain_db")
    vector_store.add_documents(documents = documents)
    return vector_store

def create_bm25_retriever(documents: list[Document]):
    tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def bm25_retriever_func(query, bm25, documents: list[Document], top_k=100):
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    return [(str(idx), doc_scores[idx]) for idx in top_indices]

def hybrid_retriever(query, vector_retriever, bm25_retriever,
                    rrf_constant = 60, v_weights = 0.5, b_weights = 0.5, top_k = 100):
    
    #Fetch results from retrivers
    v_result = vector_retriever(query, top_k)
    b_result = bm25_retriever(query, top_k)
    
    #Rank the retriver results
    vector_rank = {doc_id:rank for rank,(doc_id,_) in enumerate(v_result)}
    bm25_rank = {doc_id:rank for rank,(doc_id,_) in enumerate(b_result)}
    
    score = {}
    
    #Vector scores for rankings
    for doc_id, rank in vector_rank.items():
        score[doc_id] = score.get(doc_id,0.0) + (v_weights/(rank+rrf_constant))
        
    #BM25 scores for rankings
    for doc_id, rank in bm25_rank.items():
        score[doc_id] = score.get(doc_id,0.0) + (b_weights/(rank+rrf_constant))
        
    return sorted(score.items(), key = lambda x: x[1], reverse = True)


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
    vector_store = create_vector_store(embeddings, documents)
    
    # Create BM25 index
    bm25_index = create_bm25_retriever(documents)
    
    # Define vector retriever function
    def vector_retriever_func(query, top_k=100):
        return vector_store.similarity_search_with_score(query, k=top_k)
    
    # Define BM25 retriever function (wraps the bm25_retriever_func with fixed parameters)
    def bm25_wrapper(query, top_k=100):
        return bm25_retriever_func(query, bm25_index, documents, top_k)
    
    # Test hybrid retrieval
    query = "cat dog"
    results = hybrid_retriever(
        query=query,
        vector_retriever=vector_retriever_func,
        bm25_retriever=bm25_wrapper,
        rrf_constant=60,
        v_weights=0.5,
        b_weights=0.5,
        top_k=10
    )
    
    print(f"\nResults for query: '{query}'")
    print("-" * 50)
    for doc_id, score in results:
        doc_index = int(doc_id)
        print(f"Doc {doc_id}: {documents[doc_index].page_content}")
        print(f"Score: {score:.6f}\n")
