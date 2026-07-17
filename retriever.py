from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain.retrievers import EnsembleRetriever 


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