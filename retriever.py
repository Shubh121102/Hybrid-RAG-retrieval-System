from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.retrievers import EnsembleRetriever 


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