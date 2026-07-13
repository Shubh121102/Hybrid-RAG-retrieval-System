from langchain_core.documents import Document
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional 
from dotenv import load_dotenv
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
    

def load_pdf(file_path:str)-> list[Document]:
    """
    Load a PDF file and return a list of Document objects.

    Args:
        file_path (str): The path to the PDF file."""
    reader = pypdf.PdfReader(file_path)
    return [
        Document(page_content = page.extract_text() or "",
                 metadata = {"source": file_path, "page": i})

        for i, page in enumerate(reader.pages)
    
    ]

def split_docs(docs: list[Document], 
               chunk_size:int = 1000, 
               chunk_overlap:int = 200, 
               add_start_index:Optional[bool] = True) -> list[Document]:
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, 
                                                   chunk_overlap = chunk_overlap, 
                                                   add_start_index = add_start_index)
    

    return text_splitter.split_documents(docs)
    


def generate_embeddings():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
    return embeddings

#

def create_vector_store(embeddings, documents: list[Document]):
    vector_store = Chroma(collection_name = "example_collection1", embedding_function = embeddings, persist_directory = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\chroma_langchain_db")
    vector_store.add_documents(documents = documents)
    # result =  vector_store.similarity_search(query)
    return vector_store


if __name__ == "__main__":    
    file_path = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\data\\nke-10k-2023.pdf"
    docs = load_pdf(file_path)
    split_documents = split_docs(docs)
    print("="*50)
    print(len(docs))
    print("="*50)
    print(len(split_documents))
    print("="*50)
    print("End")
    embeddings = generate_embeddings()
    query = "How many distribution centres does Nike have in the US?"
    vector_store = create_vector_store(embeddings, split_documents)
    result = vector_store.similarity_search(query, k=1)
    print(result[0].page_content)





    
    # vector_store = InMemoryVectorStore.from_documents(split_documents, generate_embeddings())
    # vector_store = InMemoryVectorStore(generate_embeddings())
    # ids = vector_store.add_documents(documents = split_documents)
    # result = vector_store.similarity_search("How many distribution centres does Nike have in the US?")
    # print(result[0].page_content)

