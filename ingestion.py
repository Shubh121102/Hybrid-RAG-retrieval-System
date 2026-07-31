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




