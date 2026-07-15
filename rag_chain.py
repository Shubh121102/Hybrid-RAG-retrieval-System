from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Optional 
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_chroma import Chroma
from ingestion import load_pdf, split_docs
from retriever import generate_embeddings, create_vector_store
# from langchain.chat_models import init_chat_model 

load_dotenv()


if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")



def rag_chain():
    file_path = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\data\\nke-10k-2023.pdf"
    docs = load_pdf(file_path)
    split_documents = split_docs(docs)
    embeddings = generate_embeddings()
    vector_store = create_vector_store(embeddings, split_documents)
    retriever = vector_store.as_retriever(search_type="similarity", k=1)
    llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepreinforce-ai/Ornith-1.0-9B", # Qwen/Qwen3-0.6B
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
    llm = ChatHuggingFace(llm=llm_endpoint)
    prompt = ChatPromptTemplate.from_template(""" 
Answer the questions based on the context below.
                                              
Context: {context}
Question: {question}
Answer: 
                                            
Make sure to answer in a concise manner, if you don't know the answer, just say "I don't know"                                              

""")
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm   
        | StrOutputParser()
    )

    question = "How many distribution centres does Nike have in the US?"
    result = rag_chain.invoke(question)
    print("\n\nBASIC RAG DEMO:\n")
    print(f"Q: {question}\n")
    print(f"A: {result}")
    # result = retriever.invoke(query)
    # return result[0].page_content

if __name__ == "__main__":
    rag_chain()