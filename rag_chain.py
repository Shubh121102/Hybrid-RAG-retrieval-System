from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from ingestion import load_pdf, split_docs
from retriever import generate_embeddings, create_vector_store, create_bm25_retriever, hybrid_retrieve
from reranker import rerank

load_dotenv()


if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def rag_chain(question: str):

    # Defining the file path  
    file_path = "C:\\Users\\shubh\\OneDrive\\Desktop\\RAG\\data\\nke-10k-2023.pdf"

    # Loading the Documents
    docs = load_pdf(file_path)
    split_documents = split_docs(docs)

    # Generating vector embeddings 
    embeddings = generate_embeddings()

    # Creating the vector store
    vector_store = create_vector_store(embeddings, split_documents)

    # Retriever for vector and BM25
    vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    bm25_retriever = create_bm25_retriever(split_documents)


    # Hybrid retriever  
    # def hybrid_retrieve_fn(q):
    #     return hybrid_retrieve(query = q,retrievers=[vector_retriever,bm25_retriever],weights=[0.5,0.5])
    
    #Hybrid Retriever with Reranking
    def hybrid_retrieve_fn(q):
        docs =  hybrid_retrieve(query = q,retrievers=[vector_retriever,bm25_retriever],weights=[0.5,0.5])
        return rerank(q, docs, top_k=3)  # Rerank the retrieved documents
    
    hybrid_retriever = RunnableLambda(hybrid_retrieve_fn)

    # Defining the LLM
    llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B", # Qwen/Qwen3-0.6B
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
    llm = ChatHuggingFace(llm=llm_endpoint)

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(""" 
Answer the questions based on the context below.
                                              
Context: {context}
Question: {question}
Answer: 
                                            
Make sure to answer in a concise manner, if you don't know the answer, just say "I don't know"                                              

""")
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])


    # RAG chain 
    rag_chain = (
        {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm   
        | StrOutputParser()
    )

    # Final Output
    result = rag_chain.invoke(question)
    print("\n\nHYBRID RAG DEMO:\n")
    print(f"Q: {question}\n")
    print(f"A: {result}")
    return result


# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    question = "How many distribution centres in the US?"
    rag_chain(question)

