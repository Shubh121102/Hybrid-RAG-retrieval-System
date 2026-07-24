from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from ingestion import load_pdf, split_docs
from retriever import generate_embeddings, create_vector_store, create_bm25_retriever, bm25_retriever_func, hybrid_retriever
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

    # Create BM25 index
    bm25_index = create_bm25_retriever(split_documents)
    
    # Define retriever functions
    def vector_retriever_func(query, top_k=10):
        return vector_store.similarity_search_with_score(query, k=top_k)
    
    def bm25_wrapper(query, top_k=10):
        return bm25_retriever_func(query, bm25_index, split_documents, top_k)
    
    # Create hybrid retriever function that returns documents (not just IDs)
    def hybrid_retriever_wrapper(query, top_k=10):
        # Get ranked results from hybrid_retriever
        ranked_results = hybrid_retriever(
            query=query,
            vector_retriever=vector_retriever_func,
            bm25_retriever=bm25_wrapper,
            rrf_constant=60,
            v_weights=0.5,
            b_weights=0.5,
            top_k=top_k
        )
        # Convert doc_id strings back to Document objects
        retrieved_docs = []
        for doc_id_str, score in ranked_results[:top_k]:
            doc_idx = int(doc_id_str)
            retrieved_docs.append(split_documents[doc_idx])
        return retrieved_docs

    # Use Hybrid Retriever in the RAG chain    
    retriever = hybrid_retriever_wrapper
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
    print("\n\nHYBRID RAG DEMO:\n")
    print(f"Q: {question}\n")
    print(f"A: {result}")


if __name__ == "__main__":
    rag_chain()