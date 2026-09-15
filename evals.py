from datasets import Dataset
from ragas import evaluate
from sentence_transformers import SentenceTransformer
# from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.embeddings import GoogleEmbeddings, HuggingFaceEmbeddings
from ragas.llms import llm_factory
from rag_chain import get_answer_and_contexts
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

class CustomEmbeddings:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str):
        """Embed a single query"""
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_documents(self, texts: list):
        """Embed multiple documents"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

client = genai.Client(api_key = os.environ.get("GOOGLE_API_KEY"))

questions = [
    "How many distribution centres in the US?"
]

ground_truths = [
    "There are 8 distribution centres in the US."
]

data = {
    "question":[],
    "answer":[],
    "contexts":[],
    "ground_truth":[]
}

for q, gt in zip(questions, ground_truths):
    result = get_answer_and_contexts(q)

    data["question"].append(q)
    data["answer"].append(result["answer"])
    data["contexts"].append(result["contexts"])
    data["ground_truth"].append(gt)


dataset = Dataset.from_dict(data)

llm = llm_factory(
    "gemini-3.5-flash-lite",
    provider = "google",
    client = client
)

# embeddings = GoogleEmbeddings(client = client, model = "gemini-embedding-001")
embeddings = CustomEmbeddings(model_name="BAAI/bge-small-en-v1.5")

metrics = [
    Faithfulness(llm = llm),
    AnswerRelevancy(llm = llm, embeddings = embeddings),
    ContextPrecision(llm = llm),
    ContextRecall(llm = llm)
]

results = evaluate(
    dataset = dataset,
    metrics = metrics,
    llm = llm,
    embeddings = embeddings
)

print("\n==========RAGAS Evaluation Results==========")
print(results)