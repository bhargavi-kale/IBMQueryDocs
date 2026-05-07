import os
from huggingface_hub import InferenceClient
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

def load_mistral_model():
    """Returns the raw InferenceClient to allow direct chat_completion calls."""
    load_dotenv()
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=os.getenv("HF_TOKEN")
    )

def setup_retrieval_tools(vectorstore):
    """Retriever and reranker setup remain the same."""
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.6}
    )
    rerank_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return retriever, rerank_model