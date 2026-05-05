"""Model loader module for initializing Mistral LLM and retrieval tools."""
# import os
# from transformers import (AutoTokenizer, AutoModelForCausalLM,
#                           BitsAndBytesConfig)
# from sentence_transformers import CrossEncoder
# from dotenv import load_dotenv


# def load_mistral_model():
#     load_dotenv()
#     hf_token = os.getenv("HF_TOKEN")
#     model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype="float16",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, quantization_config=bnb_config, device_map="auto",
#         token=hf_token
#     )
#     return model, tokenizer


# def setup_retrieval_tools(vectorstore):
#     retriever = vectorstore.as_retriever(
#         search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50,
#                                           "lambda_mult": 0.6}
#     )
#     rerank_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
#     return retriever, rerank_model
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv


def load_mistral_model():
    """Loading Mistral"""
    load_dotenv()
    # You still need your Hugging Face Token for the API
    hf_token = os.getenv("HF_TOKEN")

    # Mistral-7B-Instruct-v0.2 hosted on HF Inference Endpoints
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # This replaces the local 4-bit model loading
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        max_new_tokens=400,
        temperature=0.1,
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    # We return the llm object; tokenizer is handled by the API
    return llm


def setup_retrieval_tools(vectorstore):
    """Retrieval step"""
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.6}
    )
    # The reranker still typically runs locally for speed/low overhead
    rerank_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return retriever, rerank_model
