"""Main entry point for IBMQueryDocs RAG application with Gradio interface."""
import gradio as gr
from src.data_loader import load_and_split_data
from src.vector_store import create_vector_store
from src.model_loader import load_mistral_model, setup_retrieval_tools
from src.rag_pipeline import run_rag_pipeline


def main():
    """Initialize and launch the RAG pipeline with Gradio chat interface."""
    # Steps 1-3 remain the same as your current main.py
    print("Step 1: Loading Data...")
    docs = load_and_split_data()
    print("Step 2: Building Vector Database...")
    vectorstore = create_vector_store(docs)
    print("Step 3: Loading Model & Retriever...")
    model = load_mistral_model()
    retriever, rerank_model = setup_retrieval_tools(vectorstore)

    # NEW: Define the function Gradio will call
    def chat_wrapper(message,chat_history):
        # We pass the loaded tools into your existing pipeline
        print(chat_history)
        # answer = run_rag_pipeline(message, retriever, rerank_model, model)
        # return answer
        answer, sources = run_rag_pipeline(message, retriever, rerank_model, model)
        print(sources)
        full_response = f"{answer}"
        return full_response

    # NEW: Launch the Gradio Interface
    print("Step 4: Launching Gradio Interface...")
    demo = gr.ChatInterface(fn=chat_wrapper, title="IBMQueryDocs")
    demo.launch(share=True)  # Generates your 72-hour public link


if __name__ == "__main__":
    main()
