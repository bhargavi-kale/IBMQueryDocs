---
title: QueryDocs-IBM
app_file: main.py
sdk: gradio
sdk_version: 6.14.0
---
# RAG Pipeline Project

## Authorship
- **Author**: Bhargavi Kale
- **Date**: April 28, 2026
- **GitHub**: https://github.com/2026-spring-comp-741-841/project-bhargavi.git

## Brief Summary
This project implements a Retrieval-Augmented Generation (RAG) pipeline for question answering along with citation. It loads documents from the IBM WatsonxDocsQA dataset, processes them into chunks, generates embeddings, stores them in a ChromaDB vector database, and uses the Mistral-7B-Instruct-v0.2 model to generate answers based on relevant retrieved context. The system includes reranking for improved retrieval accuracy.

## Folder/File Structure
```
project/
├── rag_end_to_end.ipynb          # End-to-end RAG notebook
├── README.md                     # This file
├── .env                          # Environment variables (HF_TOKEN)
├── .gitignore                    # Git ignore file
├── docs/                         # Documentation folder
│   ├── DESIGN.md                 # Design document 
│   ├── sprints.md                # Sprint notes
│   └── WORK_NOTES.md             # Workflow and implementation notes
└── src/                          # Source code
    ├── data_loader.py            # Loads and chunks documents from dataset
    ├── model_loader.py           # Loads Mistral model and sets up retriever/reranker
    ├── rag_pipeline.py           # Implements the RAG question-answering logic
    └── vector_store.py           # Creates and manages ChromaDB vector store
├── main.py                       # Main script orchestrating the pipeline
├── requirements.txt              # Python dependencies
```

## Installation Instructions

### Prerequisites
- Python 3.11 or higher
- Git (for cloning if applicable)
- Hugging Face account with access token (for downloading models)

### Setup Steps
1. **Clone or download the project** to your local machine.

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env` file and update the values:
     - `HF_TOKEN`: Your Hugging Face API token (required for downloading Mistral model)

## Description of Datasets
The project uses the `ibm-research/watsonxDocsQA` dataset from Hugging Face:
- **Corpus subset**: Contains training documents with titles and content
- **Question-answers subset**: Contains Q&A pairs for evaluation (not used in main pipeline)
- Documents are split into chunks of 1000 characters with 300-character overlap
- Total chunks: Approximately 7212 after processing

## Configuration Information
- **.env file**: Contains API keys and tokens
  - `HF_TOKEN`: Required for accessing Hugging Face models
- **Model configuration**: Mistral-7B-Instruct-v0.2 loaded on Colab GPU
- **Embedding model**: sentence-transformers/all-mpnet-base-v2
- **Reranker**: BAAI/bge-reranker-base (runs on CPU)
- **Vector store**: ChromaDB with MMR retrieval (k=15, fetch_k=50, lambda_mult=0.6)

## Deployment Info
For now it is not deployed, due to less Hugging face spaces

## Instructions on How to Run It and What to Expect

### Option 1: Run with Gradio Web Interface
1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Navigate to project root** and run:
   ```bash
   python main.py
   ```

3. **Expected output**:
   - Step 1: Loading Data... (loads and chunks ~7212 document chunks)
   - Step 2: Building Vector Database... (embeds chunks in batches, may take several minutes)
   - Step 3: Loading Model & Retriever... (downloads and loads Mistral-7B model, ~15GB)
   - Step 4: Launching Gradio Interface... (opens a web-based chat interface)
   - Gradio ChatInterface opens in your browser at `http://localhost:7860` with a public shareable link generated for 72 hours
   - Use the chat interface to ask questions about IBM documentation; the RAG pipeline retrieves relevant context and generates answers with citations

### Option 2: Run with Jupyter Notebook
1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Choose one of the notebooks**:
   - **`rag_end_to_end.ipynb`**: Comprehensive end-to-end RAG notebook with detailed implementation and exploration

4. **Select the Kernel**:
   - Click on the kernel selector (top-right of the notebook)
   - Select **"Colab"** kernel and connect to **Colab Server**
   - This ensures the notebook runs with the proper computational resources

5. **Run the notebooks**: Execute cells sequentially from top to bottom (use Shift+Enter or the Run button)

6. **Expected output**: Same as Option 1 - data loading, vector store creation, model loading, and interface launch
   - Notebooks provide additional visibility into intermediate steps and results


## License
Copyright (c) 2026 Bhargavi Kale
