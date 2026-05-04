# Design Document - IBMQueryDocs

## 1. Architecture Diagram
## Pre-processing:
        Documents (IBM WatsonxDocsQA)
             ↓
        Chunking (1000 chars, 300 overlap)
            ↓
        Embeddings (sentence-transformers)
            ↓
        Vector DB (ChromaDB)
## Retrieval & Generation:
        User Query
             ↓
   Query → Embedding Model
             ↓
     Vector Database (ChromaDB)
        (MMR: k=15, fetch_k=50)
             ↓
   Top 15 Chunks Retrieved
             ↓
      Reranking (BAAI/bge-reranker)
             ↓
   Top 3 Reranked Chunks
             ↓
      + User Query Combined
             ↓
        LLM (Mistral-7B-Instruct)
             ↓
      Final Answer Generated

## 2. Foundation Model Selection

Model: Mistral-7B-Instruct-v0.2
- Decoder-only transformer model
- Instruction-tuned for following user instructions
Why:
- Excellent for question answering with context (RAG)
- Strong performance on technical documentation
- Good balance of quality and efficiency

Strengths:
- Superior instruction following
- Effective context processing
- CPU-compatible inference

Limitations:
- Requires significant memory (~14GB)
- Slower than smaller models

## 3. Symbolic Components
- Chunking (1000 chars, 300 overlap)
- Similarity search with MMR (k=15, fetch_k=50)
- Reranking (top-3 with BAAI/bge-reranker-base)
- Citation tracking

Why:
- Helps retrieve correct data
- Reduces wrong answers
- Makes system reliable


## 4. Integration Strategy

- Documents → chunk → embed → store in vector DB
- User query → embedding → retrieve similar chunks
- Combine query + chunks → send to LLM
- LLM generates final answer


## 5. Evaluation Plan
- Using the predefined dataset :
  - Ask known questions
  - Compare answers with actual data


## 6. Risk Analysis
- Wrong retrieval
- Hallucination
- Biased data leads to biased answers
```

