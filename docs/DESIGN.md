# Design Document - QueryDocs

## 1. Architecture Diagram
## Pre-processing:
        Documents
             ↓
        Chunking
            ↓ 
        Embeddings 
            ↓
        Vector DB (ChromaDB)
## Actual Retrieval:
        User Query
             ↓
   Query → Embedding Model
             ↓
     Vector Database (ChromaDB)
        (Similarity Search)
             ↓
   Top Relevant Chunks Retrieved
             ↓
      + User Query Combined
             ↓
        LLM (Flan-T5)
             ↓
      Final Answer Generated


## 2. Foundation Model Selection

Model: Flan-T5
- Encoder-Decoder model:
  - Encoder understands context
  - Decoder generates answer
Why:
- Good for question answering
- Works well with provided context (RAG)
- Faster and lightweight

Strengths:
- Encoder-Decoder model(More understanding on context)
- Runs on CPU

Limitations:
- Less powerful than larger models

## 3. Symbolic Components
- Chunking
- Similarity search (top-k)
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

