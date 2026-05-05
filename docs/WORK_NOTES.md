# Retrieval-Augmented Generation (RAG) — Workflow

## 1. Data Loading
- Load documents from datasets or external sources (PDFs, text files, Markdown, web pages).

## 2. Document Chunking
- Split documents into smaller chunks for efficient processing.
- Maintain overlap between chunks to preserve context and avoid loss of meaning.

## 3. Embedding Generation
- Convert each document chunk into vector embeddings (numerical representations) that capture semantic meaning.

## 4. Vector Storage (ChromaDB)
- Store the generated embeddings in a vector database (e.g., ChromaDB) for efficient similarity search and retrieval.

## 5. User Query Processing
- Convert the user's query into an embedding using the same embedding model.

## 6. Similarity Search (Retrieval Step)
- Perform vector similarity search to retrieve the most relevant document chunks from the vector DB.

## 7. Context Passing to LLM
- Pass the retrieved chunks as context to Large Language Model (LLM).

## 8. Answer Generation
- The LLM generates a response based on the context and the query.

## 9. Citation Tracking
- Track which document chunks were used and provide citations/references in the generated answer.


# Dataset - ibm-research/watsonxDocsQA
## What is Split?
- A split is just a subset of the dataset, like train, test, or validation, used to organize data for training, testing, or evaluation.
- My dataset: In dataset, the subset corpus has only one split(train), while the question_answers subset has two splits(train,test).


# Vector Database and Vector Embeddings - https://www.ibm.com/think/topics/vector-database
- Vector embeddings are basically converting text (or any data) into numbers so that machines can understand the meaning, not just the words.
- Each piece of text (like a chunk) gets converted into a vector, which is just a list of numbers representing its meaning.
- Similar meanings will have similar vectors. For example, “car” and “vehicle” will be close to each other even if the words are different.
- These vectors are stored in a vector database.
- The vector DB doesn’t search using keywords like normal databases. Instead, it compares vectors and finds the closest ones.
- So when a query comes in, it is also converted into a vector, and then the DB finds the most similar vectors (i.e., relevant data).
- This is what makes semantic search possible and helps in getting more accurate results.


# Chunking - https://docs.langchain.com/oss/python/integrations/splitters
- Initially, I thought chunking is only required because we cannot pass the entire document to the LLM.
- I did not realize that chunking directly affects the accuracy and performance of the system.
- I experimented with different chunk sizes (600 → 1000 tokens).
- Increasing the chunk size improved answer quality significantly.
- Smaller chunks led to loss of context.
- This helped me understand that chunking is not just a preprocessing step, but a key factor in retrieval quality.


# Retrieval
## Similarity search:
- Retrieves chunks purely based on similarity
## MMR:
- Introduces diversity in results
- Avoids redundant chunks
- Balances relevance + diversity
## Reranking:
- Reorders retrieved chunks based on deeper query–document understanding
- Improves final relevance of top results
## Key improvement:
- By combining MMR + reranking + better chunking,retrieval accuracy improved from ~80% to ~90%.


# Model(Flan-T5 vs Mistral-7B)
- I was initially confused whether poor results were due to retrieval or the LLM.
## To test this, I:
- Passed the entire document directly to Flan-T5 (no chunking/retrieval)
- Asked questions with full context
- Flan-T5 still returned very short answers (3–4 words)
## Then switched to Mistral-7B:
- Generated complete and meaningful answers
- Model size (7B parameters) actually made a visible difference in output quality.


# Prompt Engineering
- Initially did not focus much on prompt design.
- While testing edge cases (e.g., irrelevant questions like “who is bhargavi?”), I noticed issues.
## Then:
- Refined prompts to handle irrelevant queries
- Reduced hallucinations
- Tried multiple prompt variations
- Observed clear differences in output.

# Model Deployment
- Initially ran Mistral locally using Colab with 4-bit quantization.
- Faced issues like GPU memory limitations, Session expiration and took longer time to load model.
## Modularizing into Python files:
- Could not use notebook-style execution (select/run kernel)
- Later switched to Hugging Face Inference API
    - No local GPU required
    - Easier integration


# Challenges Faced - RAGAS - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Faced memory limitations while running RAGAS locally (especially with larger context and multiple samples).
- API-related issues (key expiration, timeout errors) made it unreliable to use consistently.
- Because of these issues, I could not complete evaluation using RAGAS directly and had to switch to an alternative approach (Gemini-based evaluation).


# What Strengthened My System
- MMR (better retrieval diversity)
- Cross-encoder reranking
- Prompt optimization
- Evaluation strategy
These changes improved both retrieval accuracy and answer quality.
