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
