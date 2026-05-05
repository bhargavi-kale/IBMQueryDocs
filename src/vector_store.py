"""Vector store module for creating and managing document embeddings."""
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_store(docs,
                        model_name="sentence-transformers/all-mpnet-base-v2"):
    """Create a vector store from documents using HuggingFace embeddings.
    
    Args:
        docs: List of documents to embed.
        model_name: HuggingFace embedding model identifier.
    
    Returns:
        Chroma vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    batch_size = 100
    vectorstore = None

    print(f"Starting embedding for {len(docs)} chunks...")  # Notebook logic

    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(documents=batch,
                                                embedding=embeddings)
        else:
            vectorstore.add_documents(batch)
        # Tracking progress is essential for 7k+ chunks
        print(f"Completed {i + len(batch)} / {len(docs)} chunks...")

    print("Vector store ready.")
    return vectorstore
