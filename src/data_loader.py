"""Data loader module for loading and processing IBM WatsonxDocsQA dataset."""
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_data(chunk_size=1000, chunk_overlap=300):
    """Load IBM WatsonxDocsQA dataset and split documents into chunks.
    
    Args:
        chunk_size: Number of characters per chunk (default: 1000).
        chunk_overlap: Number of overlapping characters between chunks (default: 300).
    
    Returns:
        List of Document objects split into chunks with metadata containing source IDs.
    """
    corpus_dataset = load_dataset("ibm-research/watsonxDocsQA", "corpus",
                                  split="train")

    documents = []
    for entry in corpus_dataset:
        content = f"Title: {entry['title']}\n\n{entry['document']}"
        doc = Document(page_content=content,
                       metadata={"source": entry["doc_id"]})
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
