# backend/vectorstore.py

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def create_vectorstore(embedder, chunks, persist_dir="chroma_store"):
    # Check if chunks are valid
    if not chunks or len(chunks) == 0:
        raise ValueError("No chunks provided for vector store creation.")

    # Handle different input types
    docs = []
    for item in chunks:
        if isinstance(item, Document):
            # Already a Document object, use as-is
            docs.append(item)
        elif isinstance(item, tuple):
            # (text, metadata) tuple
            text, metadata = item
            docs.append(Document(page_content=str(text), metadata=metadata))
        else:
            # Plain text
            docs.append(Document(page_content=str(item), metadata={}))

    # Build and return the vectorstore
    return Chroma.from_documents(docs, embedding=embedder, persist_directory=persist_dir)

def query_vectorstore(vectordb, query, k=5):
    return vectordb.similarity_search_with_score(query, k=k)