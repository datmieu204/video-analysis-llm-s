# backend/app/core/build_store.py

import os
import shutil
from backend.app.core.embeddings import LawDataLoader, LawVectorStore
from backend.app.utils.splitter import text_splitter_documents

def build_and_persist_vectorstore():
    """
    Loads documents, splits them into chunks, and builds a persistent vector store.
    """
    persist_directory = "backend/app/vectorstore"

    if os.path.exists(persist_directory):
        print(f"Removing existing vector store at: {persist_directory}")
        shutil.rmtree(persist_directory)

    print("Loading law documents...")
    data_loader = LawDataLoader()
    documents = data_loader.load_documents()

    if not documents:
        print("No documents found. Aborting.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    chunks = text_splitter_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Building and persisting vector store...")
    vector_store = LawVectorStore(persist_directory=persist_directory)
    vector_store.build_vectorstore(documents=chunks, collection_name="law_data") 

    print("Vector store built and persisted successfully!")

if __name__ == "__main__":
    build_and_persist_vectorstore()