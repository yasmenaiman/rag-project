"""
05_create_chroma_store.py
"""

import os
import importlib.util
import chromadb


def load_module(module_filename):
    """Load a Python file whose filename starts with a digit."""
    path = os.path.join(os.path.dirname(__file__), module_filename)

    spec = importlib.util.spec_from_file_location(
        module_filename[:-3],
        path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def create_chroma_collection(embedded_chunks):

    # Create persistent database
    client = chromadb.PersistentClient(path="chroma_db")

    # Delete old collection if it already exists
    try:
        client.delete_collection("rag_collection")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name="rag_collection"
    )

    # Add every chunk
    for chunk in embedded_chunks:

        collection.add(

            ids=[chunk["chunk_id"]],

            embeddings=[chunk["embedding"]],

            documents=[chunk["chunk_text"]],

            metadatas=[

                {
                    "document_id": str(chunk["document_id"]),
                    "title": chunk["title"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"]
                }

            ]
        )

    return collection


if __name__ == "__main__":

    docs_module = load_module("01_documents.py")
    prep_module = load_module("02_preprocessing.py")
    chunk_module = load_module("03_chunking.py")
    vector_module = load_module("04_vector_representation.py")

    # Pipeline

    raw_documents = docs_module.load_documents()

    cleaned_documents = prep_module.preprocess_documents(
        raw_documents
    )

    chunks = chunk_module.build_chunks(
        cleaned_documents
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embedded_chunks = vector_module.create_embeddings(
        chunks,
        model
    )

    collection = create_chroma_collection(
        embedded_chunks
    )

    print()

    print("=" * 70)

    print("Chroma database created successfully!")

    print()

    print(f"Number of stored chunks : {collection.count()}")

    print()

    print("Database folder : chroma_db")

    print("=" * 70)