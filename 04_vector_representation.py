"""
04_embeddings.py
"""

import os
import importlib.util
import numpy as np
from sentence_transformers import SentenceTransformer


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


def create_embeddings(chunks, model):

    embedded_chunks = []

    for chunk in chunks:

        embedding = model.encode(
            chunk["search_text"],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        new_chunk = dict(chunk)

        new_chunk["embedding"] = embedding.tolist()

        embedded_chunks.append(new_chunk)

    return embedded_chunks


if __name__ == "__main__":

    # Load previous pipeline modules
    docs_module = load_module("01_documents.py")
    prep_module = load_module("02_preprocessing.py")
    chunk_module = load_module("03_chunking.py")

    # Pipeline
    raw_documents = docs_module.load_documents()

    cleaned_documents = prep_module.preprocess_documents(
        raw_documents
    )

    chunks = chunk_module.build_chunks(
        cleaned_documents
    )

    # Load embedding model
    print("\nLoading embedding model...\n")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Create embeddings
    embedded_chunks = create_embeddings(
        chunks,
        model
    )

    print(f"\nCreated embeddings for {len(embedded_chunks)} chunks.\n")

    first = embedded_chunks[0]

    print("=" * 80)

    print(f"Chunk ID : {first['chunk_id']}")

    print()

    print(first["chunk_text"][:300])

    print()

    print(f"Embedding Dimension : {len(first['embedding'])}")

    print()

    print("First 10 Values:")

    print(np.round(first["embedding"][:10], 4))

    print()

    print("Embedding Type:")

    print(type(first["embedding"]))