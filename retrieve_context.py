"""
06_retrieve_context.py
"""

import os
import importlib.util
import chromadb
from sentence_transformers import SentenceTransformer


def load_module(module_filename):
    """Load python files starting with numbers."""

    path = os.path.join(
        os.path.dirname(__file__),
        module_filename
    )

    spec = importlib.util.spec_from_file_location(
        module_filename[:-3],
        path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to Chroma database
client = chromadb.PersistentClient(path="chroma_db")


try:
    # Try to load existing collection
    collection = client.get_collection("rag_collection")


except:

    # Create Chroma database if it does not exist

    print("Creating Chroma database...")

    store_module = load_module(
        "05_create_chroma_store.py"
    )

    docs_module = load_module(
        "01_documents.py"
    )

    prep_module = load_module(
        "02_preprocessing.py"
    )

    chunk_module = load_module(
        "03_chunking.py"
    )

    vector_module = load_module(
        "04_vector_representation.py"
    )


    raw_documents = docs_module.load_documents()


    cleaned_documents = prep_module.preprocess_documents(
        raw_documents
    )


    chunks = chunk_module.build_chunks(
        cleaned_documents
    )


    embedded_chunks = vector_module.create_embeddings(
        chunks,
        model
    )


    collection = store_module.create_chroma_collection(
        embedded_chunks
    )

    print("Chroma database created!")


def retrieve_context(question, top_k=3):
    """
    Retrieve the most relevant chunks for a user's question.
    """


    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )


    documents = results["documents"][0]
    ids = results["ids"][0]


    context = "\n\n".join(documents)


    return context, ids



if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )


    context, sources = retrieve_context(question)


    print("\nRetrieved Context:\n")
    print(context)


    print("\nSources:")

    for source in sources:
        print("-", source)