"""
06_retrieve_context.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma database
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("rag_collection")


def retrieve_context(question, top_k=3):
    """
    Retrieve the most relevant chunks for a user's question.
    """

    # Convert question to embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    context = "\n\n".join(documents)

    return context, ids


if __name__ == "__main__":

    question = input("Enter your question: ")

    context, sources = retrieve_context(question)

    print("\nRetrieved Context:\n")
    print(context)

    print("\nSources:")
    for source in sources:
        print("-", source)