import importlib.util
import os


def load_module(module_filename):
    """Load a Python module whose filename starts with a digit."""
    path = os.path.join(os.path.dirname(__file__), module_filename)
    spec = importlib.util.spec_from_file_location(module_filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_text(text, chunk_size=100, overlap=30):
    """
    Split text into overlapping chunks based on words.
    """

    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def build_chunks(documents, chunk_size=120, overlap=30):
    """
    Convert documents into chunk dictionaries.
    """

    chunk_rows = []

    for doc in documents:

        chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk in enumerate(chunks):

            chunk_rows.append({

                "chunk_id": f"chunk_{chunk_index}",

                "document_id": doc["document_id"],

                "title": doc["title"],

                "source_file": doc["source_file"],

                "chunk_index": chunk_index,

                "chunk_text": chunk,

                # النص الذى سيتم عمل Embedding له
                "search_text": chunk

            })

    return chunk_rows


if __name__ == "__main__":

    docs_module = load_module("01_documents.py")
    prep_module = load_module("02_preprocessing.py")

    raw_documents = docs_module.load_documents()

    cleaned_documents = prep_module.preprocess_documents(raw_documents)

    chunks = build_chunks(cleaned_documents)

    print(f"\nCreated {len(chunks)} chunks.\n")

    for chunk in chunks[:5]:

        print("=" * 80)

        print(f"Chunk ID     : {chunk['chunk_id']}")

        print(f"Document ID  : {chunk['document_id']}")

        print(f"Chunk Number : {chunk['chunk_index']}")

        print()

        print(chunk["chunk_text"])