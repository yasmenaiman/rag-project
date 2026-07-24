"""
02_preprocessing.py
"""

import re
import importlib.util
import os


def load_module(module_filename):
    path = os.path.join(os.path.dirname(__file__), module_filename)
    spec = importlib.util.spec_from_file_location(module_filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    """

    # Remove carriage returns
    text = text.replace("\r", " ")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove bullet symbols extracted from PDF
    text = re.sub(r"[•●◦▪■□►]", " ", text)

    # Remove long underscores
    text = re.sub(r"_{3,}", " ", text)

    # Replace multiple new lines with one space
    text = re.sub(r"\n+", " ", text)

    # Remove extra spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_documents(documents):
    cleaned_documents = []

    for doc in documents:
        cleaned_doc = dict(doc)
        cleaned_doc["text"] = clean_text(doc["text"])
        cleaned_documents.append(cleaned_doc)

    return cleaned_documents


if __name__ == "__main__":

    docs_module = load_module("01_documents.py")

    raw_documents = docs_module.load_documents()

    cleaned_documents = preprocess_documents(raw_documents)

    print(f"Preprocessed {len(cleaned_documents)} document(s).\n")

    for doc in cleaned_documents:
        print("=" * 70)
        print(f"Document ID : {doc['document_id']}")
        print(f"Title       : {doc['title']}")
        print(f"Source      : {doc['source_file']}\n")
        print(doc["text"][:800])