"""
01_documents.py
----------------
Step 1: Read the PDF and load it as one document.
"""

import os
from pypdf import PdfReader

# Path to the source PDF
PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "AI.pdf"
)


def read_pdf_text(pdf_path: str) -> str:
    """
    Read all pages from the PDF and return one text string.
    """
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def load_documents(pdf_path: str = PDF_PATH):
    """
    Return ONE document containing the whole handbook.
    """

    full_text = read_pdf_text(pdf_path)

    documents = [
        {
            "document_id": 0,
            "title": "AI",
            "source_file": os.path.basename(pdf_path),
            "is_current": True,
            "text": full_text
        }
    ]

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print(f"Loaded {len(docs)} document.\n")

    for doc in docs:

        print(f"Document ID : {doc['document_id']}")
        print(f"Title       : {doc['title']}")
        print(f"Source File : {doc['source_file']}")
        print(f"Current     : {doc['is_current']}")
        print("\nPreview:\n")

        print(doc["text"][:800])