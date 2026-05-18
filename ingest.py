from __future__ import annotations

import json
import pickle
from pathlib import Path
import re

from pypdf import PdfReader
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = VECTORSTORE_DIR / "vectorizer.pkl"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_STRIDE = CHUNK_SIZE - CHUNK_OVERLAP


def remove_existing_vectorstore() -> None:
    if not VECTORSTORE_DIR.exists():
        return
    for item in VECTORSTORE_DIR.iterdir():
        if item.is_file():
            item.unlink()
        else:
            for nested in sorted(item.rglob("*"), reverse=True):
                if nested.is_file():
                    nested.unlink()
                elif nested.is_dir():
                    nested.rmdir()
            if item.exists():
                item.rmdir()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def load_pdf_documents() -> list[dict]:
    documents: list[dict] = []
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCUMENTS_DIR}")

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            documents.append(
                {
                    "text": cleaned,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                }
            )

    if not documents:
        raise ValueError("Text extraction completed, but no usable content was found in the PDFs.")

    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        words = document["text"].split()
        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                metadata = dict(document["metadata"])
                metadata["chunk_index"] = chunk_index
                chunks.append({"text": chunk_text, "metadata": metadata})
                chunk_index += 1

            if end >= len(words):
                break
            start += CHUNK_STRIDE

    return chunks


def build_vectorstore(chunks: list[dict]) -> None:
    remove_existing_vectorstore()
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)

    save_npz(INDEX_PATH, matrix)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [{"text": text, "metadata": metadata} for text, metadata in zip(texts, metadatas)],
            file,
            ensure_ascii=True,
            indent=2,
        )


def main() -> None:
    documents = load_pdf_documents()
    chunks = split_documents(documents)
    build_vectorstore(chunks)
    print(
        f"Ingestion complete. Stored {len(chunks)} chunks from {len(documents)} pages in {VECTORSTORE_DIR}."
    )


if __name__ == "__main__":
    main()
