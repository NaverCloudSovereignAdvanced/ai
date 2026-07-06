import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagIndex:
    chunks: list[str]
    vectorizer: TfidfVectorizer
    matrix: object


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def build_index(text: str, output_path: Path) -> int:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("문서에서 추출된 텍스트가 없습니다.")

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    index = RagIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        pickle.dump(index, fp)
    return len(chunks)


def load_index(path: Path) -> RagIndex:
    with path.open("rb") as fp:
        return pickle.load(fp)


def retrieve(path: Path, query: str, top_k: int = 8) -> list[str]:
    index = load_index(path)
    query_vector = index.vectorizer.transform([query])
    scores = cosine_similarity(query_vector, index.matrix).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [index.chunks[i] for i in ranked if scores[i] > 0]
