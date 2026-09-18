"""Persistent Chroma stores for trusted knowledge and separate user preferences."""

from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4


def chunk_text(text: str, size: int = 450, overlap: int = 80) -> list[str]:
    """Split text on paragraph boundaries while retaining small semantic overlaps."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > size:
            chunks.append(current)
            current = (current[-overlap:] + "\n\n" + paragraph).strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(text: str, size: int = 700) -> list[dict]:
    """Create heading-aware chunks that retain their document and section context."""
    title = "Untitled"
    section = "Overview"
    sections: list[tuple[str, str, list[str]]] = []
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append((title, section, list(paragraphs)))
            paragraphs.clear()

    for block in [item.strip() for item in text.split("\n\n") if item.strip()]:
        if block.startswith("# "):
            flush()
            title = block[2:].strip()
            section = "Overview"
        elif block.startswith("## "):
            flush()
            section = block[3:].strip()
        else:
            paragraphs.append(block)
    flush()

    chunks = []
    for document_title, section_title, content in sections:
        prefix = f"{document_title} > {section_title}\n"
        current = ""
        for paragraph in content:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(prefix) + len(candidate) > size:
                chunks.append(
                    {
                        "title": document_title,
                        "section": section_title,
                        "document": prefix + current,
                    }
                )
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(
                {
                    "title": document_title,
                    "section": section_title,
                    "document": prefix + current,
                }
            )
    return chunks


def _terms(text: str) -> set[str]:
    """Return useful lowercase terms for transparent lexical reranking."""
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "which",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in stop
    }


class ChromaStores:
    """Own two collections so user statements cannot contaminate trusted knowledge."""

    def __init__(self, directory: str | Path, embedder):
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("Install requirements-assistant.txt to use Chroma.") from error
        self.client = chromadb.PersistentClient(path=str(directory))
        self.knowledge = self.client.get_or_create_collection(
            "diamond_knowledge_cosine_v1", metadata={"hnsw:space": "cosine"}
        )
        self.memory = self.client.get_or_create_collection(
            "diamond_user_memory_cosine_v1", metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder

    def close(self) -> None:
        """Release Chroma index files, which Windows keeps locked until closed."""
        self.client.close()

    def index_knowledge(self, knowledge_dir: str | Path) -> int:
        files = sorted(Path(knowledge_dir).glob("*.md"))
        source = "\n".join(path.name + "\n" + path.read_text(encoding="utf-8") for path in files)
        fingerprint = sha256(source.encode("utf-8")).hexdigest()[:16]
        chunks = [
            {"source": path.name, **chunk}
            for path in files
            for chunk in chunk_markdown(path.read_text(encoding="utf-8"))
        ]
        ids = [f"{fingerprint}-{number}" for number in range(len(chunks))]
        stored = self.knowledge.get(include=["metadatas"])
        stored_versions = {item.get("version") for item in stored.get("metadatas", []) if item}
        if fingerprint not in stored_versions:
            if stored.get("ids"):
                self.knowledge.delete(ids=stored["ids"])
            self.knowledge.add(
                ids=ids,
                documents=[item["document"] for item in chunks],
                metadatas=[
                    {
                        "source": item["source"],
                        "title": item["title"],
                        "section": item["section"],
                        "version": fingerprint,
                    }
                    for item in chunks
                ],
                embeddings=self.embedder.embed([item["document"] for item in chunks]),
            )
            return len(chunks)
        return 0

    def search_knowledge(self, queries: list[str], limit: int = 2) -> list[str]:
        return [item["document"] for item in self.search_knowledge_details(queries, limit)]

    def search_knowledge_details(self, queries: list[str], limit: int = 2) -> list[dict]:
        """Return deduplicated chunks with their query, source, and cosine similarity."""
        found: dict[str, dict] = {}
        if not queries:
            return []
        for query, vector in zip(queries, self.embedder.embed(queries)):
            result = self.knowledge.query(
                query_embeddings=[vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for document, metadata, distance in zip(documents, metadatas, distances):
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))
                query_terms = _terms(query)
                document_terms = _terms(document)
                lexical = (
                    len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
                )
                retrieval_score = 0.75 * similarity + 0.25 * lexical
                candidate = {
                    "query": query,
                    "source": (metadata or {}).get("source", "unknown"),
                    "section": (metadata or {}).get("section", "Overview"),
                    "similarity": round(similarity, 4),
                    "lexical_overlap": round(lexical, 4),
                    "retrieval_score": round(retrieval_score, 4),
                    "document": document,
                }
                if (
                    document not in found
                    or candidate["retrieval_score"] > found[document]["retrieval_score"]
                ):
                    found[document] = candidate
        return sorted(found.values(), key=lambda item: item["retrieval_score"], reverse=True)

    def search_memory(self, question: str, limit: int = 4) -> list[str]:
        if not self.memory.count():
            return []
        result = self.memory.query(
            query_embeddings=[self.embedder.embed([question])[0]], n_results=limit
        )
        return result.get("documents", [[]])[0]

    def add_memory(self, text: str) -> None:
        self.memory.add(
            ids=[str(uuid4())], documents=[text], embeddings=self.embedder.embed([text])
        )
