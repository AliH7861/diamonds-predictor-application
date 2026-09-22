"""Persistent Chroma stores for trusted knowledge and separate user preferences.

This file is the storage/retrieval layer of the Diamond Assistant's RAG system.

Main responsibilities:
1. Break knowledge documents into chunks.
2. Embed those chunks into vectors.
3. Store them in persistent Chroma collections.
4. Search those collections using semantic similarity.
5. Add a small lexical/keyword score to improve ranking.
6. Keep trusted knowledge, buyer profiles, and user memory separate.
"""

from hashlib import sha256  # Used to fingerprint knowledge so we know when documents changed.
from pathlib import Path  # Safer filesystem/path handling than raw strings.
import json  # Used to create a stable representation of segmentation/profile records.
import re  # Used for simple token extraction during lexical reranking.
from uuid import uuid4  # Generates unique IDs for user-memory entries.
# BASIC TEXT CHUNKIN====


def chunk_text(text: str, size: int = 450, overlap: int = 80) -> list[str]:
    """Split plain text into smaller overlapping chunks.

    Why chunk?
    Embedding/retrieval usually works better when we store smaller pieces of
    information instead of embedding an entire long document as one vector.

    `size` is an approximate character limit, not a token limit.
    `overlap` keeps some text from the previous chunk so context is not lost.
    """

    # Split on blank lines so we try to preserve paragraph meaning.
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]

    chunks: list[str] = []  # Final list of chunks.
    current = ""  # Text currently being accumulated into one chunk.

    for paragraph in paragraphs:
        # Try adding the next paragraph to the current chunk.
        candidate = f"{current}\n\n{paragraph}".strip()

        # If that would make the chunk too large, save the old chunk first.
        if current and len(candidate) > size:
            chunks.append(current)

            # Start the next chunk with a small tail from the previous chunk.
            # This overlap helps preserve context across chunk boundaries.
            current = (current[-overlap:] + "\n\n" + paragraph).strip()
        else:
            # Still under the size threshold, so keep accumulating.
            current = candidate

    # Save the last unfinished chunk.
    if current:
        chunks.append(current)

    return chunks


# MARKDOWN-AWARE CHUNKIN====


def chunk_markdown(text: str, size: int = 700) -> list[dict]:
    """Create heading-aware chunks while retaining document/section context.

    Example Markdown:

        # Diamond Clarity
        ## VS Clarity
        VS diamonds ...

    becomes a chunk whose stored text starts with:

        Diamond Clarity > VS Clarity

    This gives the retriever more context than storing the paragraph alone.
    """

    title = "Untitled"  # Current Markdown '# ' heading.
    section = "Overview"  # Current Markdown '## ' heading.

    # Each entry becomes:
    # (document title, section title, [paragraphs])
    sections: list[tuple[str, str, list[str]]] = []

    paragraphs: list[str] = []  # Paragraphs collected for the current section.

    def flush() -> None:
        """Save the currently collected paragraphs before switching sections."""
        if paragraphs:
            sections.append((title, section, list(paragraphs)))
            paragraphs.clear()

    # Break the Markdown into blocks separated by blank lines.
    for block in [item.strip() for item in text.split("\n\n") if item.strip()]:
        if block.startswith("# "):
            # New top-level document title.
            flush()
            title = block[2:].strip()
            section = "Overview"

        elif block.startswith("## "):
            # New section inside the current document.
            flush()
            section = block[3:].strip()

        else:
            # Normal content paragraph.
            paragraphs.append(block)

    # Save the last section.
    flush()

    chunks = []

    # Now turn each logical section into one or more size-limited chunks.
    for document_title, section_title, content in sections:
        # Prefix gives every chunk context about where it came from.
        prefix = f"{document_title} > {section_title}\n"

        current = ""

        for paragraph in content:
            candidate = f"{current}\n\n{paragraph}".strip()

            # If adding this paragraph would make the chunk too large,
            # save the current chunk and start a new one.
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

        # Save the last chunk for this section.
        if current:
            chunks.append(
                {
                    "title": document_title,
                    "section": section_title,
                    "document": prefix + current,
                }
            )

    return chunks


# SIMPLE KEYWORD/TOKEN EXTRACTIO====


def _terms(text: str) -> set[str]:
    """Extract useful lowercase terms for transparent lexical reranking.

    Vector similarity captures semantic meaning.
    This helper gives us a second signal: exact-ish word overlap.
    """

    # Common words that are usually not useful for search ranking.
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

    # Extract alphanumeric tokens, lowercase them, and remove stop words.
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in stop
    }


# CHROMA VECTOR DATABASE WRAPPE====


class ChromaStores:
    """Own separate Chroma collections for different kinds of information.

    Important design choice:
    - trusted diamond knowledge stays in one collection
    - authoritative customer profiles stay in another
    - user memory stays in another

    This prevents something a user says from becoming trusted domain knowledge.
    """

    def __init__(self, directory: str | Path, embedder):

        try:
            import chromadb  # Imported lazily so the error message is clearer.
        except ImportError as error:
            raise RuntimeError("Install requirements-assistant.txt to use Chroma.") from error

        # PersistentClient means vectors survive program restarts.
        self.client = chromadb.PersistentClient(path=str(directory))

        # Main trusted RAG knowledge collection.
        # Cosine distance is used for vector similarity.
        self.knowledge = self.client.get_or_create_collection(
            "diamond_knowledge_cosine_v1",
            metadata={"hnsw:space": "cosine"},
        )

        # Saved/authoritative buyer segmentation profiles.
        self.segmentation = self.client.get_or_create_collection(
            "diamond_customer_profiles_v1",
            metadata={"hnsw:space": "cosine"},
        )

        # User-specific statements/memory.
        # This is intentionally separate from trusted knowledge.
        self.memory = self.client.get_or_create_collection(
            "diamond_user_memory_cosine_v1",
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding client/model supplied by runtime.py.
        self.embedder = embedder

    def close(self) -> None:
        """Release Chroma index files.

        This especially matters on Windows because open database/index files
        may stay locked until the client is explicitly closed.
        """
        self.client.close()

    # INDEX TRUSTED KNOWLEDGE

    def index_knowledge(self, knowledge_dir: str | Path) -> int:
        """Read Markdown knowledge files, chunk them, embed them, and store them.

        Returns:
            Number of chunks indexed if the source changed.
            0 if the current indexed version is already up-to-date.
        """

        # Load every Markdown knowledge file in the configured directory.
        files = sorted(Path(knowledge_dir).glob("*.md"))

        # Create one stable combined string from all files.
        # This is used only to detect whether the source knowledge changed.
        source = "\n".join(path.name + "\n" + path.read_text(encoding="utf-8") for path in files)

        # Short SHA-256 fingerprint = version ID of the current knowledge base.
        fingerprint = sha256(source.encode("utf-8")).hexdigest()[:16]

        # Convert every Markdown file into heading-aware chunks.
        # Each chunk keeps:
        # - source filename
        # - title
        # - section
        # - actual chunk text
        chunks = [
            {"source": path.name, **chunk}
            for path in files
            for chunk in chunk_markdown(path.read_text(encoding="utf-8"))
        ]

        # Give every chunk a deterministic ID tied to this knowledge version.
        ids = [f"{fingerprint}-{number}" for number in range(len(chunks))]

        # Look at the versions currently stored in Chroma.
        stored = self.knowledge.get(include=["metadatas"])

        stored_versions = {item.get("version") for item in stored.get("metadatas", []) if item}

        # Only rebuild the knowledge index when the source files changed.
        if fingerprint not in stored_versions:
            # Remove the previous version so stale chunks do not stay searchable.
            if stored.get("ids"):
                self.knowledge.delete(ids=stored["ids"])

            # Embed and store all current knowledge chunks.
            self.knowledge.add(
                ids=ids,
                # Raw text Chroma returns during retrieval.
                documents=[item["document"] for item in chunks],
                # Metadata lets us audit exactly where a chunk came from.
                metadatas=[
                    {
                        "source": item["source"],
                        "title": item["title"],
                        "section": item["section"],
                        "version": fingerprint,
                    }
                    for item in chunks
                ],
                # Convert text chunks into vectors.
                embeddings=self.embedder.embed([item["document"] for item in chunks]),
            )

            # Tell the caller how many chunks were rebuilt.
            return len(chunks)

        # Nothing changed, so no re-indexing was necessary.
        return 0

    # SEARCH TRUSTED KNOWLEDGE

    def search_knowledge(
        self,
        queries: list[str],
        limit: int = 2,
    ) -> list[str]:
        """Convenience method that returns only retrieved text."""

        return [item["document"] for item in self.search_knowledge_details(queries, limit)]

    def search_knowledge_details(
        self,
        queries: list[str],
        limit: int = 2,
    ) -> list[dict]:
        """Search the RAG stores and return scored, auditable chunks.

        Ranking uses a hybrid score:

            75% semantic/vector similarity
            25% lexical word overlap

        This means a chunk can rank well because it is semantically related,
        while exact matching terms still give it an extra boost.
        """

        # Dictionary is keyed by document text so duplicate retrieved chunks
        # can be collapsed into a single best-scoring result.
        found: dict[str, dict] = {}

        if not queries:
            return []

        # Embed all user/search queries in one batch.
        for query, vector in zip(
            queries,
            self.embedder.embed(queries),
        ):
            # Search trusted knowledge if it contains anything.
            collections = [self.knowledge] if self.knowledge.count() else []

            # Buyer profiles are also searchable as a separate authoritative source.
            if self.segmentation.count():
                collections.append(self.segmentation)

            # Search each applicable Chroma collection.
            for collection in collections:
                result = collection.query(
                    query_embeddings=[vector],
                    # Never request more results than actually exist.
                    n_results=min(limit, collection.count()),
                    # We need text + metadata + distance for scoring/auditing.
                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],
                )

                # Chroma returns results nested by query.
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]

                # Score each retrieved chunk.
                for document, metadata, distance in zip(
                    documents,
                    metadatas,
                    distances,
                ):
                    # Convert cosine distance into a user-friendly similarity score.
                    # Clamp it between 0 and 1 for safety.
                    similarity = max(
                        0.0,
                        min(1.0, 1.0 - float(distance)),
                    )

                    # Extract meaningful words from the query and document.
                    query_terms = _terms(query)
                    document_terms = _terms(document)

                    # Fraction of query terms that also appear in the document.
                    lexical = (
                        len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
                    )

                    # Hybrid retrieval:
                    # semantic vectors matter most, but lexical overlap helps.
                    retrieval_score = 0.75 * similarity + 0.25 * lexical

                    # Build an auditable result object.
                    candidate = {
                        "query": query,
                        "source": (metadata or {}).get(
                            "source",
                            "unknown",
                        ),
                        "section": (metadata or {}).get(
                            "section",
                            "Overview",
                        ),
                        "similarity": round(similarity, 4),
                        "lexical_overlap": round(lexical, 4),
                        "retrieval_score": round(retrieval_score, 4),
                        "document": document,
                    }

                    # If this same chunk was found by multiple queries,
                    # keep only the version with the strongest retrieval score.
                    if (
                        document not in found
                        or candidate["retrieval_score"] > found[document]["retrieval_score"]
                    ):
                        found[document] = candidate

        # Highest-quality chunks are returned first.
        return sorted(
            found.values(),
            key=lambda item: item["retrieval_score"],
            reverse=True,
        )

    # INDEX AUTHORITATIVE BUYER / CLUSTER PROFILES

    def index_segmentation_records(self, records: list[dict]) -> int:
        """Index saved customer profiles separately from trusted documents.

        These records may describe things like cluster IDs and buyer profiles.
        They are authoritative project artifacts, not casual user memory.
        """

        if not records:
            return 0

        # Stable JSON representation so we can detect whether profiles changed.
        source = json.dumps(
            records,
            sort_keys=True,
            default=str,
        )

        # Version fingerprint for this profile dataset.
        fingerprint = sha256(source.encode("utf-8")).hexdigest()[:16]

        # Inspect currently stored profile metadata.
        stored = self.segmentation.get(include=["metadatas"])

        versions = {item.get("version") for item in stored.get("metadatas", []) if item}

        # If this exact profile version is already stored, do nothing.
        if fingerprint in versions:
            return 0

        # Remove old profiles before storing the current version.
        if stored.get("ids"):
            self.segmentation.delete(ids=stored["ids"])

        # Store profile text and embeddings.
        self.segmentation.add(
            ids=[str(record["id"]) for record in records],
            documents=[str(record["text"]) for record in records],
            metadatas=[
                {
                    "source": "profile_registry.json",
                    "section": str(record["profile_name"]),
                    "profile_name": str(record["profile_name"]),
                    "cluster_id": int(record["cluster_id"]),
                    "version": fingerprint,
                }
                for record in records
            ],
            embeddings=self.embedder.embed([str(record["text"]) for record in records]),
        )

        return len(records)

    # USER MEMORY
    def search_memory(
        self,
        question: str,
        limit: int = 4,
    ) -> list[str]:
        """Search user-memory vectors.

        This is separate from trusted RAG knowledge.
        """

        # Nothing stored yet.
        if not self.memory.count():
            return []

        # Embed the question and search for semantically similar memories.
        result = self.memory.query(
            query_embeddings=[self.embedder.embed([question])[0]],
            n_results=limit,
        )

        return result.get(
            "documents",
            [[]],
        )[0]

    def add_memory(self, text: str) -> None:
        """Embed and store one user-memory statement."""

        self.memory.add(
            # Unique ID so each stored statement has its own record.
            ids=[str(uuid4())],
            # Original user-memory text.
            documents=[text],
            # Vector representation used later for semantic search.
            embeddings=self.embedder.embed([text]),
        )
