"""MemoryStore - persistent knowledge base backed by memvid (.mv2 files).

Wraps the memvid-sdk to provide a simple interface for:
- Querying prior research findings
- Ingesting new research documents after each pipeline run
- Tracking statistics about the knowledge base

Usage::

    from recon.memory.store import MemoryStore

    with MemoryStore("./memory/recon.mv2") as store:
        # Query for relevant prior knowledge
        prior = store.query("AI agent frameworks", k=5)

        # After a pipeline run, ingest new research
        count = store.ingest_research("./research", topic="AI agents", run_id="run-001")

        # Check stats
        print(store.stats())
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_memvid() -> tuple[Any, Any]:
    """Import memvid_sdk lazily and return (create, use) functions.

    Raises:
        ImportError: If memvid-sdk is not installed.
    """
    try:
        from memvid_sdk import create, use

        return create, use
    except ImportError:
        msg = (
            "memvid-sdk is required for cross-run memory. "
            "Install it with: pip install recon-ai[memory]"
        )
        raise ImportError(msg) from None


class MemoryStore:
    """Persistent knowledge base backed by a memvid .mv2 file.

    Supports both creation of new memory files and opening existing ones.
    Use as a context manager for automatic resource cleanup.

    Args:
        path: Path to the .mv2 memory file. Created if it doesn't exist.
        embedding_provider: Embedding provider for indexing (``local`` or ``openai``).
    """

    def __init__(
        self,
        path: str = "./memory/recon.mv2",
        embedding_provider: str = "local",
    ) -> None:
        self._path = Path(path)
        self._embedding_provider = embedding_provider
        self._mv: Any = None
        self._opened = False

    def _ensure_open(self) -> Any:
        """Open or create the memory file if not already open."""
        if self._mv is not None:
            return self._mv

        create_fn, use_fn = _get_memvid()

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            self._mv = use_fn("basic", str(self._path), mode="open")
            logger.debug("Opened existing memory: %s", self._path)
        else:
            self._mv = create_fn(str(self._path))
            logger.info("Created new memory file: %s", self._path)

        self._opened = True
        return self._mv

    def query(
        self,
        topic: str,
        questions: list[str] | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for relevant prior findings in the knowledge base.

        Args:
            topic: Main research topic to search for.
            questions: Optional specific questions to refine the search.
            k: Maximum number of results to return.

        Returns:
            List of dicts with keys: title, text, metadata, score.
            Empty list if memory is empty or no relevant results found.
        """
        mv = self._ensure_open()

        # Build a search query from topic + questions
        query_parts = [topic]
        if questions:
            query_parts.extend(questions[:3])  # Limit to avoid overly long queries
        search_query = " | ".join(query_parts)

        try:
            results = mv.find(search_query, k=k, mode="auto")
        except Exception:
            logger.warning("Memory query failed, returning empty results", exc_info=True)
            return []

        # Normalize results into a consistent format
        hits: list[dict[str, Any]] = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    hits.append(
                        {
                            "title": r.get("title", ""),
                            "text": r.get("text", r.get("snippet", "")),
                            "metadata": r.get("metadata", {}),
                            "score": r.get("score", 0.0),
                        }
                    )
                else:
                    # Some versions return tuples or strings
                    hits.append(
                        {
                            "title": "",
                            "text": str(r),
                            "metadata": {},
                            "score": 0.0,
                        }
                    )

        logger.debug("Memory query returned %d results for '%s'", len(hits), topic)
        return hits

    def ingest_research(
        self,
        research_dir: str,
        topic: str,
        run_id: str,
    ) -> int:
        """Ingest all research markdown files from a directory.

        Args:
            research_dir: Path to directory containing research ``*.md`` files.
            topic: Research topic for metadata tagging.
            run_id: Unique identifier for this pipeline run.

        Returns:
            Number of documents ingested.
        """
        mv = self._ensure_open()
        research_path = Path(research_dir)
        md_files = sorted(research_path.glob("*.md"))

        if not md_files:
            logger.debug("No research files found in %s", research_dir)
            return 0

        timestamp = datetime.now(tz=UTC).isoformat()
        documents: list[dict[str, Any]] = []

        for f in md_files:
            content = f.read_text()
            if not content.strip():
                continue

            documents.append(
                {
                    "title": f"Research: {f.stem} ({topic})",
                    "label": "research",
                    "text": content,
                    "metadata": {
                        "source_file": f.name,
                        "topic": topic,
                        "run_id": run_id,
                        "phase": "investigation",
                        "ingested_at": timestamp,
                    },
                    "tags": ["research", topic.lower().replace(" ", "-")],
                }
            )

        if not documents:
            return 0

        try:
            mv.put_many(documents)
            logger.info(
                "Ingested %d research documents into memory for topic '%s'",
                len(documents),
                topic,
            )
        except Exception:
            logger.warning("Failed to ingest research into memory", exc_info=True)
            return 0

        return len(documents)

    def ingest_report(
        self,
        report_path: str,
        topic: str,
        run_id: str,
        phase: str,
    ) -> None:
        """Ingest a single report file into memory.

        Args:
            report_path: Path to the report markdown file.
            topic: Research topic for metadata tagging.
            run_id: Unique identifier for this pipeline run.
            phase: Pipeline phase (``verification`` or ``synthesis``).
        """
        mv = self._ensure_open()
        path = Path(report_path)

        if not path.exists():
            logger.debug("Report not found, skipping ingest: %s", report_path)
            return

        content = path.read_text()
        if not content.strip():
            return

        timestamp = datetime.now(tz=UTC).isoformat()

        try:
            mv.put(
                title=f"{phase.title()} Report: {topic}",
                label=phase,
                text=content,
                metadata={
                    "source_file": path.name,
                    "topic": topic,
                    "run_id": run_id,
                    "phase": phase,
                    "ingested_at": timestamp,
                },
                tags=[phase, topic.lower().replace(" ", "-")],
            )
            logger.info("Ingested %s report into memory", phase)
        except Exception:
            logger.warning("Failed to ingest %s report into memory", phase, exc_info=True)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the memory file.

        Returns:
            Dict with keys like frame_count, size_bytes.
            Empty dict if memory is not open or stats fail.
        """
        try:
            mv = self._ensure_open()
            result = mv.stats()
            if isinstance(result, dict):
                return result
            return {"raw": str(result)}
        except Exception:
            logger.warning("Failed to read memory stats", exc_info=True)
            return {}

    def close(self) -> None:
        """Flush and close the memory file."""
        if self._mv is not None:
            try:
                self._mv.seal()
                logger.debug("Memory file sealed: %s", self._path)
            except Exception:
                logger.warning("Failed to seal memory file", exc_info=True)
            finally:
                self._mv = None
                self._opened = False

    def __enter__(self) -> MemoryStore:
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self._opened else "closed"
        return f"MemoryStore(path='{self._path}', state={state})"
