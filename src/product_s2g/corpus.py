"""Read-only full-corpus catalog and reproducible deployment manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.product_s2g.io import atomic_json
from src.retrieval.corpus import RetrievalDocument, load_retrieval_corpus

from .config import ProductConfig


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FullCorpusCatalog:
    documents: tuple[RetrievalDocument, ...]
    chunks: dict[str, dict[str, Any]]
    source_files: dict[str, Path]
    document_pages: dict[str, int]
    document_ids: tuple[str, ...]
    page_count: int
    unavailable_pages: tuple[dict[str, Any], ...]
    role_counts: dict[str, int]
    extraction_counts: dict[str, int]
    canonical_chunks_sha256: str

    @classmethod
    def load(cls, config: ProductConfig) -> "FullCorpusCatalog":
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        coverage = json.loads(config.coverage_path.read_text(encoding="utf-8"))
        expected = (manifest.get("artifacts") or {}).get("chunks.jsonl")
        actual = sha256_file(config.corpus_path)
        if expected != actual:
            raise ValueError("Canonical Corpus V1.1 chunks hash mismatch")
        if manifest.get("document_count") != 27 or manifest.get("chunk_count") != 3482:
            raise ValueError("Unexpected frozen canonical corpus cardinality")
        if manifest.get("coverage", {}).get("accounted_ratio") != 1.0:
            raise ValueError("Canonical corpus source coverage is not complete")

        retrieval_documents, loader_report = load_retrieval_corpus(
            config.corpus_path,
            indexed_roles=config.indexed_roles,
            text_field=config.text_field,
        )
        chunks: dict[str, dict[str, Any]] = {}
        roles: Counter[str] = Counter()
        extraction: Counter[str] = Counter()
        by_document: defaultdict[str, int] = defaultdict(int)
        for line in config.corpus_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row["chunk_id"]
            if chunk_id in chunks:
                raise ValueError(f"Duplicate canonical chunk ID: {chunk_id}")
            if not isinstance(row.get(config.text_field), str) or not row[config.text_field].strip():
                raise ValueError(f"Empty searchable source text: {chunk_id}")
            chunks[chunk_id] = row
            roles[row["retrieval_role"]] += 1
            methods = set(row.get("extraction_methods") or ())
            extraction["ocr"] += int(any("ocr" in method for method in methods))
            extraction["embedded"] += int(not any("ocr" in method for method in methods))
            by_document[row["document_id"]] += 1
        if len(chunks) != manifest["chunk_count"] or len(retrieval_documents) != len(chunks):
            raise ValueError("Full-corpus loader silently excluded a canonical chunk")

        source_files: dict[str, Path] = {}
        document_pages: dict[str, int] = {}
        unavailable: list[dict[str, Any]] = []
        for row in coverage["documents"]:
            source = config.raw_pdf_root / row["source_file"]
            if not source.is_file():
                raise ValueError(f"Missing source PDF: {source}")
            source_files[row["document_id"]] = source
            document_pages[row["document_id"]] = int(row["pages"])
            if by_document[row["document_id"]] == 0:
                raise ValueError(f"No searchable chunk for source document: {row['document_id']}")
            for page in row.get("pages_without_chunks", ()):
                unavailable.append({
                    "document_id": row["document_id"],
                    "source_file": row["source_file"],
                    "page": page["page"],
                    "reason": page["reason"],
                    "quality_flags": page.get("quality_flags", []),
                })
        raw_pdf_names = {path.name for path in config.raw_pdf_root.glob("*.pdf")}
        coverage_names = {path.name for path in source_files.values()}
        if raw_pdf_names != coverage_names or len(raw_pdf_names) != 27:
            raise ValueError("Raw PDF set differs from the canonical coverage ledger")
        if loader_report.get("indexed_chunks") not in {None, len(chunks)}:
            raise ValueError("Retrieval corpus loader reported incomplete indexing")
        return cls(
            documents=tuple(retrieval_documents),
            chunks=chunks,
            source_files=source_files,
            document_pages=document_pages,
            document_ids=tuple(sorted(source_files)),
            page_count=int(manifest["page_count"]),
            unavailable_pages=tuple(unavailable),
            role_counts=dict(roles),
            extraction_counts=dict(extraction),
            canonical_chunks_sha256=actual,
        )

    def manifest(self, config: ProductConfig) -> dict[str, Any]:
        role_by_document: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for row in self.chunks.values():
            role_by_document[row["document_id"]][row["retrieval_role"]] += 1
        unavailable_by_document: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.unavailable_pages:
            unavailable_by_document[row["document_id"]].append(row)
        return {
            "schema_version": "s2g-full-pdf-deployment-manifest-v1",
            "product_id": config.product_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "canonical_corpus": str(config.corpus_path.relative_to(Path(__file__).resolve().parents[2])),
            "canonical_chunks_sha256": self.canonical_chunks_sha256,
            "source_pdf_count": len(self.source_files),
            "source_page_count": self.page_count,
            "indexed_chunk_count": len(self.documents),
            "indexed_document_count": len({row.document_id for row in self.documents}),
            "indexed_roles": list(config.indexed_roles),
            "role_counts": self.role_counts,
            "extraction_counts": self.extraction_counts,
            "documents": [
                {
                    "document_id": document_id,
                    "source_file": self.source_files[document_id].name,
                    "page_count": self.document_pages[document_id],
                    "chunk_count": sum(role_by_document[document_id].values()),
                    "role_counts": dict(role_by_document[document_id]),
                    "unavailable_pages": unavailable_by_document.get(document_id, []),
                }
                for document_id in self.document_ids
            ],
            "unavailable_pages": list(self.unavailable_pages),
            "coverage_statement": "All non-empty Canonical Corpus V1.1 source_text chunks are searchable; unreadable/blank pages are explicitly listed and never inferred.",
            "canonical_corpus_modified": False,
        }

    def write_manifest(self, config: ProductConfig) -> dict[str, Any]:
        value = self.manifest(config)
        atomic_json(config.derived_manifest, value)
        return value
