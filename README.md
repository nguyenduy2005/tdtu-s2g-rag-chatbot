# Full-PDF S2G-RAG Chatbot

This workspace contains one end-to-end Streamlit product for searching 27
Vietnamese university regulatory PDFs with the judge-first S2G-RAG controller.
The retired retrieval-first controller has been removed. Old benchmarks,
comparator implementations, and experiment reports have also been removed from
the product workspace.

## Data coverage

- 27 source PDFs in `data/raw/`.
- 27 traceability JSON documents in `data/processed/documents/`.
- Frozen searchable corpus in `data/corpus/canonical_corpus_v1_1/`.
- 339 declared pages and 3,482 non-empty canonical chunks.
- Corpus SHA-256:
  `fd2358332668de3dca8c099ff6a3dd0cf1d23b47c378f7c92835df4a08c34c02`.

The product searches all three canonical roles: 1,091 `content`, 2,030
`review`, and 361 `context_only` chunks. Lower-trust evidence remains visibly
flagged for PDF verification.

## Current structure

```text
chatbot/
├── .streamlit/                  # Streamlit theme/server settings
├── config/                      # Product and required S2G policy guards
├── data/
│   ├── raw/                     # 27 immutable source PDFs
│   ├── processed/documents/     # 27 source-traceability JSON files
│   ├── corpus/canonical_corpus_v1_1/
│   ├── evaluation/s2g_pilot_v1/ # 20-query pilot benchmark and GT review
│   └── product/s2g_full_pdf/    # Derived deployment manifest
├── docs/
│   ├── s2g-full-pdf-chatbot.md
│   ├── s2g-research-architecture.md # Judge-first research controller
│   ├── evaluation-protocol-vi.md
│   ├── s2g-pilot-rerun-2026-09-13.md # Latest pilot report
│   └── archive/                 # Superseded/English reports
├── output/pdf/                  # Current user-facing PDF report
├── outputs/
│   ├── s2g_product/cache/       # Reusable full-corpus embedding cache
│   └── evaluation/              # Current and archived experiment artifacts
├── src/
│   ├── product_s2g/             # Streamlit UI and end-to-end orchestration
│   ├── retrieval/               # Retrieval primitives and judge-first S2G
│   ├── s2g_runtime/             # OpenAI, validation, retry, and audit support
│   └── evaluation/              # Benchmark validation, runners, and metrics
└── tests/
    ├── product_s2g/
    ├── s2g_research/
    └── evaluation/
```

The workspace contains one S2G pilot benchmark and its evaluation utilities; it
does not contain comparator-method implementations. Product runtime traces and
the large per-query pilot runtime directory are ignored by Git.

## Install

Use the approved Miniconda environment:

```bash
/opt/miniconda3/envs/ml-env/bin/python -m pip install -r requirements-product.txt
```

## Run Streamlit

```bash
OPENAI_API_KEY="$(launchctl getenv OPENAI_API_KEY)" \
  /opt/miniconda3/envs/ml-env/bin/python -m streamlit run \
  src/product_s2g/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501/`. The first initialization loads the dense encoder
and reranker; later sessions reuse
`outputs/s2g_product/cache/full_corpus_embeddings.npz`.

## Validate

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /opt/miniconda3/envs/ml-env/bin/python -m pytest \
  -p no:cacheprovider tests/product_s2g -q
```

See `docs/s2g-full-pdf-chatbot.md` for the product flow, coverage semantics,
citations, quality warnings, and source limitations. The active controller is
implemented under `src/retrieval/s2g_research/` and configured by
`config/s2g_research.json`. It implements `Judge(C0=empty) -> gap-guided
retrieval -> sentence-pointer extraction -> append-only evidence -> Judge`,
with at most four retrieval turns and six presented chunks per turn.

## Retrieval and answer evaluation

The evaluation foundation under `src/evaluation/` measures Recall@k, MRR@10,
evidence-group coverage, retrieval/end-to-end latency, recorded token usage, and
estimated API cost. It also generates a blank human-review worksheet for answer
quality. See [`docs/evaluation-protocol-vi.md`](docs/evaluation-protocol-vi.md).

The 20-query pilot under `data/evaluation/s2g_pilot_v1/` is agent-assisted and
still awaits human verification. The evaluator hard-fails by default until that
checkpoint is complete; provisional engineering runs require an explicit
`--allow-pending-human` flag.

The current diagnostic run is
`outputs/evaluation/s2g_pilot_rerun_20260913/`. The superseded first run is kept
under `outputs/evaluation/archive/s2g_pilot_v1/` for provenance. See
`outputs/evaluation/README.md` before interpreting experiment artifacts.
