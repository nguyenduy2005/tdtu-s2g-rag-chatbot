# End-to-end S2G-RAG chatbot

## Purpose

This deployment provides a usable Vietnamese regulatory-document chatbot over
all readable material in the repository's 27 source PDFs. It uses the validated
Canonical Corpus V1.1 and widens deployment retrieval eligibility so every
non-empty `content`, `review`, and `context_only` chunk remains searchable.

## End-to-end flow

```text
User question
  -> initialize C0 as empty
  -> S2G Judge reads only (question, C0)
  -> structured gap item
  -> deterministic gap-guided query
  -> BM25 + multilingual E5 over 3,482 chunks
  -> reciprocal-rank fusion and cross-encoder reranking
  -> top 6 parent chunks
  -> source-ordered sentence segmentation
  -> constrained pointer-only Evidence Extractor
  -> append-only evidence context
  -> S2G Judge reads the updated context
  -> repeat for at most 4 retrieval turns
  -> decoupled Answer Reasoner reads selected evidence only
  -> answer with document/page/chunk citations and trust warnings
```

The extractor can return only stable sentence IDs from its current
request-specific enum. Answer generation can cite only chunk IDs in the final
S2G Evidence Context and cannot inspect unselected parent-chunk text. Document
text is treated as untrusted data, and the answer prompt forbids following
instructions embedded in source documents.

## Coverage

The deployment verifies the frozen `chunks.jsonl` SHA-256 before loading. It
indexes every non-empty canonical chunk, without altering `source_text`:

- 27/27 source PDF files;
- 339 declared source pages;
- 3,482/3,482 canonical chunks;
- 1,091 `content` chunks;
- 2,030 `review` chunks;
- 361 `context_only` chunks;
- 1,540 OCR-derived and 1,942 embedded-text chunks.

`review` and `context_only` are searchable to avoid losing documents that lack
an independently eligible `content` chunk. They are not silently upgraded:
the response includes their original role and flags, and the UI labels
low-trust citations for PDF verification.

Two source pages contain no usable canonical text. Visual inspection confirmed
that the MOS document's page 4 is blank. The tuition document's page 76 is a
severely corrupted/bleed-through scan from which conservative Vietnamese OCR
does not produce reliable evidence. Both pages remain explicit in the coverage
notice; the chatbot never invents content for them.

Coverage therefore means **all non-empty, readable canonical source text is
searchable**, not that every query is guaranteed to retrieve the correct text
or that unreadable pixels can be recovered.

## Components

- Configuration: `config/s2g_product.json`
- Derived manifest: `data/product/s2g_full_pdf/deployment_manifest.json`
- Corpus verifier: `src/product_s2g/corpus.py`
- Full-corpus hybrid index: `src/product_s2g/retrieval.py`
- S2G product orchestration: `src/product_s2g/service.py`
- Audited OpenAI runtime: `src/s2g_runtime/`
- Judge-first state machine: `src/retrieval/s2g_research/pipeline.py`
- Research/controller configuration: `config/s2g_research.json`
- Grounded answer contract: `src/product_s2g/answer.py`
- Streamlit UI (primary): `src/product_s2g/streamlit_app.py`
- Streamlit theme: `.streamlit/config.toml`
- CLI: `src/product_s2g/cli.py`
- Runtime audit/cache: `outputs/s2g_product/`

## Streamlit UI

The primary user interface uses Streamlit chat components. It loads the
verified catalog immediately and caches the heavy S2G service once per server
process. It provides:

- conversational history within the browser session;
- full-corpus coverage and API-key status;
- an inspectable list of all 27 documents;
- answer confidence, rounds, evidence size, and semantic stop reason;
- expandable citations with page ranges, roles, quality flags, and excerpts;
- PDF download for direct source verification;
- visible low-trust warnings for OCR/review/context-only evidence.

## Run locally with Streamlit

```bash
/opt/miniconda3/envs/ml-env/bin/python -m pip install -r requirements-product.txt
/opt/miniconda3/envs/ml-env/bin/python -m src.product_s2g.build_manifest
OPENAI_API_KEY="$(launchctl getenv OPENAI_API_KEY)" \
  /opt/miniconda3/envs/ml-env/bin/python -m streamlit run \
  src/product_s2g/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

The first initialization creates the full-corpus embedding cache. Runtime
requests are recorded by request ID under `outputs/s2g_product/runtime/`, with
complete before/after Evidence Context and controller-attempt audit records.

Open `http://127.0.0.1:8501/`. Press **Khởi tạo S2G engine** to warm the models,
or submit a question and let the UI initialize them automatically.

## Validation performed

- Full catalog: 27 PDFs, 3,482 chunks, all three roles.
- Derived source hash: frozen Canonical Corpus V1.1 hash matches.
- Real index build: PASS on Apple MPS.
- Streamlit AppTest: PASS with title, chat input, and document selector rendered
  without application exceptions.
- Judge-first order, T=4, top-k=6, K=1 gap mapping, pointer validity, append-only
  context, semantic stops, infrastructure failures, and evidence-only answer
  input are covered by automated tests.

## Limitations

- Broad deployment eligibility improves document coverage but can increase
  noise from OCR, tables, headings, and administrative fragments.
- Low-trust warnings are essential; users should open the cited PDF for formal
  decisions.
- The controller and answer generator require the OpenAI API.
- This is document-grounded informational assistance, not an authority for
  lifecycle validity or legal interpretation.
