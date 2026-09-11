# Judge-first S2G-RAG research architecture

## Status and isolation

This implementation is the active research-grade, judge-first S2G controller
used by the Streamlit product. The retired retrieval-first controller has been
removed. Canonical Corpus V1.1 remains read-only.

Configuration: `config/s2g_research.json`.

## State machine

```text
Original question q
  -> initialize C0 = empty Evidence Context
  -> J0 = Judge(q, C0)
  -> take the first structured gap only (K=1)
  -> build q_tilde = q + target/slot, otherwise q + description
  -> retrieve top 6 parent chunks
  -> deterministic source-ordered sentence segmentation
  -> constrained Evidence Extractor returns evidence_global_ids only
  -> append new, valid sentence records to Ct
  -> Judge(q, Ct) again
  -> repeat for at most four retrieval turns
  -> decoupled Answer Reasoner receives q and selected Ct evidence only
```

`max_turns=4` means four retrieval/extraction opportunities. Because the Judge
runs before retrieval and after each context update, the maximum is five Judge
calls, four Evidence Extractor calls, and one Answer Reasoner call.

## Module boundaries

| Module | Input | Output | Invariant |
|---|---|---|---|
| S2G Judge | Original question and current `Ct` | `sufficient`, `gap_items` | Context-only; never answers |
| Gap Query Builder | Original question and first gap | Retrieval query | Deterministic K=1 mapping |
| Retriever | Query, `top_k`, turn | Ranked parent chunks | Pluggable BM25/dense/hybrid |
| Sentence segmenter | Immutable chunk text | Stable sentence records | No non-whitespace loss |
| Evidence Extractor | Question, active gap, sentence pool | `evidence_global_ids` | Pointers only; no generated evidence |
| Evidence Context | Current context and selected pointers | Append-only context | Deduplicated, bounded, traceable |
| Answer Reasoner | Original question and final selected sentences | Grounded answer | No unselected parent text |

## Structured gap contract

```json
{
  "sufficient": false,
  "gap_items": [
    {
      "category": "attribute",
      "target": "học bổng",
      "slot": "điều kiện",
      "description": "Điều kiện sinh viên cần đáp ứng"
    }
  ]
}
```

Allowed categories are `bridge_entity`, `attribute`, `relation`,
`evidence_span`, and `other`. `sufficient=true` requires an empty gap list;
`sufficient=false` requires at least one gap. Although the Judge may describe
up to three gaps for auditability, query construction uses only the first gap.

## Evidence pointer contract

```json
{"evidence_global_ids":["s2g:<chunk-id>:<start>:<end>:<hash>"]}
```

The IDs are stable strings rather than local integer positions. Each ID binds
the parent chunk, exact character interval, and sentence-content hash. The
provider schema enumerates the exact current pool, and local validation rejects
unknown or duplicate pointers. Source text is never rewritten.

## Retrieval modes

All modes implement:

```python
retrieve(query: str, top_k: int, turn_index: int) -> RetrievalBatch
```

Supported configuration values:

- `bm25`
- `dense`
- `hybrid_rrf`
- `hybrid_rrf_reranker`

The configured `top_k=6` is the number of parent chunks exposed to the
Evidence Extractor. Internal BM25/dense candidate depths remain separate and
are retained in the retrieval audit.

## Stop and failure taxonomy

Semantic terminal states:

- `STOP_SUFFICIENT`
- `STOP_MAX_TURNS`
- `STOP_NO_VALID_NEW_QUERY`
- `STOP_EMPTY_RETRIEVAL`
- `STOP_NO_NEW_EVIDENCE`

Provider, schema, retriever, or Answer Reasoner failures use `STOP_ERROR`, set
`closed=false`, and are not interpreted as evidence insufficiency.

## Trace contract

Every Judge turn records:

- original question and turn index;
- full `evidence_context_before`;
- Judge raw/parsed provider call record;
- all predicted gaps and the active K=1 gap;
- derived retrieval query;
- ranked results, scores, and retrieval audit;
- exact sentence pool;
- Extractor raw/parsed provider call record;
- selected pointer IDs and admitted sentence records;
- full `evidence_context_after`;
- semantic or infrastructure stop reason.

The trace validator checks contiguous turns, before/after continuity,
append-only evidence, pointer uniqueness, and final-context equality. Runtime
interfaces contain no ground-truth or evaluator fields.

## Product binding

The research factory builds the complete pipeline from the frozen full-PDF
catalog and existing OpenAI runtime. `src/product_s2g/service.py` binds the
Streamlit product to this pipeline. The provider/controller failure audit and
the final result artifact are persisted under the product request ID.

## Validation

Run in the approved Miniconda environment:

```bash
/opt/miniconda3/envs/ml-env/bin/python -m pytest -q tests/s2g_research
```

The synthetic suite covers the judge-first order, maximum call budget,
top-six retrieval, gap K=1, repeated evidence, empty retrieval, retriever
failure, invalid Judge output, duplicate queries, strict provider field names,
retriever adapters, and evidence-only answer input.

One audited query can be executed through the product CLI with:

```bash
OPENAI_API_KEY="$(launchctl getenv OPENAI_API_KEY)" \
  /opt/miniconda3/envs/ml-env/bin/python -m \
  src.product_s2g.cli "Câu hỏi cần tra cứu"
```

The runtime starts from `C0`; the product assigns a unique request ID and
separates successful and failed traces under `outputs/s2g_product/runtime/`.
