# Canonical Corpus V1.1 narrow-patch build report

## Scope

Only the explicit next-unit/appendix hard stop, minimum interpretable table relation, and bidirectional substantive-evidence role gate were implemented. No retrieval or generation component was built.

## Build and integrity

- Tests: **PASS** (43 tests).
- Documents/pages/chunks: 27/339/3482.
- Validation: **PASS**; hard errors: 0.
- Deterministic rerun: **PASS** (all requested artifacts byte-identical).
- V0 artifacts unchanged: **PASS**.
- V1 artifacts unchanged: **PASS**.
- Source PDF and processed-JSON hashes unchanged: **PASS**.
- Lifecycle/status mutation count: **0**.
- Coverage: chunked 483970, excluded 0, unaccounted 0, overlap 0, ratio 1.000000.

## Targeted regression

- Result: **11/11 PASS**; 0 FAIL.

| V1 rank | Result | V1.1 matches | Final roles | Observed behavior |
|---:|---|---:|---|---|
| 2 | PASS | 1 | content | Department 8 is content. |
| 3 | PASS | 1 | review | Ambiguous OCR table evidence remains review. |
| 6 | PASS | 1 | review | Ambiguous OCR table evidence remains review. |
| 8 | PASS | 2 | content | 2.2 and the explicit 2.3/footer locus are separate, fully accounted chunks. |
| 9 | PASS | 12 | content, context_only, review | Clause 2 terminates before separately accounted appendix headings. |
| 10 | PASS | 1 | content | IELTS 5.0 is grouped with its contiguous criterion/condition/value row. |
| 14 | PASS | 1 | review | Ambiguous OCR table evidence remains review. |
| 15 | PASS | 1 | context_only | The non-substantive locus is not content. |
| 18 | PASS | 1 | content | Complete processed OCR clause is content. |
| 20 | PASS | 1 | context_only | The non-substantive locus is not content. |
| 26 | PASS | 1 | context_only | The non-substantive locus is not content. |

## Processed-structure non-regression

- Status: **PASS**.
- Previously correct boundary controls: 5/5.
- Previously complete semantic controls: 4/4.
- Previously adequate context controls: 5/5.

## V1 → V1.1 comparison

- Chunks: 3518 → 3482 (-36).
- Stable chunk IDs: 3403.
- Respan IDs removed/added: 115/79.
- Role deltas (content/context_only/review): -6/-27/-3.

## Targeted QA

`qa_sample_v1_1.jsonl` contains 26 unique affected loci, promotion/demotion guards, and unchanged processed-structure controls. Every HUMAN field is blank.

## Remaining limits

- OCR-corrupted tables remain review; no column reconstruction, OCR correction, reordering, or inferred cell content was performed.
- Detached appendix/table/admin spans remain in the source ledger as context_only, review, or conservative fallback content; no source text was deleted.
- Retrieval roles are metadata only. No BM25, dense index, embeddings, fusion, reranker, LLM, API, or chatbot was implemented.
