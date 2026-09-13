# S2G-RAG pilot evaluation results

## Status

**Execution: PASS (20/20 queries completed).**  This is an engineering pilot, not
yet a frozen thesis benchmark. The benchmark contains agent-assisted source
verification and **0/20 independently human-verified queries**. The strict official
validator therefore fails by design. Results below must be labelled *provisional*
until `ground_truth_review.csv` is completed.

- Corpus: Canonical Corpus V1.1, 27 PDFs, 3,482 indexed chunks.
- Corpus SHA-256: `fd2358332668de3dca8c099ff6a3dd0cf1d23b47c378f7c92835df4a08c34c02`.
- Pilot: 20 queries, 27 required evidence groups, 23 unique GT chunks.
- Distribution: 5 direct lookup, 5 paraphrase, 3 terminology mismatch,
  4 multi-evidence, 3 cross-document.
- Runtime GT firewall: PASS. Runtime input contains only `id` and `query`.
- Model: requested `gpt-5-nano`; all 89 responses resolved to
  `gpt-5-nano-2025-08-07`; all response IDs were present and unique.
- API/infrastructure failures: 0.

## Metric definitions

- Recall@k is the proportion of unique GT chunks found in top-k, macro-averaged.
- MRR@10 uses the rank of the first GT chunk.
- Evidence-group coverage (EGC)@k is the proportion of required evidence groups
  with at least one acceptable chunk in top-k.
- Complete evidence is achieved only when every group for a query is covered.

Every pilot evidence group currently has one acceptable chunk, so Recall@k and
EGC@k are numerically equal. They remain separate implementations because future
groups may contain multiple acceptable chunks.

## B0 retrieval results

| Stage | R@1 | R@3 | R@5 | R@10 | MRR@10 | EGC@10 | Complete @10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.525 | 0.800 | 0.850 | 0.875 | 0.746 | 0.875 | 16/20 |
| Dense | 0.500 | 0.625 | 0.775 | 0.850 | 0.672 | 0.850 | 16/20 |
| Hybrid + RRF | 0.525 | 0.825 | 0.825 | 0.875 | 0.742 | 0.875 | 16/20 |
| Hybrid + RRF + reranker | **0.575** | **0.825** | **0.825** | **0.825** | **0.767** | **0.825** | **15/20** |

The reranker improves rank-one performance and MRR over raw hybrid, but it pushes
one query's complete evidence out of top-10 (16 to 15). This is a measured trade-off,
not evidence for tuning on the same pilot.

For the final B0 stage, direct lookup is 1.000 R@10, terminology mismatch is
1.000, paraphrase is 0.800, cross-document is 0.667, and multi-evidence is 0.625.
Only 2/4 multi-evidence queries and 1/3 cross-document queries have complete evidence
at top-10.

## S2G retrieval/evidence results

The candidate and final-evidence layers are reported separately. The iterative
candidate pool contains the top six reranked chunks from each executed retrieval
round. Selected evidence contains parent chunks admitted by the sentence selector;
answer citations are the subset used by the answer reasoner.

| Stage | R@1 | R@3 | R@5 | R@10 | MRR@10 | EGC@10 | Complete @10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iterative candidate pool | 0.625 | 0.750 | 0.800 | 0.800 | 0.777 | 0.800 | 14/20 |
| S2G selected evidence | 0.625 | 0.700 | 0.750 | 0.750 | 0.785 | 0.750 | 13/20 |
| Answer citations | 0.600 | 0.750 | 0.750 | 0.750 | 0.767 | 0.750 | 13/20 |

S2G does not outperform B0 top-10 retrieval on this pilot. Compared with B0
hybrid+reranker, S2G candidate R@10 changes from 0.825 to 0.800, and selected/cited
R@10 falls to 0.750. Its selected-evidence MRR@10 is slightly higher (0.785 versus
0.767), showing stronger early precision for recovered evidence but weaker total
coverage.

The main failure is multi-evidence coverage. At the selected-evidence stage its
R/EGC@10 is 0.250 and complete evidence is 0/4. Direct lookup and terminology
mismatch are both 1.000; paraphrase is 0.800; cross-document is 0.667.

### Evidence-flow diagnosis

- P007 and P015: GT is absent from both B0 top-10 and the S2G candidate pool.
- P014 and P018-P019: only one of two required groups is recovered/cited.
- P016: both GT chunks are present in the iterative candidate pool, but neither is
  admitted to selected evidence. This is specifically a selector/controller loss,
  not a first-stage retrieval miss.
- P017: B0 top-10 contains both pilot GT chunks, while the S2G top-six-per-round
  candidate pool contains one. Human review also found that the question has a
  possible ground-truth scope ambiguity between a general and a specific rule.

No retrieval parameter or prompt was changed after observing these cases.

## Agent-assisted answer-quality review

This review is useful for debugging but is **not an independent human evaluation**.
PASS rates below use all 20 queries as the denominator; REVIEW is not counted as
PASS.

| Criterion | PASS | FAIL | REVIEW | PASS / 20 |
|---|---:|---:|---:|---:|
| Correctness | 14 | 5 | 1 | 70% |
| Completeness | 13 | 6 | 1 | 65% |
| Faithfulness | 15 | 4 | 1 | 75% |
| Citation adequacy | 12 | 7 | 1 | 60% |
| Clarity | 18 | 2 | 0 | 90% |

Ten of 20 answers pass all four core criteria (correctness, completeness,
faithfulness, citation adequacy). Dominant problems are failure to answer all parts
of multi-evidence questions, citing only one required document/group, and answering
from a plausible but wrong nearby provision. P013 incorrectly implies that possessing
a MOS certificate alone is enough to skip the course; the source requires an
assessment and threshold. P015 is materially wrong, P016 declares insufficient,
and P019 omits the undergraduate half. P017 remains REVIEW because two applicable
source provisions express timing at different specificity levels.

The untouched human worksheet is `outputs/evaluation/s2g_pilot_v1/answer_quality_review.csv`.

## Latency, tokens, and API cost

### B0 local retrieval latency (seconds/query)

| Component | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| BM25 | 0.011 | 0.010 | 0.020 | 0.022 |
| Dense | 0.088 | 0.062 | 0.131 | 0.766 |
| Reranker | 0.999 | 0.956 | 1.215 | 1.913 |
| Total | **1.099** | **1.017** | **1.368** | **2.694** |

The B0 API cost is zero because its retrieval models ran locally. This does not
include hardware, electricity, or engineering cost.

### S2G resources

- End-to-end latency: mean 22.978 s, median 21.666 s, P95 41.059 s,
  maximum 45.716 s.
- Provider-call latency: mean 20.895 s/query.
- Retrieval latency across iterative rounds: mean 2.046 s/query.
- Retrieval rounds: mean 1.30, median 1, maximum 3.
- Stop reasons: 17 `STOP_SUFFICIENT`, 3 `STOP_NO_NEW_EVIDENCE`.
- Provider calls: 89 total, 4.45/query: 43 judge, 26 sentence selector,
  and 20 answer reasoner calls.
- Tokens: 143,847 input; 65,214 output, including 46,720 reasoning tokens;
  zero cached input tokens were reported.
- Estimated OpenAI API cost: **USD 0.03327795 total**, or
  **USD 0.00166390/query**.

The estimate uses the recorded token usage and the GPT-5 nano prices observed on
2026-09-13: USD 0.05/M input tokens, USD 0.005/M cached input tokens, and
USD 0.40/M output tokens. It excludes local compute and network costs.

## Validity decision and next checkpoint

The software and runtime execution pass; the evidence shows the deployed S2G system
works well on direct/terminology questions but is not yet reliable for multi-evidence
answers. These results are suitable for engineering diagnosis and a progress update.
They are **not yet suitable as final thesis claims**, for two reasons:

1. the benchmark has not received independent human ground-truth verification;
2. answer quality was agent-assisted, and P017 exposes at least one annotation
   ambiguity.

The next step is to review every row in `ground_truth_review.csv`, resolve P017
without looking at retrieval rankings, and freeze the benchmark. Then run a fresh,
untuned official experiment and have answer quality reviewed independently. Do not
change retrieval/controller parameters on the basis of this pilot and reuse the same
questions as a supposedly unbiased test.
