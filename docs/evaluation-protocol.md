# S2G-RAG evaluation protocol

## Scope and status

This protocol measures the current full-PDF product on five dimensions:

1. retrieval recall;
2. reciprocal rank (MRR);
3. evidence-group coverage (EGC);
4. human-reviewed answer quality;
5. end-to-end latency and OpenAI API cost.

`data/evaluation/s2g_pilot_v1/queries.json` is a 20-query, 27-evidence-group
pilot grounded in Canonical Corpus V1.1. It is currently
`DRAFT_AGENT_VERIFIED_PENDING_HUMAN`. The raw scores may be used for engineering
diagnosis, but must not be described as results from an independent human-annotated
study. Official thesis reporting requires completion of
`ground_truth_review.csv` and changing each verification status only after review.

## Ground-truth firewall

The benchmark file contains evidence labels. The runtime export contains exactly
`id` and `query`; loaders reject every additional field. Retrieval and S2G execution
therefore cannot inspect ground truth. The offline evaluator loads evidence labels
only after rankings/results have been finalized.

The default CLI requires `human_verified`. `--allow-pending-human` is an explicit
opt-in for provisional engineering measurements.

## Retrieval metrics

- `Recall@k`: unique relevant chunk IDs retrieved in the first `k`, divided by all
  unique relevant chunk IDs for the query.
- `MRR@10`: reciprocal rank of the first relevant chunk in the first 10; zero when
  none is present.
- `Evidence-group coverage@k`: required evidence groups covered in the first `k`,
  divided by the number of required groups. A group is covered when at least one of
  its acceptable chunk IDs appears.
- `Complete-evidence@k`: true only when every required group is covered.

Metrics are reported per query, as macro averages, and by query type. For S2G the
report deliberately separates the iterative candidate pool, chunks presented by
the reranker, sentence-selected parent chunks, and answer citations. These are not
interchangeable stages.

## Answer quality

Automated checks cover only enforceable contracts: citation IDs are drawn from final
evidence and a substantive answer has citations. Semantic quality is reviewed in
`answer_quality_review.csv`. A reviewer records TRUE/FALSE for correctness,
completeness, faithfulness to evidence, citation adequacy, and clarity, plus notes.
All HUMAN fields are blank when generated. Report both numerator/denominator and
percentage; do not silently exclude insufficient answers.

## Latency and cost

Retrieval traces contain BM25, dense, fusion, reranker, and total latency. S2G traces
contain per-retrieval latency, provider-call latency, and end-to-end pipeline latency.
Reports use mean, median, P95, and maximum across queries.

Token cost uses recorded Responses API usage (`input_tokens`, cached input tokens,
and `output_tokens`). The configuration records the observed price date and source.
The estimate excludes local compute, electricity, and network costs and must be
reported as an API estimate, not total system cost.

## Reproducible commands

Use the project environment:

```bash
conda run -n ml-env python -m src.evaluation.cli validate
conda run -n ml-env python -m src.evaluation.cli export-runtime \
  --output data/evaluation/s2g_pilot_v1/runtime_queries.json
conda run -n ml-env python -m src.evaluation.cli run-b0 \
  --runtime-queries data/evaluation/s2g_pilot_v1/runtime_queries.json \
  --output outputs/evaluation/s2g_pilot_v1/b0_rankings.json
conda run -n ml-env python -m src.evaluation.cli evaluate-b0 \
  --rankings outputs/evaluation/s2g_pilot_v1/b0_rankings.json \
  --output outputs/evaluation/s2g_pilot_v1/b0_metrics.json
conda run -n ml-env python -m src.evaluation.cli run-s2g \
  --runtime-queries data/evaluation/s2g_pilot_v1/runtime_queries.json \
  --output-root outputs/evaluation/s2g_pilot_v1/s2g_runtime
conda run -n ml-env python -m src.evaluation.cli evaluate-s2g \
  --runtime-root outputs/evaluation/s2g_pilot_v1/s2g_runtime \
  --output-root outputs/evaluation/s2g_pilot_v1
```

Until human verification is complete, append `--allow-pending-human` only to
validation, export, and offline evaluation commands. Execution commands consume the
already-sanitized runtime file and never load the benchmark.

## Interpretation limits

Twenty stratified questions are a pilot, not a comprehensive estimate of all possible
student questions. Query authoring, ground-truth review, model behavior, OCR noise,
and document lifecycle ambiguity are separate threats to validity. No prompt,
retrieval setting, or query may be tuned after inspecting individual pilot outcomes
and then reported on the same set as an unbiased result.
