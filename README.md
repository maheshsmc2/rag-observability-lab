# RAG Observability Lab

This project demonstrates how to **inspect, debug, and understand**
a Retrieval-Augmented Generation (RAG) system at every stage.

## Why this project exists
Most RAG demos show *final answers*.
This project shows **how the answer was decided**.

## Pipeline
Query → Dense Retrieval → BM25 → Hybrid Scoring → Reranking → Final Context

## Key Feature: Retrieval Trace
Each query produces a JSON trace containing:
- Dense scores
- BM25 scores
- Hybrid scores (with alpha)
- Reranker scores
- Timing per stage

This allows precise debugging of:
- wrong rankings
- poor alpha choice
- dense vs lexical conflicts
- reranker effectiveness

## Files
- `retriever.py` — Hybrid + reranker retrieval engine
- `trace_helpers.py` — Non-intrusive observability tools
- `run_trace.py` — Run a query and generate a trace
- `trace/trace_sample.json` — Example trace output

## Example Use
```bash
python run_trace.py
## Day 69 — Reranker-Aware Confidence Gating

Modern RAG pipelines often use cross-encoder rerankers whose scores are
**unnormalized logits** (frequently negative). Absolute thresholds that work
for dense or hybrid retrieval do not generalize to rerankers.

### Problem
- Reranker scores are not comparable to similarity scores
- Absolute `min_score` gates caused false negatives
- Ambiguous queries sometimes passed without dominance checks

### Solution
This project implements **route-aware confidence gating**:

#### Non-rerank routes (dense / hybrid)
- Absolute confidence gate: `best_score >= min_score`
- Dominance gate: `(top1 - top2) >= MIN_SCORE_MARGIN`

#### Rerank routes (hybrid + cross-encoder)
- **Absolute gate disabled** (scores are uncalibrated)
- Strong dominance gate only:
  `(top1 - top2) >= RERANK_MIN_MARGIN`

### Benefits
- Correct handling of negative reranker scores
- Robust rejection of ambiguous or nonsensical queries
- Meaningful policy queries pass even when scores are negative
- Safer, production-style RAG behavior

### Key Principle
> Confidence must be **relative** when scores are unnormalized.

This mirrors real-world RAG systems used in production assistants,
where rerankers are judged by dominance, not raw score values.
## Example: Safe Refusal (Ambiguous Match)

Query:
> explain attendance policy

Result:
- passed_confidence_gate: false
- reason: Low score dominance (ambiguous match)
- best_score: -11.00
- score_margin: 0.08

Why:
Top-2 reranked documents were too close in score, indicating ambiguity.
System refused to answer instead of hallucinating.
## Example: Confident Answer (Clear Dominance)

Query:
> probation leave policy

Result:
- passed_confidence_gate: true
- best_score: -4.79
- score_margin: 3.57

Why:
Clear dominance between top-1 and top-2 results.
System allowed answer despite uncalibrated reranker logits.
