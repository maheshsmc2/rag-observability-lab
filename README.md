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
