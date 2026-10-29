# Traceability

Every table, figure, and numeric claim in the accompanying manuscript, traced back to the file and evidence in this repository that produces it. Compiled 2026-10-29. Status legend: **VERIFIED** = recomputed/located directly from this repo's code or data and matches the paper. **MISMATCH** = recomputed value differs from the paper's claim. **NOT IN CODE** = no corresponding implementation found in this repo. **UNVERIFIABLE** = cannot be checked from this repository alone (e.g. hardware self-report).

## 1. Evaluation tables (Tables 3–6 and the per-question appendix)

All four aggregate tables and the 30-row per-question appendix in the paper are reproducible from `rag_results.json` to three decimal places, **provided the original 30-question subset is used** (see §2 below) — not all rows currently in this file.

| Claim | Evidence in this repo |
|---|---|
| Cosine 0.780, F1 0.285, BERTScore 0.359, ROUGE-L 0.182, BLEU-2 0.095 (RAG, N=30) | `rag_results.json`, averaged over the matching 30-question subset: 0.7797 / 0.2852 / 0.3585 / 0.1821 / 0.0953 |
| Same 5 metrics, generator-only baseline: 0.625 / 0.241 / 0.282 / 0.154 / 0.075 | `rag_results.json` `generator_*` fields, same subset: 0.6245 / 0.2412 / 0.2821 / 0.1535 / 0.0754 |
| Estimated Precision@5 0.297, Recall@5 0.205 | `rag_results.json` `precision_at_k`/`recall_at_k`, same subset: 0.2967 / 0.2049 |
| Semantic-alignment tiers: High 6 (20%), Intermediate 18 (60%), Low 6 (20%) | Re-thresholding `cosine_similarity` at 0.85/0.70 cutoffs over the same subset reproduces 6/18/6 exactly — **but this thresholding script is not itself present anywhere in this repository** (see §4) |
| Per-question B2S/RS/FS/BS/CS values | 26 of 30 rows match `rag_results.json` to ±0.002; 4 rows differ by more, one substantially (see §5) |

## 2. Dataset scope

- `pcos_qna.json` in this repository currently has **41 questions**. The paper's reported evaluation covers a specific, enumerated **30-question subset** (also listed in the manuscript's own Methodology section). The 11 extra questions were added to this file after the reported evaluation and are not reflected in the paper's tables.
- `Data/` (populated by you — see `PAPERS.md`) should contain **67 papers**, matching the manuscript exactly. (An earlier internal draft of this document said 71; recounted and corrected — 67 is correct.)
- **If you re-run `main.py` on the current 41-question file, you will not reproduce the paper's numbers.** The aggregate actually printed by the last full run (all 41 rows, see `rag_results.json`'s own trailing log) is: BLEU-2 0.0795, ROUGE-L 0.1700, F1 0.2692, BERTScore 0.3313, Cosine 0.7499, Precision@5 0.2638, Recall@5 0.1984 — all below the paper's N=30 figures. To reproduce the paper exactly, restrict evaluation to the original 30 questions (ask the maintainers for the exact list, or see the manuscript's Methodology section, which enumerates them by category).

## 3. Model / architecture specifications

| Claim | Evidence |
|---|---|
| `all-MiniLM-L6-v2`: 384-dimensional embeddings | Model's own `config.json`: `hidden_size: 384` |
| `all-MiniLM-L6-v2`: 6 layers, 12 attention heads | `config.json`: `num_hidden_layers: 6`, `num_attention_heads: 12` |
| `all-MiniLM-L6-v2`: ~22 million parameters | Counted directly from `model.safetensors` header: 22,713,216 |
| `all-MiniLM-L6-v2`: truncates input beyond 256 word-pieces | `tokenizer_config.json`: `model_max_length: 256` |
| TinyLlama-1.1B-Chat-v1.0: 2048-token context window | Model's own `config.json`: `max_position_embeddings: 2048` |
| TinyLlama-1.1B-Chat-v1.0: ~1.1B parameters | Counted directly from `model.safetensors` header: 1,100,048,384 |
| `max_new_tokens` dynamically capped between 32 and 512 | `ResponseGenerator.py`, `generate_response()`: `max(32, min(512, max_total_tokens - input_len))` |
| Greedy decoding, repetition penalty 1.1 | `ResponseGenerator.py`: `do_sample=False, repetition_penalty=1.1` |

(Model config files are not included in this repo — download the models per `README.md` and check the claims above against your own copies.)

## 4. Pipeline / configuration claims

| Claim | Evidence |
|---|---|
| Chunk size: up to 500 words, no overlap | `DocumentRetriever.py`, `__init__`: `chunk_size=500`; `load_documents()` packs sentences greedily with no overlap |
| Retrieval depth: top-k = 5 by default | `main.py` argparse default `--retrieval_k 5`; `DocumentQA.get_answer(k=5)` |
| k reduced automatically when context exceeds the token budget; actual k logged per question | `DocumentQA.py`, `get_answer()` (the `while k >= 1` retry loop); every row in `rag_results.json` carries a `k_used` field |
| FAISS `IndexFlatL2` (exact L2 search) | `DocumentRetriever.py`, `build_index()`: `faiss.IndexFlatL2(dimension)` |
| Cosine-similarity relevance threshold 0.75 for Precision@k/Recall@k | `main.py` argparse default `--retrieval_relevance_threshold 0.75` |
| Semantic-alignment tiers at 0.85/0.70 cosine cutoffs | **Not implemented by any script in this repository.** `main.py`'s only labeling function, `assign_groundedness_label()`, uses different thresholds (0.90/0.75) and different label names ("Grounded"/"Partially Grounded"/"Hallucinated"). The 0.85/0.70 tiers used in the paper's Table 6 and appendix were computed by a method not present in this codebase. |
| Extractive-sentence fallback when generation fails | `DocumentQA.py`, `_extractive_fallback()` — fires when the generator's output is empty or one of a few known failure strings (see `get_answer()`) |

## 5. Known data discrepancies (not yet resolved)

- One per-question result (**"What are the observed benefits of myo-inositol and folic acid supplementation in PCOS?"**) differs substantially between the paper's reported value and the current `rag_results.json` — the two generated answers share an identical opening clause then diverge completely (text-similarity ratio 0.327 of 1.0). Root cause not determined; this repository's current entry for that question should not be assumed to match the published paper until investigated.
- Three smaller per-question differences (beyond rounding) exist for the insulin-resistance, GI-diet, and mental-health questions.
- A heuristic check (string-matching each generated answer against its own retrieved chunks) suggests the extractive fallback may have fired for at most 2 of the 41 current rows — unconfirmed, not a precise count.

## 6. What this document does not cover

- Hardware specs and total-inference-time claims in the paper are self-reported and cannot be verified from this repository.
- Background/epidemiological citations (e.g., PCOS prevalence rates) are sourced to the literature, not to this repository's code or data — see the manuscript's own bibliography.
- Statistical claims computed over the 30-question subset (mean answer length, Pearson correlations between cosine similarity and lexical metrics) were verified against `rag_results.json` at the time this document was written; re-verify if `rag_results.json` changes.

---

*This file documents reproducibility status, not a complete audit log of every change made to this project. If you update `rag_results.json`, `pcos_qna.json`, or the pipeline code, re-check the claims above before assuming they still hold.*
