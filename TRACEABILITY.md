# Traceability

Every table, figure, and numeric claim in the accompanying manuscript, traced back to the file and evidence in this repository that produces it. Compiled 2026-10-29. Status legend: **VERIFIED** = recomputed/located directly from this repo's code or data and matches the paper. **MISMATCH** = recomputed value differs from the paper's claim. **NOT IN CODE** = no corresponding implementation found in this repo. **UNVERIFIABLE** = cannot be checked from this repository alone (e.g. hardware self-report).

## 1. Evaluation tables (Tables 3–6 and the per-question appendix)

All four aggregate tables and the 30-row per-question appendix in the paper are reproducible from `rag_results.json` to three decimal places, using the 30 questions in this repo's `pcos_qna.json` (see §2 below).

| Claim | Evidence in this repo |
|---|---|
| Cosine 0.780, F1 0.285, BERTScore 0.358, ROUGE-L 0.182, BLEU-2 0.095 (RAG, N=30) | `rag_results.json`, averaged over the matching 30-question subset: 0.7797 / 0.2852 / 0.35849 / 0.1821 / 0.0953 (0.35849 rounds to 0.358; the manuscript's BERTScore was corrected from 0.359 to 0.358 on 2026-10-30 after a rounding-direction error was caught in an earlier verification pass) |
| Same 5 metrics, generator-only baseline: 0.625 / 0.241 / 0.282 / 0.154 / 0.075 | `rag_results.json` `generator_*` fields, same subset: 0.6245 / 0.2412 / 0.2821 / 0.1535 / 0.0754 |
| Estimated Precision@5 0.297, Recall@5 0.205 | `rag_results.json` `precision_at_k`/`recall_at_k`, same subset: 0.2967 / 0.2049 |
| Semantic-alignment tiers: High 6 (20%), Intermediate 18 (60%), Low 6 (20%) | `compute_semantic_tiers.py` (run it: `python compute_semantic_tiers.py`) reproduces 6/18/6 exactly from `rag_results.json` |
| Per-question B2S/RS/FS/BS/CS values | All 30 of 30 rows match `rag_results.json` exactly (see §5) |

## 2. Dataset scope

- `pcos_qna.json` and `rag_results.json` in this repository both have exactly **30 questions/results**, matching the paper's reported evaluation set precisely (also enumerated in the manuscript's own Methodology section). Re-verified: aggregating `rag_results.json` reproduces Tables 3-6 to within 0.0005.
- `Data/` (populated by you — see `PAPERS.md`) should contain **67 papers**, matching the manuscript exactly.

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
| Semantic-alignment tiers at 0.85/0.70 cosine cutoffs | `compute_semantic_tiers.py`, `tier()` function. (Note: `main.py`'s own labeling function, `assign_groundedness_label()`, uses different thresholds (0.90/0.75) and different label names ("Grounded"/"Partially Grounded"/"Hallucinated") for a separate, older scheme -- `compute_semantic_tiers.py` is the one that matches the paper's Table 6 and Table A1 tier column.) |
| Extractive-sentence fallback when generation fails | `DocumentQA.py`, `_extractive_fallback()` — fires when the generator's output is empty or one of a few known failure strings (see `get_answer()`). Checked against all 30 reported answers via verbatim-match analysis: exactly 1 of 30 ("What are the common metabolic issues associated with PCOS?") bears the fallback's signature; documented in the manuscript's Methodology section (Query Answering Workflow) as of 2026-10-30. |
| Section 6.4's "divergence explicitly marked" methodology rule | Audited all 30 per-question discussion subsections against the paper's own stated rule (automated tier vs. manual judgment, divergence explicitly marked when they disagree): 16 agree, 8 already marked the disagreement, and 6 (items 4, 10, 19, 20, 21, 28) had the same disagreement but were unmarked -- fixed 2026-10-30 by adding a "despite its [tier] automated tier" clause to each, with no score/verdict change. |

## 5. Known data discrepancies

- **ALL RESOLVED:** four per-question results previously differed from the paper's reported values -- "What are the observed benefits of myo-inositol and folic acid supplementation in PCOS?" (the largest gap: two generated answers shared only an opening clause before diverging completely, text-similarity ratio 0.327 of 1.0), plus the insulin-resistance, GI-diet, and mental/emotional-wellbeing questions (smaller gaps). Root cause for the differences was never determined. By author decision, the current `rag_results.json` value was treated as authoritative in every case, and the manuscript's table rows were updated to match. Verified: the paper's aggregate tables were already computed from each question's current value, so no other numbers needed to change. All 30 Table A1 rows now match `rag_results.json` exactly.
- **RESOLVED (2026-10-30):** the extractive fallback's usage was checked precisely against the final 30-question set (verbatim-match analysis against each answer's own retrieved chunks, cross-checked against `DocumentQA._extractive_fallback()`'s actual trigger logic): exactly 1 of 30 rows bears its signature. Documented in the manuscript.
- **Model-name spelling (Reviewer #5):** the manuscript previously spelled the generator model three ways ("Tiny LLaMA", "TinyLLaMA", "TinyLlama"); standardized to "TinyLlama" (the model's actual name) throughout, 2026-10-30.
- **Reference-answer typo (Reviewer #5):** a "notSonly" typo (for "not only") was found in the reference-answer text for the insulin-resistance question, traced to `rag_results.json`'s own `reference_answer` field (a pre-existing data-entry typo, not introduced by manuscript editing). Fixed in both the manuscript and this repository's `pcos_qna.json`/`rag_results.json`, 2026-10-30.
- **OPEN, found during this sync (2026-10-30) -- not yet resolved:** the main working copy of this project (outside this repository -- `E:\SnehaVasishth\archive\RAG-Medical-PCOS-master\RAG-Medical-PCOS-master\pcos_qna.json` and its `rag_results.json`, plus the `Version-2\` mirror of the same) currently contain **41** question/result entries again, not the 30 this repository (`RAG-PCOS`) and the manuscript both correctly have. This repository's own `pcos_qna.json`/`rag_results.json` were re-checked directly as part of this update and confirmed still correct at exactly 30 each -- the regression is confined to the other working copy and has not been investigated further or fixed; flagging it here since other project notes (this file and `discrepancy.docx`/`reviewer7.docx`) have previously and inconsistently described this exact issue as both "resolved" and "still open."

## 6. What this document does not cover

- Hardware specs and total-inference-time claims in the paper are self-reported and cannot be verified from this repository.
- Background/epidemiological citations (e.g., PCOS prevalence rates) are sourced to the literature, not to this repository's code or data — see the manuscript's own bibliography.
- Statistical claims computed over the 30-question subset (mean answer length, Pearson correlations between cosine similarity and lexical metrics) were verified against `rag_results.json` at the time this document was written; re-verify if `rag_results.json` changes.

---

*This file documents reproducibility status, not a complete audit log of every change made to this project. If you update `rag_results.json`, `pcos_qna.json`, or the pipeline code, re-check the claims above before assuming they still hold.*
