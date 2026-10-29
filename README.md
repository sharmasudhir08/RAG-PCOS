# RAG-PCOS

A Retrieval-Augmented Generation (RAG) system for question-answering on Polycystic Ovary Syndrome (PCOS), built for patient-facing, non-specialist use. Source PDFs are chunked and embedded with `all-MiniLM-L6-v2`, indexed with FAISS, and answered with a locally-run TinyLlama-1.1B-Chat-v1.0 generator, grounded in the retrieved passages.

This repository accompanies the manuscript *"Retrieval-Augmented Generation for Medical QA: A Case Study on Polycystic Ovary Syndrome"* (under review). The full paper will be linked here once published — see `CITATION.cff` for how to cite the software in the meantime.

## Pipeline

```
Data/*.pdf --> DocumentRetriever.py --> FAISS index (all-MiniLM-L6-v2 embeddings)
                                              |
question --> DocumentQA.py (retrieve top-k, build context) --> ResponseGenerator.py (TinyLlama) --> answer
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Download the models

The pretrained models are not included in this repository (see `NOTICE`). Download them into a local `models/` folder:

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

### Build the retrieval corpus

This repository does not redistribute the source PDFs (see `PAPERS.md` for why, and for the full list of papers used in the original study). Create a `Data/` folder and populate it with PDF files of PCOS research literature — ideally matching `PAPERS.md`'s titles if you want to reproduce the original study's results.

```
Data/
├── paper-1.pdf
├── paper-2.pdf
└── ...
```

## Running

```bash
python main.py --retrieval_k 5
```

This builds the FAISS index from `Data/`, answers every question in `pcos_qna.json` (both with and without retrieval), scores each answer against its reference with BLEU-2, ROUGE-L, F1, BERTScore, and cosine similarity, and writes full per-question results to `rag_results.json`. Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--retrieval_k` | 5 | top-k chunks requested per question (actual k used may be lower — see below) |
| `--cosine_grounded_threshold` | 0.90 | cosine-similarity cutoff for the "Grounded" label |
| `--cosine_partial_threshold` | 0.75 | cosine-similarity cutoff for the "Partially Grounded" label |
| `--retrieval_relevance_threshold` | 0.75 | cosine-similarity cutoff used to estimate Precision@k/Recall@k |

## Files

| File | Role |
|---|---|
| `DocumentRetriever.py` | Loads PDFs, chunks to ≤500 words/chunk (no overlap), embeds with `all-MiniLM-L6-v2`, builds a FAISS `IndexFlatL2` index |
| `DocumentQA.py` | Retrieves top-k chunks, fits them into the generator's token budget, calls the generator, falls back to extractive-sentence selection if generation fails |
| `ResponseGenerator.py` | Wraps TinyLlama-1.1B-Chat-v1.0 for both grounded (RAG) and ungrounded (no-retrieval baseline) generation |
| `main.py` | Evaluation driver — runs both conditions over `pcos_qna.json`, computes all metrics, writes `rag_results.json` |
| `pcos_qna.json` | Gold question → reference-answer pairs used for evaluation |
| `rag_results.json` | Full per-question evaluation output from the last run (included for reproducibility without re-running inference) |
| `PAPERS.md` | Bibliography manifest of the source corpus (PDFs not redistributed here) |
| `TRACEABILITY.md` | Every reported number/table traced back to its source file and line |

## Reproducibility note

`pcos_qna.json` in this repository may contain more questions than the 30 reported in the original study (the dataset has grown since). See `TRACEABILITY.md` for exactly which 30 questions and results correspond to the published numbers.

## License

MIT — see `LICENSE`. Third-party models and libraries remain under their own licenses — see `NOTICE`. The source literature in `PAPERS.md` is not covered by this repository's license.

## Citation

See `CITATION.cff`, or click "Cite this repository" in the GitHub sidebar.
