# RAG-PCOS

A Retrieval-Augmented Generation (RAG) system for question-answering on Polycystic Ovary Syndrome (PCOS), built for patient-facing, non-specialist use. Source PDFs are chunked and embedded with `all-MiniLM-L6-v2`, indexed with FAISS, and answered with a locally-run TinyLlama-1.1B-Chat-v1.0 generator, grounded in the retrieved passages.

This repository accompanies the manuscript *"Retrieval-Augmented Generation for Medical QA: A Case Study on Polycystic Ovary Syndrome"* (under review). The full paper will be linked here once published — see `CITATION.cff` for how to cite the software in the meantime.

## Repository Structure

```
RAG-PCOS/
├── README.md              this file
├── LICENSE                MIT
├── NOTICE                 third-party model/library attribution
├── CITATION.cff            software citation (paper citation added after publication)
├── .gitignore
├── requirements.txt
├── setup.py                 one-command setup (venv, deps, models)
├── main.py                 evaluation driver
├── DocumentRetriever.py     PDF ingestion, chunking, FAISS indexing
├── DocumentQA.py            retrieval + context assembly + fallback
├── ResponseGenerator.py     TinyLlama wrapper (RAG + no-retrieval baseline)
├── pcos_qna.json            30 gold question/reference-answer pairs
├── rag_results.json         30 full per-question evaluation results
├── compute_semantic_tiers.py  reproduces the High/Intermediate/Low tier distribution (Table 6)
├── PAPERS.md                bibliography manifest for the source corpus
├── TRACEABILITY.md          every reported number traced to its source
│
├── Data/                   (not included -- create and populate yourself, see below)
│   └── *.pdf
└── models/                 (not included -- download yourself, see below)
    ├── all-MiniLM-L6-v2/
    └── TinyLlama-1.1B-Chat-v1.0/
```

## Pipeline

```
Data/*.pdf --> DocumentRetriever.py --> FAISS index (all-MiniLM-L6-v2 embeddings)
                                              |
question --> DocumentQA.py (retrieve top-k, build context) --> ResponseGenerator.py (TinyLlama) --> answer
```

## Setup

One command sets up everything -- virtual environment, dependencies, and both pretrained models (see `NOTICE` for their licenses):

```bash
python setup.py
```

Safe to re-run; it skips steps that are already done. When it finishes, activate the environment:

```bash
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
```

### Build the retrieval corpus

This repository does not redistribute the source PDFs (see `PAPERS.md` for why, and for the full list of papers used in the original study). `setup.py` creates an empty `Data/` folder for you -- populate it with PDF files of PCOS research literature, ideally matching `PAPERS.md`'s titles if you want to reproduce the original study's results:

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
| `compute_semantic_tiers.py` | Reproduces the High/Intermediate/Low semantic-alignment tier distribution (0.85/0.70 cosine cutoffs) from `rag_results.json` — run `python compute_semantic_tiers.py` |
| `pcos_qna.json` | Gold question → reference-answer pairs used for evaluation |
| `rag_results.json` | Full per-question evaluation output from the last run (included for reproducibility without re-running inference) |
| `PAPERS.md` | Bibliography manifest of the source corpus (PDFs not redistributed here) |
| `TRACEABILITY.md` | Every reported number/table traced back to its source file and line |

## Reproducibility note

`pcos_qna.json` and `rag_results.json` in this repository both contain exactly the 30 questions and results reported in the published study. See `TRACEABILITY.md` for how each reported number traces back to this data.

## License

MIT — see `LICENSE`. Third-party models and libraries remain under their own licenses — see `NOTICE`. The source literature in `PAPERS.md` is not covered by this repository's license.

## Citation

See `CITATION.cff`, or click "Cite this repository" in the GitHub sidebar.
