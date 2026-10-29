#!/usr/bin/env python3
"""
Reproduces Table 6 (Semantic Alignment Distribution) and the per-question
tier column (H/I/L) in Table A1, from rag_results.json.

This script did not exist anywhere in the repository before -- main.py's
only labeling function, assign_groundedness_label(), uses different
thresholds (0.90/0.75) and different label names ("Grounded"/"Partially
Grounded"/"Hallucinated"). This script implements the actual 0.85/0.70
"High/Intermediate/Low Semantic Alignment" scheme used in the manuscript.

Usage:
    python compute_semantic_tiers.py
"""
import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_results.json")

HIGH_THRESHOLD = 0.85
INTERMEDIATE_THRESHOLD = 0.70


def tier(cosine_similarity):
    if cosine_similarity >= HIGH_THRESHOLD:
        return "High"
    elif cosine_similarity >= INTERMEDIATE_THRESHOLD:
        return "Intermediate"
    else:
        return "Low"


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    n = len(results)
    counts = {"High": 0, "Intermediate": 0, "Low": 0}

    print(f"{'Question':<70s} {'Cosine':>8s}  Tier")
    print("-" * 92)
    for row in results:
        t = tier(row["cosine_similarity"])
        counts[t] += 1
        q = row["question"][:68]
        print(f"{q:<70s} {row['cosine_similarity']:8.4f}  {t}")

    print()
    print(f"Semantic Alignment Distribution (N={n}):")
    for t in ["High", "Intermediate", "Low"]:
        pct = counts[t] / n * 100
        print(f"  {t:<14s}: {counts[t]} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
