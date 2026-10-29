from DocumentRetriever import DocumentRetriever
from ResponseGenerator import ResponseGenerator
from DocumentQA import DocumentQA
import json
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk import word_tokenize
from sentence_transformers import util
from bert_score import score as bert_score
import torch
import argparse
import time
import math
import os
parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_k", type=int, default=5, help="Top K documents to retrieve")
parser.add_argument("--cosine_grounded_threshold", type=float, default=0.90, help="Minimum cosine similarity for Grounded label")
parser.add_argument("--cosine_partial_threshold", type=float, default=0.75, help="Minimum cosine similarity for Partially Hallucinated label")
parser.add_argument("--retrieval_relevance_threshold", type=float, default=0.75, help="Semantic similarity threshold used to estimate whether a chunk is relevant")
args = parser.parse_args()


def compute_f1(reference, prediction):
    ref_tokens = set(word_tokenize(reference.lower()))
    pred_tokens = set(word_tokenize(prediction.lower()))
    common = ref_tokens.intersection(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _get_text_from_chunk(chunk):
    """Safely extracts plain text from a retrieved/corpus chunk."""
    if chunk is None:
        return ""
    if isinstance(chunk, dict):
        return str(chunk.get("text", "")).strip()
    return str(chunk).strip()


def _get_all_corpus_texts(retriever):
    """
    Attempts to extract all corpus chunk texts from the retriever.

    This is needed for estimated Recall@k:
        relevant retrieved chunks / total relevant chunks in the corpus

    The function is intentionally robust because different versions of
    DocumentRetriever may store chunks as retriever.documents, retriever.chunks,
    or retriever.page_to_chunk_mapping.
    """
    candidate_attrs = ["documents", "chunks", "page_to_chunk_mapping"]
    corpus_texts = []

    for attr in candidate_attrs:
        if hasattr(retriever, attr):
            value = getattr(retriever, attr)
            if isinstance(value, list) and len(value) > 0:
                for item in value:
                    text = _get_text_from_chunk(item)
                    if text:
                        corpus_texts.append(text)
                if corpus_texts:
                    break

    # Remove obvious duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in corpus_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return unique_texts


def compute_semantic_precision_recall_at_k(reference, retrieved_chunks, retriever, embed_model, relevance_threshold=0.75):
    """
    Estimates Precision@k and Recall@k using semantic relevance between
    retrieved chunks and the reference answer.

    A chunk is treated as relevant if:
        cosine_similarity(chunk_text, reference_answer) >= relevance_threshold

    Precision@k:
        relevant retrieved chunks / retrieved chunks

    Recall@k:
        relevant retrieved chunks / estimated total relevant chunks in corpus

    Important manuscript wording:
        Precision@k and Recall@k were estimated using semantic relevance between
        retrieved chunks and reference answers, rather than manual relevance labels.
    """
    retrieved_texts = [_get_text_from_chunk(chunk) for chunk in retrieved_chunks]
    retrieved_texts = [text for text in retrieved_texts if text]

    if not retrieved_texts:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "relevant_retrieved_estimated": 0,
            "retrieved_count": 0,
            "total_relevant_corpus_estimated": 0,
            "retrieval_relevance_threshold": relevance_threshold,
            "retrieved_chunk_reference_similarities": []
        }

    with torch.no_grad():
        ref_embedding = embed_model.encode(reference, convert_to_tensor=True)
        retrieved_embeddings = embed_model.encode(retrieved_texts, convert_to_tensor=True)
        retrieved_sims = util.cos_sim(ref_embedding, retrieved_embeddings)[0]

    retrieved_sim_values = [float(x) for x in retrieved_sims]
    relevant_retrieved = sum(sim >= relevance_threshold for sim in retrieved_sim_values)
    precision_at_k = relevant_retrieved / len(retrieved_texts)

    corpus_texts = _get_all_corpus_texts(retriever)
    total_relevant_corpus = 0

    if corpus_texts:
        # Encode in batches to avoid memory spikes on larger corpora.
        batch_size = 64
        for start_idx in range(0, len(corpus_texts), batch_size):
            batch = corpus_texts[start_idx:start_idx + batch_size]
            with torch.no_grad():
                batch_embeddings = embed_model.encode(batch, convert_to_tensor=True)
                batch_sims = util.cos_sim(ref_embedding, batch_embeddings)[0]
            total_relevant_corpus += sum(float(sim) >= relevance_threshold for sim in batch_sims)

    if total_relevant_corpus > 0:
        recall_at_k = relevant_retrieved / total_relevant_corpus
        recall_at_k = min(recall_at_k, 1.0)
    else:
        # If no corpus chunk crosses the semantic threshold, recall cannot be meaningfully estimated.
        recall_at_k = 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "relevant_retrieved_estimated": int(relevant_retrieved),
        "retrieved_count": int(len(retrieved_texts)),
        "total_relevant_corpus_estimated": int(total_relevant_corpus),
        "retrieval_relevance_threshold": relevance_threshold,
        "retrieved_chunk_reference_similarities": retrieved_sim_values
    }


def assign_groundedness_label(cosine_similarity):
    """
    Groundedness based only on cosine similarity.
    """

    if cosine_similarity >= args.cosine_grounded_threshold:
        return "Grounded"

    elif cosine_similarity >= args.cosine_partial_threshold:
        return "Partially Grounded"

    else:
        return "Hallucinated"

def main():
    print("Building document index...")
    retriever = DocumentRetriever(data_dir='Data')
    generator = ResponseGenerator("models/TinyLlama-1.1B-Chat-v1.0")
    qa_system = DocumentQA(retriever, generator)
    embed_model = retriever.embeddings

    with open("pcos_qna.json", "r", encoding="utf-8") as file:
        pcos_qna = json.load(file)

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_bleu = 0.0
    total_rougeL = 0.0
    total_f1 = 0.0
    total_bert_score = 0.0
    total_cosine = 0.0
    total_precision_at_k = 0.0
    total_recall_at_k = 0.0
    total_gen_bleu = 0.0
    total_gen_rougeL = 0.0
    total_gen_f1 = 0.0
    total_gen_bertscore = 0.0
    total_gen_cosine = 0.0
    count = 0
    result_data = []

    for question, reference in pcos_qna.items():
        try:
            result = qa_system.get_answer(question, k=args.retrieval_k)
            prediction = result["answer"]
            generator_prediction = generator.generate_without_context(question)
            retrieved_chunks = result["retrieved_chunks"]

            if not prediction or prediction.lower().strip() in ["", "i don't know", "not found in the passage."]:
                prediction = "Not found in the passage."

            # BLEU-2 Score
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=smooth)

            # ROUGE-L Score
            rouge_scores = rouge.score(reference, prediction)
            rouge_l = rouge_scores['rougeL'].fmeasure

            # F1 Score
            f1 = compute_f1(reference, prediction)

            # Cosine Similarity between generated and reference answer
            with torch.no_grad():
                emb_ref = embed_model.encode(reference, convert_to_tensor=True)
                emb_pred = embed_model.encode(prediction, convert_to_tensor=True)
            cosine_similarity = float(util.cos_sim(emb_pred, emb_ref)[0])

            # Generator-only metrics

            gen_ref_tokens = word_tokenize(reference.lower())
            gen_pred_tokens = word_tokenize(generator_prediction.lower())

            gen_bleu = sentence_bleu(
                [gen_ref_tokens],
                gen_pred_tokens,
                weights=(0.5, 0.5),
                smoothing_function=smooth
            )

            gen_rouge_scores = rouge.score(
                reference,
                generator_prediction
            )

            gen_rougeL = gen_rouge_scores['rougeL'].fmeasure

            gen_f1 = compute_f1(
                reference,
                generator_prediction
            )

            with torch.no_grad():
                gen_emb_pred = embed_model.encode(
                    generator_prediction,
                    convert_to_tensor=True
                )

            gen_cosine = float(
                util.cos_sim(gen_emb_pred, emb_ref)[0]
            )
            # BERTScore
            try:
                P, R, bert_f1 = bert_score([prediction], [reference], lang="en", rescale_with_baseline=True, model_type="bert-base-uncased")
                bertscore = bert_f1[0].item()
            except Exception as e:
                print(f"BERTScore Error: {e}")
                bertscore = 0.0

            try:
                P_g, R_g, bert_f1_g = bert_score(
                    [generator_prediction],
                    [reference],
                    lang="en",
                    rescale_with_baseline=True,
                    model_type="bert-base-uncased"
                )

                gen_bertscore = bert_f1_g[0].item()

            except Exception as e:
                print(f"Generator BERTScore Error: {e}")
                gen_bertscore = 0.0

            # ===================== SEMANTIC RETRIEVAL METRICS =====================
            retrieval_metrics = compute_semantic_precision_recall_at_k(
                reference=reference,
                retrieved_chunks=retrieved_chunks,
                retriever=retriever,
                embed_model=embed_model,
                relevance_threshold=args.retrieval_relevance_threshold
            )
            precision_at_k = retrieval_metrics["precision_at_k"]
            recall_at_k = retrieval_metrics["recall_at_k"]
            # =====================================================================

            # Hybrid groundedness label
            label = assign_groundedness_label(cosine_similarity)

            # Print results
            print(f"\nQ: {question}")
            print(f"Generated Answer: {prediction}")
            print(
                f"Generator Only Answer: "
                f"{generator_prediction}"
            )
            print(f"Reference Answer: {reference}")
            print(f"BLEU-2: {bleu:.4f}, ROUGE-L: {rouge_l:.4f}, F1: {f1:.4f}, BERTScore: {bertscore:.4f}")
            print(f"Cosine Similarity: {cosine_similarity:.4f}")
            print(f"Estimated Precision@{args.retrieval_k}: {precision_at_k:.4f}")
            print(f"Estimated Recall@{args.retrieval_k}: {recall_at_k:.4f}")
            print(f"Estimated Relevant Retrieved: {retrieval_metrics['relevant_retrieved_estimated']} / {retrieval_metrics['retrieved_count']}")
            print(f"Estimated Total Relevant Corpus Chunks: {retrieval_metrics['total_relevant_corpus_estimated']}")
            print(f"HALLUCINATION LABEL (cosine): {label}")
            print(f"\n[GENERATOR ONLY] " 
            f"BLEU-2: {gen_bleu:.4f}, "
            f"ROUGE-L: {gen_rougeL:.4f}, "
            f"F1: {gen_f1:.4f}, "
            f"BERTScore: {gen_bertscore:.4f}")
            print(f"Cosine Similarity: {gen_cosine:.4f}")
            print(f"HALLUCINATION LABEL (cosine): "
            f"{assign_groundedness_label(gen_cosine)}")
            
            k_used = result["k_used"]
            result_data.append({
                "question": question,
                "generated_answer": prediction,
                "reference_answer": reference,
                "bleu_2": bleu,
                "rouge_l": rouge_l,
                "f1": f1,
                "bert_score": bertscore,
                "cosine_similarity": cosine_similarity,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "relevant_retrieved_estimated": retrieval_metrics["relevant_retrieved_estimated"],
                "retrieved_count": retrieval_metrics["retrieved_count"],
                "total_relevant_corpus_estimated": retrieval_metrics["total_relevant_corpus_estimated"],
                "retrieval_relevance_threshold": retrieval_metrics["retrieval_relevance_threshold"],
                "retrieved_chunk_reference_similarities": retrieval_metrics["retrieved_chunk_reference_similarities"],
                "hallucination_label": label,
                "retrieved_chunks": retrieved_chunks,
                "k_used": k_used,
                "generator_only_answer":generator_prediction,
                "generator_bleu_2":gen_bleu,
                "generator_rouge_l":gen_rougeL,
                "generator_f1":gen_f1,
                "generator_bert_score":gen_bertscore,
                "generator_cosine_similarity":gen_cosine,
            })

            total_bleu += bleu
            total_rougeL += rouge_l
            total_f1 += f1
            total_bert_score += bertscore
            total_cosine += cosine_similarity
            total_precision_at_k += precision_at_k
            total_recall_at_k += recall_at_k
            total_gen_bleu += gen_bleu
            total_gen_rougeL += gen_rougeL
            total_gen_f1 += gen_f1
            total_gen_bertscore += gen_bertscore
            total_gen_cosine += gen_cosine
            count += 1

        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Exception: {e}")
            continue

    # Print average scores
    print("\n===== AVERAGE METRICS =====")
    if count > 0:
        print(f"Average BLEU-2: {total_bleu / count:.4f}")
        print(f"Average ROUGE-L: {total_rougeL / count:.4f}")
        print(f"Average F1 Score: {total_f1 / count:.4f}")
        print(f"Average BERTScore: {total_bert_score / count:.4f}")
        print(f"Average Cosine Similarity: {total_cosine / count:.4f}")
        print(f"Average Estimated Precision@{args.retrieval_k}: {total_precision_at_k / count:.4f}")
        print(f"Average Estimated Recall@{args.retrieval_k}: {total_recall_at_k / count:.4f}")
        print(
            "\n===== GENERATOR ONLY "
            "BASELINE ====="
        )

        print(
            f"Average BLEU-2: "
            f"{total_gen_bleu/count:.4f}"
        )

        print(
            f"Average ROUGE-L: "
            f"{total_gen_rougeL/count:.4f}"
        )

        print(
            f"Average F1 Score: "
            f"{total_gen_f1/count:.4f}"
        )

        print(
            f"Average BERTScore: "
            f"{total_gen_bertscore/count:.4f}"
        )

        print(
            f"Average Cosine Similarity: "
            f"{total_gen_cosine/count:.4f}"
        )
    # Save all results
    with open("rag_results.json", "w", encoding="utf-8") as out_file:
        json.dump(result_data, out_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Total Time Taken: ', end - start)
