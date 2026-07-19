"""
Evaluate the RAG pipeline using RAGAS metrics.

This is the piece that turns this from a "tutorial project" into a
resume-worthy engineering artifact — it gives you real numbers to
report instead of just "it works on my machine."

TODO: Build a test set of (question, ground_truth_answer) pairs based
on your actual documents. Aim for at least 15-20 pairs covering easy,
medium, and tricky questions (including ones where the answer isn't
in the docs at all, to test hallucination resistance).
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag_chain import build_rag_chain

# TODO: replace with real questions and ground-truth answers from your dataset
EVAL_SET = [
    {
        "question": "REPLACE ME",
        "ground_truth": "REPLACE ME",
    },
]


def run_evaluation(eval_set=EVAL_SET):
    chain, retriever = build_rag_chain()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in eval_set:
        question = item["question"]
        retrieved_docs = retriever.invoke(question)
        answer = chain.invoke(question)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return results


if __name__ == "__main__":
    results = run_evaluation()
    print(results)
    # TODO: paste your real results into the README once you have a
    # populated EVAL_SET
