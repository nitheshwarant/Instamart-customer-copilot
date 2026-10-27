
"""
Part D - Policy RAG Assistant
Instamart Customer Intelligence & Support Copilot

Tasks:
1. Chunk five policy documents
2. Vectorize using TF-IDF
3. Retrieve using cosine similarity
4. Answer through the offline mock LLM
5. Evaluate retrieval and answers
6. Compare overlapping and non-overlapping chunks
"""

from pathlib import Path
import sys

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Import mock_llm.py from the project root
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

import mock_llm


POLICY_DIR = Path("data/policies")
EVAL_FILE = Path("data/raw/policy_eval.csv")


# ==========================================
# STEP 1: LOAD AND CHUNK POLICY DOCUMENTS
# ==========================================

class PolicyRAG:

    def __init__(self, overlap=10):

        if not (0 <= overlap < 40):
            raise ValueError(
                "Overlap must be between 0 and 39"
            )

        self.overlap = overlap
        self.chunks = []

        # Load the five policy documents
        for path in sorted(POLICY_DIR.glob("*.txt")):

            words = path.read_text(
                encoding="utf-8"
            ).split()

            # 40 words per chunk
            # 10-word overlap means step = 30
            for start in range(
                0, len(words), 40 - overlap
            ):

                part = words[start:start + 40]

                if not part:
                    break

                self.chunks.append({
                    "source": path.name,
                    "text": " ".join(part)
                })

                if start + 40 >= len(words):
                    break

        # ==================================
        # STEP 2: CREATE TF-IDF VECTORS
        # ==================================

        self.vectorizer = TfidfVectorizer()

        self.chunk_vectors = (
            self.vectorizer.fit_transform(
                chunk["text"]
                for chunk in self.chunks
            )
        )

    # ======================================
    # STEP 3: RETRIEVE RELEVANT CHUNKS
    # ======================================

    def retrieve(self, query, k=3):

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.chunk_vectors
        )[0]

        # Highest similarity comes first
        indices = sorted(
            range(len(similarities)),
            key=lambda i: -similarities[i]
        )[:k]

        results = []

        for i in indices:

            results.append({
                **self.chunks[i],
                "score": float(similarities[i])
            })

        return results

    # ======================================
    # STEP 4: ANSWER POLICY QUESTIONS
    # ======================================

    def answer(self, question):

        results = self.retrieve(question, k=3)

        # Include source labels in the context
        context = "\n".join(
            f"[{item['source']}] {item['text']}"
            for item in results
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT. Answer using only "
                    "the supplied context. If the policy "
                    "does not contain the answer, say "
                    "I could not find that information "
                    "in the policy documents."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n"
                    f"QUESTION:\n{question}"
                )
            }
        ]

        response = mock_llm.chat(messages)

        answer_text = (
            response.choices[0].message.content
        )

        sources = [
            item["source"] for item in results
        ]

        return answer_text, sources


# ==========================================
# STEP 5: EVALUATE THE RAG ASSISTANT
# ==========================================

def evaluate(rag):

    questions = pd.read_csv(
        EVAL_FILE,
        keep_default_na=False
    )

    document_hits = 0
    fact_hits = 0
    fallback_hits = 0
    failed_questions = []

    for row in questions.itertuples(index=False):

        answer, sources = rag.answer(row.question)

        # Questions not covered by policies
        if row.expected_doc == "NONE":

            if answer.startswith("I could not find"):
                fallback_hits += 1

        else:

            # Check if expected source is retrieved
            if row.expected_doc in sources:
                document_hits += 1

            # Check exact expected fact in answer
            if row.expected_fact in answer:
                fact_hits += 1

            else:
                failed_questions.append({
                    "question": row.question,
                    "answer": answer,
                    "sources": sources
                })

    return {
        "retrieval": document_hits,
        "facts": fact_hits,
        "fallback": fallback_hits,
        "failed": failed_questions
    }


# ==========================================
# STEP 6: RUN EXPERIMENTS
# ==========================================

def main():

    # Compare 10-word overlap against zero overlap
    for overlap in [10, 0]:

        rag = PolicyRAG(overlap=overlap)

        print(
            f"\n===== OVERLAP {overlap} ====="
        )

        print("Total chunks:", len(rag.chunks))

        counts = pd.Series([
            chunk["source"]
            for chunk in rag.chunks
        ]).value_counts().sort_index()

        print("\nChunks per document:")
        print(counts.to_string())

        # Test retrieval
        results = rag.retrieve(
            "refund policy late delivery"
        )

        print("\nTop 3 retrieved documents:")

        for item in results:
            print(
                item["source"],
                round(item["score"], 3)
            )

        # Evaluate all 14 questions
        stats = evaluate(rag)

        print("\n===== EVALUATION =====")

        print(
            f"Correct retrieval: "
            f"{stats['retrieval']}/12"
        )

        print(
            f"Expected facts found: "
            f"{stats['facts']}/12"
        )

        print(
            f"Correct fallbacks: "
            f"{stats['fallback']}/2"
        )

        for item in stats["failed"]:
            print(
                "Failed question:",
                item["question"]
            )

    print("\nPOLICY RAG EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
