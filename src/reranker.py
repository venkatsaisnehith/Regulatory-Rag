"""
Local Reranker Module (reranker.py)
Implements a Cross-Encoder reranking model locally using Hugging Face's sentence-transformers.
Re-orders retrieved vector search chunks to prioritize direct legal relevance.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import logging

from config import RERANK_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reranker")


class LocalReranker:
    """
    Loads a local Cross-Encoder model to re-score and re-rank document chunks
    against the query.
    """

    def __init__(self):
        logger.info(f"Loading local Cross-Encoder model: {RERANK_MODEL_NAME}...")
        # BAAI/bge-reranker-base is highly accurate and runs fast on CPU
        try:
            self.model = CrossEncoder(RERANK_MODEL_NAME)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {e}")
            raise e

    def rerank_documents(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-scores and re-ranks retrieved chunks based on their contextual relationship to the query.

        Args:
            query: The user's original compliance query.
            retrieved_chunks: List of retrieved chunks from the vector store.
            top_n: Number of final chunks to return.

        Returns:
            Sorted list of chunks containing their new cross-encoder scores.
        """
        if not retrieved_chunks:
            return []

        logger.info(f"Reranking {len(retrieved_chunks)} document chunks using Cross-Encoder...")

        # Form pairs of [query, chunk_text] for the Cross-Encoder model
        pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
        
        # Predict scores (higher = more contextually relevant)
        scores = self.model.predict(pairs)

        # Attach scores to the chunks
        for i, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[i])

        # Sort chunks by rerank score in descending order
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        logger.info("Successfully completed reranking.")
        return sorted_chunks[:top_n]


if __name__ == "__main__":
    # Self-test stub
    reranker = LocalReranker()
    test_query = "What AI practices are prohibited?"
    test_docs = [
        {"text": "Under Article 5 of the EU AI Act, cognitive behavioral manipulation is a prohibited practice.", "metadata": {}},
        {"text": "Article 12 discusses the technical documentation requirements for high-risk systems.", "metadata": {}}
    ]
    res = reranker.rerank_documents(test_query, test_docs, top_n=2)
    print("\n--- Reranked Results (Highest Score First) ---")
    for doc in res:
        print(f"Score: {doc['rerank_score']:.4f} | Text: {doc['text']}")
