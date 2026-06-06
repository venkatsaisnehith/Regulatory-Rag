"""
Qdrant Vector Database Interface (vector_store.py)
Manages connection, collection schema creation, document indexing,
and hybrid vector search (dense semantic search + keyword filtering).
Uses SentenceTransformers locally for vector embeddings.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any
import logging

from config import QDRANT_LOCATION, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorStore")


class RegulatoryVectorStore:
    """
    Interfaces with the Qdrant database to index and retrieve regulatory chunks.
    Utilizes local SentenceTransformer model for generating vector embeddings.
    """

    def __init__(self):
        logger.info(f"Connecting to Qdrant at location: {QDRANT_LOCATION}")
        self.client = QdrantClient(location=QDRANT_LOCATION)
        
        logger.info(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Dimensions for sentence-transformers/all-MiniLM-L6-v2 is 384
        self.vector_dim = 384
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections_response = self.client.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        
        if QDRANT_COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection '{QDRANT_COLLECTION_NAME}' (Dimensions: {self.vector_dim})")
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create payload indexes to enable fast keyword filtering (e.g. searching by specific PDF)
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="pdf_name",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        else:
            logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds text chunks and uploads them to the Qdrant collection.

        Args:
            chunks: List of parsed chunk dictionaries.
        """
        logger.info(f"Embedding and uploading {len(chunks)} chunks to Qdrant...")
        
        points = []
        for chunk in chunks:
            text = chunk["text"]
            metadata = chunk["metadata"]
            
            # Generate local dense embedding vector
            vector = self.encoder.encode(text).tolist()
            
            # Create a point object for Qdrant
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    # We store both raw text and metadata inside payload
                    payload={
                        "text": text,
                        **metadata
                    }
                )
            )
            
        # Bulk upsert into Qdrant
        self.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points
        )
        logger.info("Successfully indexed documents.")

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filter_pdf: str = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search on indexed regulations.

        Args:
            query: The user's search text.
            top_k: Number of nearest matches to return.
            filter_pdf: Optional keyword filter for a specific document.

        Returns:
            List of matching records with similarity scores and payloads.
        """
        query_vector = self.encoder.encode(query).tolist()
        
        # Build query filters (e.g. if the user wants to search ONLY a specific regulation)
        query_filter = None
        if filter_pdf:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="pdf_name",
                        match=models.MatchValue(value=filter_pdf)
                    )
                ]
            )
            
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        results = response.points
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {
                    "pdf_name": hit.payload.get("pdf_name", ""),
                    "pages": hit.payload.get("pages", []),
                    "chapter": hit.payload.get("chapter", ""),
                    "section": hit.payload.get("section", ""),
                    "article": hit.payload.get("article", ""),
                    "hierarchy_path": hit.payload.get("hierarchy_path", "")
                }
            })
            
        return formatted_results


if __name__ == "__main__":
    # Self-test stub
    db = RegulatoryVectorStore()
    test_chunks = [
        {
            "text": "Article 5: Prohibited Artificial Intelligence Practices. High risk models are forbidden.",
            "metadata": {"pdf_name": "test.pdf", "chapter": "Ch I", "section": "Sec 1", "article": "Art 5", "pages": [1]}
        }
    ]
    db.add_documents(test_chunks)
    res = db.search_semantic("Prohibited AI", top_k=1)
    print("\n--- Search Result ---")
    print(res)
