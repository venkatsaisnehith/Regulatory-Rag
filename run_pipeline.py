"""
Pipeline Orchestration Script (run_pipeline.py)
Binds all Project 1 modules together.
1. Generates a physical sample regulatory PDF on-the-fly.
2. Parses the PDF hierarchically.
3. Indexes chunks into a local in-memory Qdrant database.
4. Simulates a query: runs vector retrieval, cross-encoder reranking, and guardrail validation.
"""

import os
import sys
import fitz  # PyMuPDF

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import SAMPLE_REG_DIR
from parser import HierarchicalPDFParser
from vector_store import RegulatoryVectorStore
from reranker import LocalReranker
from guardrails import FactualityGuardrail


def generate_sample_pdf(output_path: str) -> None:
    """Generates a structured mock regulatory PDF file for testing."""
    logger_print(f"Generating mock regulatory PDF: {output_path}")
    doc = fitz.open()
    
    # Page 1: Chapter I and Section 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "CHAPTER I - SCOPE AND DEFINITIONS", fontsize=16, color=(0.1, 0.1, 0.4))
    page1.insert_text((50, 100), "SECTION 1 - Prohibited Artificial Intelligence Practices", fontsize=12, color=(0.2, 0.2, 0.2))
    page1.insert_text((50, 130), "ARTICLE 5 - Forbidden systems in the European Union", fontsize=11, color=(0.2, 0.2, 0.2))
    
    # Body text containing specific rules
    body1 = (
        "The following AI systems are strictly prohibited inside the Union:\n"
        "1. AI systems that employ cognitive behavioral manipulation causing physical harm.\n"
        "2. Biometric categorization systems that categorize individuals based on political opinions.\n"
        "3. Emotion recognition systems used in educational institutions or workplaces.\n"
        "These practices are deemed to pose unacceptable risk and must cease by August 2026."
    )
    page1.insert_textbox((50, 160, 550, 350), body1, fontsize=10)
    
    # Page 2: Chapter II and Section 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "CHAPTER II - HIGH-RISK SYSTEMS OBLIGATIONS", fontsize=16, color=(0.1, 0.1, 0.4))
    page2.insert_text((50, 100), "SECTION 2 - Technical Documentation Requirements", fontsize=12, color=(0.2, 0.2, 0.2))
    page2.insert_text((50, 130), "ARTICLE 12 - Documentation parameters", fontsize=11, color=(0.2, 0.2, 0.2))
    
    body2 = (
        "High-risk AI systems must draw up detailed technical documentation.\n"
        "The documentation shall prove compliance with the safety rules and include:\n"
        "1. System architecture descriptions and detail logs.\n"
        "2. Risk management systems evaluations.\n"
        "3. Data quality and provenance metrics."
    )
    page2.insert_textbox((50, 160, 550, 350), body2, fontsize=10)
    
    doc.save(output_path)
    doc.close()
    logger_print("Mock regulatory PDF generated successfully.")


def logger_print(msg: str):
    print(f"[Orchestrator] {msg}")


def run_demo_pipeline():
    print("=" * 65)
    print("        REGULATORY PDF COMPLIANCE RAG PIPELINE DEMO       ")
    print("=" * 65)

    # 1. Generate Sample PDF
    pdf_path = os.path.join(SAMPLE_REG_DIR, "EU_AI_Act_Sample.pdf")
    generate_sample_pdf(pdf_path)

    # 2. Ingest & Parse
    logger_print("\n[Step 1/5] Extracting PDF layout and hierarchy...")
    parser = HierarchicalPDFParser()
    chunks = parser.parse_pdf(pdf_path)

    print("\n--- Parsed Chunks Preview ---")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} Metadata: {chunk['metadata']['hierarchy_path']}")
        print(f"Pages: {chunk['metadata']['pages']}")
        print(f"Text: {chunk['text'][:150]}...")

    # 3. Index into Qdrant Vector Store
    logger_print("\n[Step 2/5] Indexing text chunks into in-memory Qdrant DB...")
    db = RegulatoryVectorStore()
    db.add_documents(chunks)

    # 4. Query Retrieval
    query_text = "What AI systems are banned and what is the deadline?"
    logger_print(f"\n[Step 3/5] Querying database: '{query_text}'")
    retrieved_docs = db.search_semantic(query_text, top_k=2)
    
    print("\n--- Semantic Retrieval Matches (Before Reranking) ---")
    for doc in retrieved_docs:
        print(f" - [Score: {doc['score']:.4f}] {doc['metadata']['hierarchy_path']}")
        print(f"   Text: {doc['text'][:100]}...")

    # 5. Re-ranking
    logger_print("\n[Step 4/5] Executing Cross-Encoder Reranker...")
    reranker = LocalReranker()
    reranked_docs = reranker.rerank_documents(query_text, retrieved_docs, top_n=1)
    
    print("\n--- Reranked Context (Top Match) ---")
    top_doc = reranked_docs[0]
    print(f" - [Rerank Score: {top_doc['rerank_score']:.4f}] {top_doc['metadata']['hierarchy_path']}")
    print(f"   Full Chunk: {top_doc['text']}")

    # 6. Guardrails Validation Demo
    logger_print("\n[Step 5/5] Testing Factuality Guardrails...")
    guard = FactualityGuardrail()
    
    # Demo A: Valid Answer containing verified facts
    valid_answer = "Under Article 5, emotion recognition in schools and biometric categorization are prohibited. The deadline is August 2026."
    print(f"\nEvaluating Answer A (Valid): '{valid_answer}'")
    passed_a, text_a, violations_a = guard.validate_response(valid_answer, reranked_docs)
    print(f" -> Guardrail Passed: {passed_a}")
    
    # Demo B: Invalid Answer containing hallucinated facts (inventing Article 99 and year 2029)
    invalid_answer = "Article 99 prohibits behavioral systems, with a final compliance deadline set for November 2029."
    print(f"\nEvaluating Answer B (Hallucination): '{invalid_answer}'")
    passed_b, text_b, violations_b = guard.validate_response(invalid_answer, reranked_docs)
    print(f" -> Guardrail Passed: {passed_b}")
    print(f" -> Sanitized Output:\n{text_b}")

    print("\n" + "=" * 65)
    print("                      DEMO RUN COMPLETED                         ")
    print("=" * 65)
    print("\nTo launch the compliance REST API and audit client:")
    print(" 1. Run command: uvicorn project1_regulatory_rag.src.api:app --port 8001 --reload")
    print(" 2. Open file in Web Browser: project1_regulatory_rag/templates/client.html")
    print("=" * 65)


if __name__ == "__main__":
    run_demo_pipeline()
