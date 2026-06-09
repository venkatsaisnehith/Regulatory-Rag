"""
FastAPI Server Controller (api.py)
Exposes REST endpoints to:
1. Ingest/upload regulatory PDFs and index them in Qdrant.
2. Query the RAG pipeline (Retrieval, Reranking, HF LLM execution, Guardrail validation, and Audit Logging).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import requests
from typing import List, Dict, Any, Optional
import logging

from config import (
    HF_MODEL_ID, HF_API_TOKEN, SYSTEM_COMPLIANCE_PROMPT, 
    SAMPLE_REG_DIR, QDRANT_COLLECTION_NAME
)
from parser import HierarchicalPDFParser
from vector_store import RegulatoryVectorStore
from reranker import LocalReranker
from guardrails import FactualityGuardrail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIService")

app = FastAPI(
    title="Regulatory PDF Compliance RAG Agent API",
    description="Production-grade API to ingest regulations, query with GraphRAG details, and audit output lineage.",
    version="1.0.0"
)

# Enable CORS for frontend client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core modules
try:
    vector_store = RegulatoryVectorStore()
    pdf_parser = HierarchicalPDFParser()
    reranker = LocalReranker()
    guardrail = FactualityGuardrail()
except Exception as init_err:
    logger.error(f"Module initialization error: {init_err}")
    # Initialize module references to None to support lazy loading or fallback behavior if initialization fails.
    vector_store = None
    pdf_parser = None
    reranker = None
    guardrail = None


class ComplianceQuery(BaseModel):
    query: str
    filter_pdf: Optional[str] = None


@app.get("/health", summary="Check service and model status")
def health_check():
    return {
        "status": "active",
        "huggingface_model": HF_MODEL_ID,
        "token_loaded": len(HF_API_TOKEN) > 0,
        "modules_loaded": all(x is not None for x in [vector_store, pdf_parser, reranker, guardrail])
    }


@app.post("/upload", summary="Upload a regulatory PDF and index it in the Vector Store")
def upload_regulation(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save file to sample regulations folder
    file_path = os.path.join(SAMPLE_REG_DIR, file.filename)
    logger.info(f"Saving uploaded file to: {file_path}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse and index
    try:
        chunks = pdf_parser.parse_pdf(file_path)
        if vector_store:
            vector_store.add_documents(chunks)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_count": len(chunks),
            "message": "File parsed and indexed successfully."
        }
    except Exception as e:
        # Clean up file in case of failure
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", summary="Query the RAG pipeline with compliance auditing")
def query_compliance(payload: ComplianceQuery):
    if not all(x is not None for x in [vector_store, reranker, guardrail]):
        raise HTTPException(status_code=503, detail="Models or database connections are currently offline.")

    # 1. Semantic Retrieval (fetch top 10 potential candidates)
    retrieved = vector_store.search_semantic(
        query=payload.query,
        top_k=10,
        filter_pdf=payload.filter_pdf
    )

    if not retrieved:
        return {
            "answer": "No relevant regulation documents found in the database. Please upload a PDF first.",
            "guardrail_status": "skipped",
            "audit_trail": []
        }

    # 2. Local Cross-Encoder Reranking (compress to top 3 contextually accurate blocks)
    reranked_contexts = reranker.rerank_documents(
        query=payload.query,
        retrieved_chunks=retrieved,
        top_n=3
    )

    # 3. Construct Prompt with context
    context_str = "\n\n".join([
        f"[Source: {c['metadata']['hierarchy_path']}, Pages: {c['metadata']['pages']}]\nContext: {c['text']}"
        for c in reranked_contexts
    ])
    
    full_prompt = (
        f"{SYSTEM_COMPLIANCE_PROMPT}\n\n"
        f"--- Verified Regulatory Context ---\n{context_str}\n\n"
        f"Compliance Query: {payload.query}\n"
        f"Audit-Trail Answer:"
    )

    # 4. Generate LLM response using Hugging Face Serverless API
    raw_answer = ""
    if not HF_API_TOKEN:
        # Fallback response strategy if Hugging Face API token is not configured
        raw_answer = (
            "Offline Simulation Mode: Hugging Face API token is not configured. "
            "Context retrieval and rerank validation was executed successfully.\n"
            f"Primary Reference: {reranked_contexts[0]['metadata']['hierarchy_path']}"
        )
    else:
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        try:
            logger.info(f"Querying Hugging Face Serverless API for model: {HF_MODEL_ID}...")
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "inputs": full_prompt,
                    "parameters": {"max_new_tokens": 512, "temperature": 0.1, "return_full_text": False}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                res_json = response.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    raw_answer = res_json[0].get("generated_text", "")
                    
                    # Sometimes HF returns full text despite parameters, let's clean it if needed
                    if "Audit-Trail Answer:" in raw_answer:
                        raw_answer = raw_answer.split("Audit-Trail Answer:")[-1].strip()
                else:
                    raw_answer = str(res_json)
            else:
                logger.error(f"Hugging Face API returned error status: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail=f"Hugging Face endpoint error: {response.text}")
                
        except Exception as api_err:
            logger.error(f"Failed to query Hugging Face API: {api_err}")
            # Safe local fallback using top context chunk
            raw_answer = f"Based on {reranked_contexts[0]['metadata']['hierarchy_path']}: {reranked_contexts[0]['text'][:300]}..."

    # 5. Run Factuality Guardrails
    guard_passed, final_answer, violations = guardrail.validate_response(
        llm_response=raw_answer.strip(),
        source_chunks=reranked_contexts
    )

    # 6. Return response + audit trail
    return {
        "answer": final_answer,
        "guardrail_status": "passed" if guard_passed else "failed",
        "violations": violations,
        "audit_trail": reranked_contexts
    }


@app.get("/documents", summary="List all uploaded PDFs in the data folder")
def list_documents():
    if not os.path.exists(SAMPLE_REG_DIR):
        return []
    files = [f for f in os.listdir(SAMPLE_REG_DIR) if f.lower().endswith(".pdf")]
    return files


class AuditQuery(BaseModel):
    query_task: str


@app.post("/audit", summary="Run ReAct compliance audit agent")
def run_compliance_audit(payload: AuditQuery):
    if not all(x is not None for x in [vector_store, reranker]):
        raise HTTPException(status_code=503, detail="Models or database connections are currently offline.")
    
    try:
        from agent import ComplianceTools, AuditorAgent
        # Instantiate agent using the global active components
        tools = ComplianceTools(vector_store, reranker)
        agent = AuditorAgent(tools)
        
        result = agent.run_audit(payload.query_task)
        return result
    except Exception as e:
        logger.error(f"Compliance audit agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance audit execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
