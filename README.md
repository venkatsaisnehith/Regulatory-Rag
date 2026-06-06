# Autonomous Regulatory PDF Compliance Auditor & RAG Agent

An enterprise-grade, layout-aware Retrieval-Augmented Generation (RAG) pipeline and autonomous Auditor Agent designed to parse, index, search, and validate financial regulatory documents (such as Luxembourg CSSF circulars and the EU AI Act) without breaching data sovereignty regulations (GDPR / CSSF Art 41-1).

---

## 🚀 KEY FEATURES

*   **Hierarchical Document Parsing:** An offline PyMuPDF parser that extracts raw PDF text by identifying structural headings (`Chapter > Section > Article`) using visual layout attributes (font weight, size, bolding). This preserves legal context instead of creating arbitrary character-based chunks.
*   **Two-Stage Semantic Retrieval:**
    1.  *First Stage (Vector Search):* Queries a local in-memory Qdrant database using dense embeddings (`all-MiniLM-L6-v2`) to retrieve the top 10 candidate chunks.
    2.  *Second Stage (Reranking):* Evaluates candidates with a local Cross-Encoder re-ranker (`BAAI/bge-reranker-base`) at token level, sorting the top 3 contextually accurate blocks.
*   **Deterministic Factuality Guardrails:** Intercepts LLM answers using regular expressions, extracting numbers, percentages, and article numbers. It runs mathematical set subtraction against the source chunks to detect and block unverified claims or hallucinations.
*   **Autonomous Compliance Agent:** Implements a Reasoning + Action (ReAct) execution loop. The agent decodes query tasks, dynamically executes tools (`query_regulations`, `get_fund_metrics`, `submit_compliance_alert`), and logs breach alerts when fund metrics violate regulatory limits.
*   **Glassmorphism Analyst Dashboard:** A modern dark-mode interface built with HTML5, CSS3, and JavaScript, displaying indexed documents, chat interfaces, live agent thought-action logging, and oversight lineage sidebars.

---

## 📁 SYSTEM ARCHITECTURE

```
project1_regulatory_rag/
├── data/
│   └── sample_regulations/    # Store target regulatory PDFs
├── src/
│   ├── __init__.py
│   ├── api.py                 # FastAPI service endpoints (/upload, /query, /audit)
│   ├── parser.py              # PyMuPDF hierarchical parsing logic
│   ├── vector_store.py        # Qdrant client, schemas, & embedding generation
│   ├── reranker.py            # Local Hugging Face Cross-Encoder re-ranking
│   ├── guardrails.py          # Factuality verification firewall
│   ├── agent.py               # ReAct Auditor Agent & tool calling declarations
│   └── config.py              # Environment configurations & model labels
├── templates/
│   └── client.html            # Asynchronous frontend UI dashboard
├── run_pipeline.py            # Offline validation & orchestrator test run
└── requirements.txt           # Project dependencies
```

---

## 🛠️ INSTALLATION & LOCAL SETUP

### Prerequisites
*   Python 3.8 or higher
*   PIP (Python Package Installer)

### Step-by-Step Execution
1.  **Navigate into the directory & create a Virtual Environment:**
    ```powershell
    cd project1_regulatory_rag
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Run the offline orchestrator pipeline:**
    This script generates a mock EU AI Act PDF, parses it hierarchically, indexes it, queries it, and demonstrates the guardrail blocking a hallucinated answer.
    ```powershell
    python run_pipeline.py
    ```
4.  **Launch the live REST API:**
    ```powershell
    cd src
    uvicorn api:app --port 8001 --reload
    ```
5.  **Open the Web Dashboard:**
    Open [client.html](file:///c:/Users/saisn/Documents/finance%20project/project1_regulatory_rag/templates/client.html) directly in any browser to query circulars and run agent compliance audits!
