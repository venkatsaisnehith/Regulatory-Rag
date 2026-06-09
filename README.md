# Autonomous Regulatory RAG & Compliance Auditor Agent

## The Problem

Luxembourg is Europe's largest and the world's second-largest fund domicile, administering over **€5.8 trillion** in fund assets. Fund managers, transfer agents, and depositary banks operate under intense regulatory scrutiny from the CSSF. Ensuring strict compliance with circulars (such as CSSF Circular 18/698 governing delegation and internal governance) and emerging frameworks (like the EU AI Act) requires reviewing thousands of pages of legal text.

Traditional Large Language Models (LLMs) fail in these high-stakes compliance environments:
1. **Hallucinations**: Generative models easily make up article numbers, dates, or quantitative limits.
2. **Context Loss**: Standard character-based text splitters break sentences mid-paragraph, separating a regulatory constraint from the specific article number or scope that governs it.
3. **Data Sovereignty**: Uploading proprietary fund structures or draft regulations to public LLM endpoints often violates strict data privacy laws (e.g., GDPR or CSSF Article 41-1).

This system solves these issues by combining layout-aware parsing, local vector indexing, semantic cross-encoder reranking, mathematical factuality check guardrails, and a local ReAct compliance auditing loop.

---

## What This Project Does

This is an **enterprise-grade compliance auditing pipeline** and autonomous agent that:

1. **Parses PDF regulations layout-aware**: Breaks down documents structurally (Chapter > Section > Article) using visual font cues (size, weight) and falls back on maximum token safety bounds, preserving legal context and hierarchy.
2. **Indexes and retrieves via a two-stage pipeline**:
   - *Stage 1 (Semantic Search)*: Indexes text chunks into a local Qdrant vector database using dense embeddings (`all-MiniLM-L6-v2`) to gather the top 10 candidates.
   - *Stage 2 (Reranking)*: Re-scores candidate matches at the token level using a local Cross-Encoder (`bge-reranker-base`) to identify the top 3 contextually accurate blocks.
3. **Applies deterministic factuality guardrails**: Extracts all numbers, percentages, and article citations from the LLM's response and verifies them mathematically against the source chunks, blocking hallucinations.
4. **Executes compliance audits autonomously**: Implements a Reasoning + Action (ReAct) loop enabling a compliance agent to make thoughts, query regulations, fetch simulated operational metrics, and log breach alerts.
5. **Presents a premium visual audit journal**: Built with a clean, high-contrast Light Mode theme (alabaster backgrounds, midnight blue accents, crisp typography) optimized for clarity on executive dashboards.

---

## Technical Architecture

### Layout-Aware Parser (`parser.py`)
Extracts text hierarchically from regulatory PDFs:
- **Heading extraction** uses PyMuPDF (`fitz`) to examine visual block parameters (font size and bold styling) to isolate Chapters, Sections, and Articles.
- **Safety size cutoff** automatically splits large blocks when they exceed 1,500 characters, ensuring they remain within the vector embedding model's context limits.
- **Lineage mapping** binds every parsed text chunk to its absolute path (e.g., `CSSF_Circular_18_698.pdf > CHAPTER II > Article 5`) and exact source page numbers.

### Vector Store & Hybrid Search (`vector_store.py`)
Handles indexing and retrieval using Qdrant:
- **Local dense embeddings** are generated via a SentenceTransformer (`all-MiniLM-L6-v2`) to produce 384-dimensional vectors.
- **Keyword filtering** is implemented on top of the vector search, allowing compliance officers to narrow their scope to a single selected regulation PDF.
- **Zero-infrastructure RAM database** runs in-memory (`:memory:`), keeping installation quick and portable.

### Cross-Encoder Reranker (`reranker.py`)
Solves the semantic drift problem common in vector search:
- Loads `BAAI/bge-reranker-base` locally via Hugging Face.
- Computes token-level attention scores between the query and retrieved candidates.
- Compresses the initial 10 vector candidates down to the top 3 most precise matches, filtering out false positives before LLM processing.

### Factuality Guardrail (`guardrails.py`)
A verification firewall that shields against model errors:
- Intercepts the generated LLM response before displaying it to the user.
- Extracts dates, numbers, percentages, and article references using custom regex patterns.
- Runs set-based mathematical comparisons against the source context. If the response claims a number or clause that does not exist in the source document, the guardrail blocks it and reports a validation violation.

### Autonomous Auditor Agent (`agent.py`)
Orchestrates active compliance checks:
- Standardizes reasoning with a Thought-Action-Observation loop (ReAct model).
- Equips the LLM with active compliance tools:
  - `query_regulations`: Runs the semantic RAG pipeline to locate laws.
  - `get_fund_metrics`: Fetches portfolio parameters (e.g., AUM, cash ratios) from the active fund ledger.
  - `submit_compliance_alert`: Logs official compliance warnings if a breach is detected.
- Includes local simulation fallbacks to run offline when Hugging Face connections are unavailable.

### API Controller Service (`api.py`)
FastAPI REST service exposing core endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Upload a PDF regulation, parse it layout-aware, and index it in Qdrant. |
| `/query` | POST | Run the full RAG pipeline (search, rerank, prompt, validate guardrails). |
| `/audit` | POST | Trigger the ReAct compliance auditor agent on a specific fund task. |
| `/documents` | GET | List all active uploaded regulatory documents. |
| `/health` | GET | Check API server status, module loads, and token authorization. |

### Web Interface Ledger (`templates/client.html`)
- Premium light-mode visual ledger designed for risk teams.
- Left-aligned PDF upload box and operational mode toggles (Standard RAG vs Auditor Agent).
- Interactive audit timeline showing agent thoughts, actions, and observations in real-time.
- **Oversight Lineage Sidebar** that lets compliance officers click any answer to view the exact text chunks and source pages retrieved from the vector store.

---

## Setup & Local Execution

### Prerequisites
- Python 3.8+
- pip
- A free [Hugging Face](https://huggingface.co/) Access Token (Read permission)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/venkatsaisnehith/Regulatory-Rag.git
cd Regulatory-Rag

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Hugging Face Token
# Create a .env file in the root directory and add your token:
echo HUGGINGFACEHUB_API_TOKEN=your_token_here > .env

# 5. Run the offline orchestrator pipeline demo
# This generates a mock regulatory PDF, indexes it, and validates the guardrail
python run_pipeline.py

# 6. Start the API server
cd src
..\venv\Scripts\python -m uvicorn api:app --port 8001 --reload

# 7. Open the client interface
# Open templates/client.html directly in your web browser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Parsing** | PyMuPDF (`fitz`), Regular Expressions |
| **Vector DB** | Qdrant Client (in-memory) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Reranking** | Hugging Face Cross-Encoder (`BAAI/bge-reranker-base`) |
| **LLM Generation** | Hugging Face Serverless API (`Qwen/Qwen2.5-7B-Instruct`) |
| **REST Server** | FastAPI, Uvicorn, Pydantic, Requests |
| **Frontend UI** | Semantic HTML5, Vanilla CSS3 (Light Mode), Vanilla JavaScript |
