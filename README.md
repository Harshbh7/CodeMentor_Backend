# CodeMentor AI 🤖

An AI-powered coding assistant backend built with **FastAPI**, **LangGraph**, **Google Gemini (google-genai SDK)**, and **Agentic RAG**.

> A lightweight version of Cursor AI / GitHub Copilot for your portfolio.

---

## ⚡ Quick Start (Local — No Docker Required)

This is the recommended way to run the project locally. It uses a local SQLite fallback and skips Docker entirely — you only need Python.

### Prerequisites

- Python 3.11+ (check: `python3 --version`)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd "CodeMentor Ai"

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install google-genai          # New official Gemini SDK (handles thought_signatures)
```

### 3. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Open .env and set your GEMINI_API_KEY:
nano .env    # or: open .env, code .env, notepad .env
```

**Minimum required settings in `.env`:**

```env
GEMINI_API_KEY=your-real-gemini-api-key-here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
```

> 💡 **Free-tier API keys**: Use `gemini-3.1-flash-lite` as the model — it has the highest free-tier quota. `gemini-3.5-flash` and `gemini-flash-latest` are limited to 20 req/day on free keys.

### 4. Run Database Migrations

```bash
# With venv activated:
PYTHONPATH=. alembic upgrade head
```

### 5. Start the Backend & Frontend Server

Since the frontend is served statically via FastAPI, starting the backend will start the frontend as well!

**On Mac / Linux:**
```bash
# Ensure venv is activated
source venv/bin/activate
# Start the server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**On Windows:**
```powershell
# Ensure venv is activated (PowerShell)
venv\Scripts\Activate.ps1
# OR (Command Prompt)
venv\Scripts\activate.bat

# Start the server
set PYTHONPATH=.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Verify

Open your browser or `curl`:

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/health

# Ask the AI agent a question
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the time complexity of quicksort?"}' | json_pp
```

- **Frontend UI**: http://127.0.0.1:8000/ui/
- **Swagger UI**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/api/v1/health
- **API Root**: http://127.0.0.1:8000/

---

## 🐳 Full Docker Setup (Optional — enables ChromaDB + Redis)

If you have Docker Desktop installed and want vector search (RAG) to work:

```bash
# Start PostgreSQL, ChromaDB, and Redis
docker compose up postgres chromadb redis -d

# Verify they're running
docker compose ps

# Run migrations
PYTHONPATH=. alembic upgrade head

# Start the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or start everything including the API container:

```bash
docker compose up -d

# View logs
docker compose logs -f api

# Stop everything
docker compose down

# Reset (deletes all data!)
docker compose down -v
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
CodeMentor Ai/
├── app/
│   ├── api/v1/          # Route handlers
│   ├── core/            # Config, DB, Logging, Exceptions
│   ├── models/          # SQLAlchemy ORM models
│   ├── schemas/         # Pydantic request/response schemas
│   ├── repositories/    # Database access layer
│   ├── services/        # Business logic
│   ├── agent/           # LangGraph agent (state, nodes, graph, prompts)
│   ├── tools/           # Agent tools (RAG, calculator, code analyzer, web search)
│   ├── rag/             # RAG pipeline (chunker, embedder, retriever, vector store)
│   └── main.py          # FastAPI entry point
├── migrations/          # Alembic schema versions
├── tests/               # Test suite
├── .env.example         # Environment template
├── docker-compose.yml   # Local dev infrastructure
├── Dockerfile           # Production container
└── requirements.txt
```

---

## 🏗️ Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Project init, config, FastAPI, Docker, DB |
| 2 | ✅ Done | Agentic RAG — LangGraph agent + 4 tools (RAG, calculator, code analyzer, web search) |
| 3 | ⏳ Next | JWT Auth, User model, Login/Register |
| 4 | ⏳ | Knowledge Base, PDF Upload, ChromaDB population |
| 5 | ⏳ | Testing, Logging, Optimization |

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ |
| `GEMINI_MODEL` | Model name (e.g. `gemini-3.1-flash-lite`) | ✅ |
| `GEMINI_EMBEDDING_MODEL` | Embedding model (`models/gemini-embedding-2`) | ✅ |
| `POSTGRES_USER` | PostgreSQL username | ✅ |
| `POSTGRES_PASSWORD` | PostgreSQL password | ✅ |
| `POSTGRES_DB` | PostgreSQL database name | ✅ |
| `JWT_SECRET_KEY` | JWT signing secret | ✅ |
| `REDIS_HOST` | Redis hostname | Optional |
| `CHROMA_HOST` | ChromaDB hostname | Optional |

See [.env.example](.env.example) for the full list.

---

## 🤖 How the Agentic RAG Works

The agent uses a **LangGraph state machine** backed by **Google Gemini**:

```
User Query
    │
    ▼
think_node  ──── Gemini decides: use a tool or answer directly
    │
    ├─── Tool call? ──► tool_executor_node ──► (result appended to history)
    │                          │
    │                          └──────────────────────► think_node (loop)
    │
    └─── Final answer? ──► END → return response
```

**Available Tools:**
| Tool | When Used |
|------|-----------|
| `search_knowledge_base` | RAG search over programming docs / algorithms |
| `calculate` | Math expressions, Big-O numeric evaluation |
| `analyze_code` | Code review, complexity, bug detection |
| `web_search` | Real-time info (library versions, CVEs) |

> **Note:** `search_knowledge_base` requires ChromaDB running (Docker). Without it, the agent falls back to Gemini's built-in knowledge — which still works great for most CS questions.




# With venv activated:
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\Activate.ps1

# Run migrations (first time):
# Mac/Linux
PYTHONPATH=. alembic upgrade head
# Windows
set PYTHONPATH=.
alembic upgrade head

# Start the server:
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# API Test
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain binary search"}'



# Virtual environment activate karein (PowerShell)
venv\Scripts\Activate.ps1
# YA FIR (Command Prompt)
venv\Scripts\activate.bat

# Server ko start karein (Backend & Frontend dono chal jayenge)
set PYTHONPATH=.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
