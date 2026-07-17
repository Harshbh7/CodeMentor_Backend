# Agentic RAG Implementation Tasks

## Environment Setup
- [x] Create venv
- [x] Start PostgreSQL
- [x] Create DB + user (codementor_db / codementor_user)
- [x] Install Python 3.12 via Homebrew
- [x] Recreate venv with Python 3.12
- [x] Install requirements.txt with Python 3.12

## Tools Layer (`app/tools/`)
- [x] `base.py` — Abstract tool base class
- [x] `rag_tool.py` — Knowledge base search tool
- [x] `calculator_tool.py` — Safe math evaluator
- [x] `code_analyzer_tool.py` — Code analysis via Gemini
- [x] `web_search_tool.py` — DuckDuckGo search
- [x] `__init__.py` — Tool registry

## Agent Layer (`app/agent/`)
- [x] `state.py` — AgentState TypedDict
- [x] `prompts.py` — System prompt
- [x] `nodes.py` — think_node, tool_executor_node, should_continue
- [x] `graph.py` — LangGraph StateGraph
- [x] `__init__.py` — Export run_agent()

## API + Service Layer
- [x] `app/schemas/chat.py` — ChatRequest / ChatResponse
- [x] `app/services/chat_service.py` — ChatService
- [x] `app/api/v1/chat.py` — POST /chat endpoint
- [x] `app/api/v1/router.py` — Mount chat router

## Verification
- [x] Run alembic migrations
- [x] Start uvicorn server
- [x] Test POST /api/v1/chat via Swagger UI
