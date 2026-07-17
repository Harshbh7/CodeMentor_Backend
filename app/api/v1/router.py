"""
CodeMentor AI - API v1 Router Aggregator
==========================================
Central place to mount all v1 route modules.

Design Rationale:
- Aggregating routers here keeps main.py clean.
- New modules (auth, chat, knowledge) are simply imported and included here.
- The prefix is NOT set here — it's set in main.py via settings.api_prefix.
"""

from fastapi import APIRouter

from app.api.v1 import health

# Master router for all v1 endpoints
api_router = APIRouter()

# Mount sub-routers
api_router.include_router(health.router)

from app.api.v1 import chat
api_router.include_router(chat.router)

# Phase 2+: Authentication
# from app.api.v1 import auth
# api_router.include_router(auth.router, prefix="/auth")

# Phase 4+: Knowledge Base
# from app.api.v1 import knowledge
# api_router.include_router(knowledge.router, prefix="/knowledge")
