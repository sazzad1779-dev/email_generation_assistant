"""
Aggregates all sub-routers into a single top-level router.

``main.py`` imports this as::

    from src.api.routes import router as api_router
    app.include_router(api_router)

Each sub-router defines its own ``prefix`` and ``tags`` so the
aggregation is a simple ``include_router`` chain.
"""

from fastapi import APIRouter

from src.api.routes.comparison import router as comparison_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.generation import router as generation_router
from src.api.routes.health import router as health_router

router = APIRouter()
router.include_router(generation_router)
router.include_router(evaluation_router)
router.include_router(comparison_router)
router.include_router(health_router)

__all__ = ["router"]
