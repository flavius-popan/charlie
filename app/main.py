"""
Charlie Interactive Graph Explorer - FastAPI Application

Main application entry point that sets up FastAPI with routes,
static files, and template rendering.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import episodes, communities, entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Charlie starting up...")
    logger.info("Server ready at http://localhost:8080")
    yield
    logger.info("Charlie shutting down...")
    from app.routes.episodes import kuzu_service

    kuzu_service.close()
    logger.info("Cleanup complete")


app = FastAPI(
    title="Charlie - Interactive Graph Explorer",
    description="A journaling tool with knowledge graph visualization",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(episodes.router, tags=["episodes"])
app.include_router(communities.router, tags=["communities"])
app.include_router(entities.router, tags=["entities"])

logger.info("Charlie application initialized")
