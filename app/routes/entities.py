"""
Entity routes for Charlie.

Provides routes for viewing entity details.
"""

import logging

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.kuzu_service import KuzuService

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

kuzu_service = KuzuService()


@router.get("/entities/{uuid}", response_class=HTMLResponse)
async def entity_detail(request: Request, uuid: str):
    """
    Entity detail page.

    Args:
        request: FastAPI request object
        uuid: Entity UUID

    Returns:
        Rendered HTML template with entity details
    """
    try:
        entity = await kuzu_service.get_entity_by_uuid(uuid)

        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Fetch timeline, relationships, and communities
        timeline = await kuzu_service.get_entity_timeline(uuid)
        relationships = await kuzu_service.get_entity_relationships(uuid)
        communities = await kuzu_service.get_entity_communities(uuid)

        return templates.TemplateResponse(
            "entity_detail.html",
            {
                "request": request,
                "entity": entity,
                "timeline": timeline,
                "relationships": relationships,
                "communities": communities,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading entity {uuid}: {e}")
        raise HTTPException(status_code=500, detail="Error loading entity")
