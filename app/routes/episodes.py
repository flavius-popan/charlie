"""
Episode routes for Charlie.

Provides routes for viewing journal entries (episodes) as list and detail views.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.kuzu_service import KuzuService
from app.services.text_enrichment import get_enrichment_service

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

kuzu_service = KuzuService()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, page: int = 1, limit: int = 20):
    """
    Home page showing list of journal entries (episodes).

    Args:
        request: FastAPI request object
        page: Page number (1-indexed)
        limit: Episodes per page

    Returns:
        Rendered HTML template with episode list
    """
    try:
        offset = (page - 1) * limit
        episodes = await kuzu_service.get_episodes(limit=limit, offset=offset)
        total_episodes = await kuzu_service.count_episodes()
        total_pages = (total_episodes + limit - 1) // limit

        return templates.TemplateResponse(
            "episodes.html",
            {
                "request": request,
                "episodes": episodes,
                "page": page,
                "total_pages": total_pages,
                "total_episodes": total_episodes,
            },
        )

    except Exception as e:
        logger.error(f"Error loading home page: {e}")
        raise HTTPException(status_code=500, detail="Error loading episodes")


@router.get("/episodes/{uuid}", response_class=HTMLResponse)
async def episode_detail(request: Request, uuid: str):
    """
    Episode detail page showing full entry with entities.

    Args:
        request: FastAPI request object
        uuid: Episode UUID

    Returns:
        Rendered HTML template with episode detail
    """
    try:
        episode_with_entities = await kuzu_service.get_episode_with_entities(uuid)

        if episode_with_entities is None:
            raise HTTPException(status_code=404, detail="Episode not found")

        # Enrich episode content with markdown and entity highlighting
        enrichment_service = get_enrichment_service()
        enriched_content = enrichment_service.enrich_episode_content(
            episode_with_entities.episode.content,
            episode_with_entities.entities,
            enable_entity_highlighting=True,
        )

        # Get related entities grouped by source entity (mentioned in episode)
        mentioned_entity_uuids = {e.uuid for e in episode_with_entities.entities}
        relationships_by_entity = {}
        seen_uuids = set(mentioned_entity_uuids)

        for entity in episode_with_entities.entities:
            entity_relationships = await kuzu_service.get_entity_relationships(
                entity.uuid, limit=20
            )
            filtered_rels = []
            for rel in entity_relationships:
                related_entity = rel["entity"]
                # Only add if not already mentioned in the episode
                if related_entity.uuid not in seen_uuids:
                    # Inject HTML link into the fact text for the related entity name
                    fact_with_link = rel["relationship_fact"]
                    if fact_with_link and related_entity.name in fact_with_link:
                        fact_with_link = fact_with_link.replace(
                            related_entity.name,
                            f'<a href="/entities/{related_entity.uuid}" class="entity-link">{related_entity.name}</a>',
                            1  # Only replace first occurrence
                        )

                    filtered_rels.append({
                        "entity_name": related_entity.name,
                        "entity_uuid": related_entity.uuid,
                        "relationship_fact": fact_with_link,
                    })
                    seen_uuids.add(related_entity.uuid)

            if filtered_rels:
                relationships_by_entity[entity.uuid] = filtered_rels

        return templates.TemplateResponse(
            "episode_detail.html",
            {
                "request": request,
                "episode": episode_with_entities.episode,
                "relationships_by_entity": relationships_by_entity,
                "enriched_content": enriched_content,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading episode {uuid}: {e}")
        raise HTTPException(status_code=500, detail="Error loading episode")
