"""
Community routes for Charlie.

Provides routes for viewing entity communities (clusters of related entities)
and thematic browsing through community clusters.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.kuzu_service import KuzuService

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Initialize service
kuzu_service = KuzuService()


@router.get("/communities", response_class=HTMLResponse)
async def communities_list(
    request: Request,
    page: int = 1,
    per_page: int = 20,
    min_members: int = 2,
):
    """
    Community list page showing all entity clusters.

    Args:
        request: FastAPI request object
        page: Page number (1-indexed)
        per_page: Communities per page
        min_members: Minimum members to include (default 2, filters singletons)

    Returns:
        Rendered HTML template with community list
    """
    try:
        # Get communities (no built-in pagination, apply in-memory)
        all_communities = await kuzu_service.get_communities(
            min_members=min_members, limit=1000
        )

        # Calculate pagination
        total_communities = len(all_communities)
        total_pages = (total_communities + per_page - 1) // per_page
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1

        # Slice for current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        communities = all_communities[start_idx:end_idx]

        logger.info(
            f"Displaying page {page}/{total_pages} with {len(communities)} communities"
        )

        return templates.TemplateResponse(
            "communities.html",
            {
                "request": request,
                "communities": communities,
                "page": page,
                "total_pages": total_pages,
                "per_page": per_page,
                "has_prev": page > 1,
                "has_next": page < total_pages,
            },
        )

    except Exception as e:
        logger.error(f"Error loading communities list: {e}")
        raise HTTPException(status_code=500, detail="Failed to load communities")


@router.get("/communities/{uuid}", response_class=HTMLResponse)
async def community_detail(request: Request, uuid: str):
    """
    Community detail page showing members and related episodes.

    Args:
        request: FastAPI request object
        uuid: Community UUID

    Returns:
        Rendered HTML template with community detail
    """
    try:
        # Get community with members
        detail = await kuzu_service.get_community_detail(uuid)

        if detail is None:
            logger.warning(f"Community not found: {uuid}")
            raise HTTPException(status_code=404, detail="Community not found")

        # Get episodes that mention this community's entities
        episodes = await kuzu_service.get_community_episodes(uuid, limit=50)

        logger.info(
            f"Displaying community {uuid} with {detail.member_count} members and {len(episodes)} episodes"
        )

        return templates.TemplateResponse(
            "community_detail.html",
            {
                "request": request,
                "detail": detail,
                "episodes": episodes,
                "episode_count": len(episodes),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading community detail {uuid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load community detail")
