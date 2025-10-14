"""
Routes package for Charlie.

Contains all FastAPI route handlers organized by resource type.
"""

from . import episodes, communities, entities

__all__ = ["episodes", "communities", "entities"]
