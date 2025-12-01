"""Entity type definitions for journal entry extraction."""

from pydantic import BaseModel


class Entity(BaseModel):
    """Generic entity fallback type."""

    pass


class Person(BaseModel):
    """Person entity type."""

    pass


class Place(BaseModel):
    """Location or venue entity type."""

    pass


class Group(BaseModel):
    """Team, club, friend circle, or organization entity type."""

    pass


class Activity(BaseModel):
    """Event, occasion, outing, or routine entity type."""

    pass


entity_types = {
    "Person": Person,
    "Place": Place,
    "Group": Group,
    "Activity": Activity,
    "Entity": Entity,
}


def format_entity_types_for_llm(types: dict | None = None) -> str:
    """Convert type definitions to JSON for LLM extraction.

    Args:
        types: Entity type dict (defaults to entity_types)

    Returns:
        JSON string containing entity type definitions
    """
    import json

    if types is None:
        types = entity_types

    type_list = []

    # Explicit IDs: Entity is 0 (fallback) but listed last so LLM sees primary types first
    type_ids = {
        "Person": 1,
        "Place": 2,
        "Group": 3,
        "Activity": 4,
        "Entity": 0,
    }

    descriptions = {
        "Person": "individual people mentioned by name",
        "Place": "specific named locations and venues",
        "Group": "named teams, clubs, friend circles, or organizations",
        "Activity": "named events, occasions, outings, or recurring routines",
        "Entity": "proper nouns the author chose to mention but outside the existing categories",
    }

    for name, _ in types.items():
        type_list.append(
            {
                "entity_type_id": type_ids.get(name, 0),
                "entity_type_name": name,
                "entity_type_description": descriptions.get(
                    name, f"{name} entity type"
                ),
            }
        )

    return json.dumps(type_list)


def get_type_name_from_id(type_id: int, types: dict | None = None) -> str:
    """Map entity_type_id back to type name.

    Args:
        type_id: Entity type ID (0 = Entity, 1+ = custom types)
        types: Entity type dict (defaults to entity_types)

    Returns:
        Entity type name

    Raises:
        ValueError: If type_id is invalid
    """
    if types is None:
        types = entity_types

    if type_id == 0:
        return "Entity"

    type_names = list(types.keys())
    array_index = type_id - 1

    if 0 <= array_index < len(type_names):
        return type_names[array_index]

    raise ValueError(f"Invalid entity_type_id: {type_id}")


__all__ = [
    "Entity",
    "Person",
    "Place",
    "Group",
    "Activity",
    "entity_types",
    "format_entity_types_for_llm",
    "get_type_name_from_id",
]
