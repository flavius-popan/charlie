"""Entity type definitions for NER-extracted entities."""

from pydantic import BaseModel


class Person(BaseModel):
    """Individual people mentioned by name (PER from NER)."""

    pass


class Location(BaseModel):
    """Specific named locations and venues (LOC from NER)."""

    pass


class Organization(BaseModel):
    """Named teams, clubs, companies, or organizations (ORG from NER)."""

    pass


class Miscellaneous(BaseModel):
    """Named events, products, or other entities not fitting PER/LOC/ORG (MISC from NER)."""

    pass


entity_types = {
    "Person": Person,
    "Location": Location,
    "Organization": Organization,
    "Miscellaneous": Miscellaneous,
}

# NER label to entity type mapping
NER_LABEL_MAP = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Organization",
    "MISC": "Miscellaneous",
}

# Type ID mapping (for backwards compatibility with existing code)
TYPE_IDS = {
    "Person": 1,
    "Location": 2,
    "Organization": 3,
    "Miscellaneous": 4,
}


def get_type_name_from_ner_label(ner_label: str) -> str:
    """Map NER label (PER, LOC, ORG, MISC) to entity type name."""
    return NER_LABEL_MAP.get(ner_label, "Miscellaneous")


def get_type_name_from_id(type_id: int, types: dict | None = None) -> str:
    """Map entity_type_id back to type name.

    Args:
        type_id: Entity type ID (1-4 for Person, Location, Organization, Miscellaneous)
        types: Entity type dict (defaults to entity_types)

    Returns:
        Entity type name

    Raises:
        ValueError: If type_id is invalid
    """
    if types is None:
        types = entity_types

    type_names = list(types.keys())
    array_index = type_id - 1

    if 0 <= array_index < len(type_names):
        return type_names[array_index]

    raise ValueError(f"Invalid entity_type_id: {type_id}")


__all__ = [
    "Person",
    "Location",
    "Organization",
    "Miscellaneous",
    "entity_types",
    "NER_LABEL_MAP",
    "TYPE_IDS",
    "get_type_name_from_ner_label",
    "get_type_name_from_id",
]
