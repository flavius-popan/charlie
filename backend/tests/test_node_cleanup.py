"""Tests for entity name cleanup module."""

import pytest

from backend.utils.node_cleanup import cleanup_entity_name, cleanup_extracted_entities


class TestVerbPhraseStripping:
    """Test structural verb+preposition pattern and direct object verbs."""

    @pytest.mark.parametrize("input_name,expected", [
        # Structural: verb-ing + preposition
        ("walking in Prospect Park", "Prospect Park"),
        ("going to the gym", "gym"),
        ("eating at Jefe's", "Jefe's"),
        ("learning about Python", "Python"),
        ("sitting on the bench", None),  # -> "bench" -> discarded (furniture)
        ("hanging out with friends", "hanging out with friends"),  # "out" not a preposition
        ("working from home", "home"),

        # Structural: verb-ed + preposition
        ("walked to the store", "store"),
        ("traveled to Paris", "Paris"),

        # Direct object verbs (no preposition)
        ("watching The Expanse", "The Expanse"),
        ("watched the new Superman", "Superman"),
        ("reading Dune", "Dune"),
        ("playing chess", "chess"),
        ("preparing dinner", "dinner"),
        ("bought groceries", "groceries"),
        ("updated MIDILisp", "MIDILisp"),
        ("planned vacation", "vacation"),

        # Base form verbs + preposition
        ("walk around the park", "park"),
        ("walk in Prospect Park", "Prospect Park"),
        ("run to the store", "store"),
        ("go to the gym", "gym"),
    ])
    def test_verb_stripping(self, input_name, expected):
        result = cleanup_entity_name(input_name)
        # Handle cases where result might be None (discarded)
        if expected is None:
            assert result is None
        else:
            assert result == expected


class TestTimePrefixStripping:
    """Test time-of-day prefix stripping."""

    @pytest.mark.parametrize("input_name,expected", [
        ("morning stroll", "stroll"),
        ("Morning stroll", "stroll"),
        ("evening yoga", "yoga"),
        ("afternoon nap", "nap"),
        ("late lunch", "lunch"),
        ("early run", "run"),
    ])
    def test_time_prefix_stripping(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestDescriptivePrefixStripping:
    """Test descriptive adjective prefix stripping."""

    @pytest.mark.parametrize("input_name,expected", [
        ("quick lunch", "lunch"),
        ("long walk", "walk"),
        ("short meeting", "meeting"),
        ("chill session", "session"),
        ("lazy Sunday", "Sunday"),
        ("sunny afternoon", "afternoon"),  # then discarded
        ("family dinner", "dinner"),
    ])
    def test_descriptive_prefix_stripping(self, input_name, expected):
        result = cleanup_entity_name(input_name)
        # "sunny afternoon" -> "afternoon" -> discarded as standalone time word
        if input_name == "sunny afternoon":
            assert result is None
        else:
            assert result == expected


class TestArticleStripping:
    """Test 'the' and 'a' stripping with proper noun preservation."""

    @pytest.mark.parametrize("input_name,expected", [
        # Lowercase "the" stripped
        ("the gym", "gym"),
        ("the park", "park"),
        ("the office", "office"),

        # Capital "The" preserved (proper names)
        ("The Beatles", "The Beatles"),
        ("The Expanse", "The Expanse"),
        ("The New York Times", "The New York Times"),

        # "the new X" stripped when X starts with capital (media title)
        ("the new Superman", "Superman"),
        ("the new Barbie", "Barbie"),
        ("the new iPhone", "new iPhone"),  # 'i' is lowercase, pattern doesn't match

        # "the new x" not stripped when x is lowercase (could be proper noun)
        ("the new york times", "new york times"),

        # "a" stripped
        ("a podcast", "podcast"),
        ("a walk", "walk"),
    ])
    def test_article_stripping(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestProperNounPreservation:
    """Test that proper nouns are preserved correctly."""

    @pytest.mark.parametrize("input_name,expected", [
        ("New York", "New York"),
        ("New Jersey", "New Jersey"),
        ("The Beatles", "The Beatles"),
        ("The Expanse", "The Expanse"),
        ("Dr. Smith", "Dr. Smith"),
        ("Auntie Rosa", "Auntie Rosa"),
        ("Jefe's", "Jefe's"),
        ("Rosie's", "Rosie's"),
    ])
    def test_proper_noun_preservation(self, input_name, expected):
        assert cleanup_entity_name(input_name) == expected


class TestDiscardPatterns:
    """Test entities that should be discarded entirely."""

    @pytest.mark.parametrize("input_name", [
        # Floor numbers
        "1st floor",
        "2nd floor",
        "3rd floor",
        "4th floor",
        "10th floor",

        # Generic rooms
        "living room",
        "bedroom",
        "bathroom",
        "dining room",
        "laundry room",
        "room",

        # Informal apartment refs
        "the apt",
        "his apt",
        "her apt",
        "my apt",

        # Single-word rooms/areas
        "balcony",
        "patio",
        "porch",
        "deck",
        "basement",
        "garage",
        "attic",

        # Furniture
        "bench",
        "couch",
        "sofa",
        "table",
        "desk",
        "chair",

        # Standalone time words
        "morning",
        "evening",
        "afternoon",
        "night",
    ])
    def test_discard_patterns(self, input_name):
        assert cleanup_entity_name(input_name) is None


class TestDiscardAfterCleanup:
    """Test entities discarded after cleanup transforms them."""

    @pytest.mark.parametrize("input_name", [
        "the living room",  # -> "living room" -> discarded
        "walking to the balcony",  # -> "balcony" -> discarded
        "morning evening",  # -> "evening" -> discarded
        "sitting on the couch",  # -> "couch" -> discarded
    ])
    def test_discard_after_cleanup(self, input_name):
        assert cleanup_entity_name(input_name) is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("input_name,expected", [
        # Too short after cleanup
        ("a", None),
        ("", None),

        # Passthrough (no changes needed)
        ("yoga", "yoga"),
        ("meditation", "meditation"),
        ("brunch", "brunch"),
        ("Kenji", "Kenji"),
        ("Prospect Park", "Prospect Park"),

        # Episode pattern
        ("2 episodes of Breaking Bad", "Breaking Bad"),
        ("3 more episodes of The Office", "The Office"),
    ])
    def test_edge_cases(self, input_name, expected):
        result = cleanup_entity_name(input_name)
        assert result == expected


class TestCleanupExtractedEntities:
    """Test batch cleanup with deduplication."""

    def test_deduplication_case_insensitive(self):
        """First occurrence wins, case-insensitive matching."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="morning walk", entity_type_id=4),
            ExtractedEntity(name="evening walk", entity_type_id=4),
            ExtractedEntity(name="Walk", entity_type_id=4),
        ]

        result = cleanup_extracted_entities(entities)

        # All become "walk", first wins
        assert len(result) == 1
        assert result[0].name == "walk"
        assert result[0].entity_type_id == 4

    def test_filters_discarded_entities(self):
        """Entities matching discard patterns are removed."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="living room", entity_type_id=2),
            ExtractedEntity(name="Prospect Park", entity_type_id=2),
            ExtractedEntity(name="balcony", entity_type_id=2),
        ]

        result = cleanup_extracted_entities(entities)

        assert len(result) == 1
        assert result[0].name == "Prospect Park"

    def test_cleans_and_dedupes_combined(self):
        """Cleanup happens before deduplication."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="the gym", entity_type_id=2),
            ExtractedEntity(name="going to the gym", entity_type_id=4),
            ExtractedEntity(name="Gym", entity_type_id=2),
        ]

        result = cleanup_extracted_entities(entities)

        # All become "gym", first wins
        assert len(result) == 1
        assert result[0].name == "gym"

    def test_preserves_entity_type(self):
        """Entity type is preserved through cleanup."""
        from backend.graph.extract_nodes import ExtractedEntity

        entities = [
            ExtractedEntity(name="morning yoga", entity_type_id=4),
            ExtractedEntity(name="the gym", entity_type_id=2),
            ExtractedEntity(name="Kenji", entity_type_id=1),
        ]

        result = cleanup_extracted_entities(entities)

        assert len(result) == 3
        # Find each by name
        yoga = next(e for e in result if e.name == "yoga")
        gym = next(e for e in result if e.name == "gym")
        kenji = next(e for e in result if e.name == "Kenji")

        assert yoga.entity_type_id == 4
        assert gym.entity_type_id == 2
        assert kenji.entity_type_id == 1
