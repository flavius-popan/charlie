from pipeline import entity_edge_models as schemas


def test_person_schema_is_string_only():
    person_fields = set(schemas.Person.model_fields.keys())
    assert person_fields == {"relationship_type"}, (
        "Person should only expose relationship_type to avoid extra parsing, "
        f"found {person_fields}"
    )


def test_activity_schema_includes_purpose_string():
    activity_fields = schemas.Activity.model_fields
    assert set(activity_fields.keys()) == {"purpose"}, (
        "Activity must keep a single descriptive field named 'purpose'. "
        f"Found fields: {set(activity_fields.keys())}"
    )
    assert activity_fields["purpose"].annotation is str | None


def test_edges_are_label_only():
    assert schemas.edge_types == {}, "Edge schema map should be empty for label-only edges."
