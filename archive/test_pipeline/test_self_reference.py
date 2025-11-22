from pipeline.self_reference import SELF_ENTITY_NAME, build_provisional_self_node


def test_self_entity_name_is_I() -> None:
    assert SELF_ENTITY_NAME == "I"


def test_build_provisional_self_node_uses_I() -> None:
    node = build_provisional_self_node(group_id="test-group")
    assert node.name == "I"
