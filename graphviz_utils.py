"""Graphviz visualization utilities."""
import logging
import tempfile
import os
from graphviz import Digraph
from typing import Any
from falkordb_utils import graph


def load_written_entities(node_uuids: list[str], edge_uuids: list[str]) -> dict[str, Any]:
    """
    Query FalkorDBLite for nodes and edges by UUID.

    Args:
        node_uuids: List of node UUIDs to fetch
        edge_uuids: List of edge UUIDs to fetch

    Returns:
        dict with 'nodes' and 'edges' result sets, or 'error' key on failure
    """
    try:
        # FalkorDBLite doesn't support parameterized queries
        # Build IN clause with literal UUID values
        nodes = []
        if node_uuids:
            # Quote each UUID for IN clause
            uuid_list = ', '.join([f"'{uuid}'" for uuid in node_uuids])
            node_query = f"""
            MATCH (n:Entity)
            WHERE n.uuid IN [{uuid_list}]
            RETURN n.uuid AS uuid, n.name AS name
            """
            node_result = graph.query(node_query)
            nodes = node_result.result_set or []

        edges = []
        if edge_uuids:
            # Quote each UUID for IN clause
            uuid_list = ', '.join([f"'{uuid}'" for uuid in edge_uuids])
            edge_query = f"""
            MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
            WHERE r.uuid IN [{uuid_list}]
            RETURN r.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid, r.name AS name
            """
            edge_result = graph.query(edge_query)
            edges = edge_result.result_set or []

        return {
            "nodes": nodes,  # List[tuple]: [(uuid, name), ...]
            "edges": edges,  # List[tuple]: [(uuid, src_uuid, tgt_uuid, name), ...]
        }
    except Exception as exc:
        logging.exception("Failed to verify FalkorDBLite write")
        return {"error": str(exc)}


def render_graph_from_db(db_data: dict[str, Any]) -> str | None:
    """
    Render Graphviz graph from FalkorDB query results.

    Args:
        db_data: Dict with 'nodes' and 'edges' keys from load_written_entities()

    Returns:
        Path to PNG file, or None on error (errors logged to console)
    """
    # Check for query errors
    if "error" in db_data:
        logging.error(f"Cannot render graph: {db_data['error']}")
        return None

    try:
        dot = Digraph(format="png")

        # Graph layout settings
        dot.attr("graph",
                 rankdir="LR",
                 splines="spline",
                 pad="0.35",
                 nodesep="0.7",
                 ranksep="1.0",
                 bgcolor="transparent")

        # Node styling
        dot.attr("node",
                 shape="circle",
                 style="filled",
                 fontname="Helvetica",
                 fontsize="11",
                 color="transparent",
                 fontcolor="#1f2937")

        # Edge styling
        dot.attr("edge",
                 color="#60a5fa",
                 penwidth="1.6",
                 arrowsize="1.0",
                 arrowhead="vee",
                 fontname="Helvetica",
                 fontcolor="#e2e8f0",
                 fontsize="10")

        # Add nodes from result_set: [(uuid, name), ...]
        for row in db_data["nodes"]:
            uuid, name = row
            # Color logic: highlight "author" or "I"
            normalized = name.lower()
            is_author = normalized in {"author", "i"}
            fillcolor = "#facc15" if is_author else "#dbeafe"

            dot.node(uuid,
                    label=name,
                    fillcolor=fillcolor,
                    fontcolor="#1f2937",
                    tooltip=name)

        # Add edges from result_set: [(uuid, source_uuid, target_uuid, name), ...]
        for row in db_data["edges"]:
            edge_uuid, source_uuid, target_uuid, edge_name = row
            edge_label = edge_name.replace("_", " ")

            dot.edge(source_uuid,
                    target_uuid,
                    label=edge_label,
                    color="#60a5fa",
                    fontcolor="#e2e8f0",
                    tooltip=edge_label)

        # Render to temporary file
        fd, tmp_png_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        dot.render(tmp_png_path[:-4], format="png", cleanup=True)

        return tmp_png_path

    except Exception as exc:
        logging.exception("Failed to render Graphviz graph")
        return None
