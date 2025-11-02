"""Graphviz visualization utilities.

NOTE: FalkorDB Query Results
============================

FalkorDB returns data in result.statistics, NOT result.result_set.
See falkordb_utils.py module docstring for complete explanation.

Key pattern for extracting data:
    result = graph.query("MATCH (n) RETURN n.uuid, n.name")
    for row in result.statistics:
        uuid = row[0][1]  # [type_code, value] -> extract value
        name = row[1][1]
"""
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
            RETURN n.uuid, n.name
            """
            node_result = graph.query(node_query)
            # FalkorDB returns data in result.statistics, not result_set
            # Format: [[[type_code, value], [type_code, value]], ...] where each inner list is a row
            if node_result.statistics:
                for row in node_result.statistics:
                    # Each row is [[type, uuid], [type, name]]
                    # Extract the actual values (second element of each pair)
                    uuid = row[0][1] if len(row) > 0 else None
                    name = row[1][1] if len(row) > 1 else None
                    if uuid is not None and name is not None:
                        nodes.append((uuid, name))

        edges = []
        if edge_uuids:
            # Quote each UUID for IN clause
            uuid_list = ', '.join([f"'{uuid}'" for uuid in edge_uuids])
            edge_query = f"""
            MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
            WHERE r.uuid IN [{uuid_list}]
            RETURN r.uuid, source.uuid, target.uuid, r.name
            """
            edge_result = graph.query(edge_query)
            # FalkorDB returns data in result.statistics, not result_set
            if edge_result.statistics:
                for row in edge_result.statistics:
                    # Each row is [[type, r.uuid], [type, source.uuid], [type, target.uuid], [type, r.name]]
                    edge_uuid = row[0][1] if len(row) > 0 else None
                    source_uuid = row[1][1] if len(row) > 1 else None
                    target_uuid = row[2][1] if len(row) > 2 else None
                    edge_name = row[3][1] if len(row) > 3 else None
                    if all(v is not None for v in [edge_uuid, source_uuid, target_uuid, edge_name]):
                        edges.append((edge_uuid, source_uuid, target_uuid, edge_name))

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
            # Convert to string (FalkorDB may return as int/bytes)
            uuid = str(uuid)
            name = str(name) if name else ""

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
            # Convert to strings (FalkorDB may return as int/bytes)
            edge_uuid = str(edge_uuid)
            source_uuid = str(source_uuid)
            target_uuid = str(target_uuid)
            edge_name = str(edge_name) if edge_name else ""
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
