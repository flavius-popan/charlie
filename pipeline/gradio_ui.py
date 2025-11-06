"""Gradio UI for testing ExtractNodes module.

Provides interactive testing interface for entity extraction and resolution.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import gradio as gr
import dspy
from dspy_outlines import OutlinesAdapter, OutlinesLM

from pipeline import add_journal
from pipeline.db_utils import (
    fetch_entities_by_group,
    fetch_recent_episodes,
    get_db_stats,
    reset_database,
    write_episode_and_nodes,
)
from settings import DB_PATH, GROUP_ID, MODEL_CONFIG, DEFAULT_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing ExtractNodes UI")
logger.info("Database: %s", DB_PATH)
logger.info("Model: %s", DEFAULT_MODEL_PATH)
logger.info("Model config: %s", MODEL_CONFIG)

lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)

logger.info("DSPy configured with OutlinesLM")


def _parse_reference_time(value: str | None) -> datetime:
    """Parse reference time from ISO8601 string, default to now."""
    if not value or not value.strip():
        return datetime.now()
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        logger.warning("Invalid reference time: %s, using now()", value)
        return datetime.now()


def _format_entity_list(nodes) -> str:
    """Format entity nodes as readable list with custom attributes.

    Note: Attributes will be populated once Stage 3 (extract_attributes) is implemented.
    Currently, Stage 1 extracts only entity names and types, following graphiti-core's pattern.
    """
    if not nodes:
        return "(no entities extracted)"

    lines = []
    for node in nodes:
        labels = ", ".join(node.labels) if node.labels else "Entity"
        base_info = f"- {node.name} [{labels}]"

        if node.attributes:
            attrs = []
            if "Person" in node.labels:
                if relationship := node.attributes.get("relationship_type"):
                    attrs.append(relationship)
            elif "Emotion" in node.labels:
                if emotion := node.attributes.get("specific_emotion"):
                    attrs.append(emotion)
                if category := node.attributes.get("category"):
                    attrs.append(category)

            if attrs:
                base_info += f" ({', '.join(attrs)})"

        base_info += f" (UUID: {node.uuid[:8]}...)"
        lines.append(base_info)

    return "\n".join(lines)


def _format_stats(metadata: dict) -> str:
    """Format extraction statistics."""
    return f"""Extracted: {metadata.get('extracted_count', 0)} entities
Resolved: {metadata.get('resolved_count', 0)} entities
Exact matches: {metadata.get('exact_matches', 0)}
Fuzzy matches: {metadata.get('fuzzy_matches', 0)}
New entities: {metadata.get('new_entities', 0)}"""


def _format_uuid_map(uuid_map: dict, nodes) -> str:
    """Format UUID mapping table."""
    if not uuid_map:
        return "(no UUID mappings)"

    lines = ["Provisional UUID → Resolved UUID"]
    lines.append("-" * 50)

    node_names = {node.uuid: node.name for node in nodes}

    for prov_uuid, res_uuid in uuid_map.items():
        name = node_names.get(res_uuid, "Unknown")
        status = "NEW" if prov_uuid == res_uuid else "MATCHED"
        lines.append(f"{prov_uuid[:8]}... → {res_uuid[:8]}... [{status}] ({name})")

    return "\n".join(lines)


def _format_edge_list(edges) -> str:
    """Format entity edges as readable list."""
    if not edges:
        return "(no edges extracted)"

    lines = []
    for edge in edges:
        source_short = edge.source_node_uuid[:8]
        target_short = edge.target_node_uuid[:8]
        fact_preview = edge.fact[:150] + "..." if len(edge.fact) > 150 else edge.fact

        valid_str = edge.valid_at.isoformat() if edge.valid_at else "N/A"
        invalid_str = edge.invalid_at.isoformat() if edge.invalid_at else "N/A"

        lines.append(
            f"- {source_short}... → {target_short}... [{edge.name}]\n"
            f"  Fact: {fact_preview}\n"
            f"  Valid: {valid_str} | Invalid: {invalid_str}"
        )

    return "\n\n".join(lines)


def _format_edge_stats(metadata: dict) -> str:
    """Format edge extraction statistics."""
    edges_meta = metadata.get("edges", {})
    return f"""Extracted: {edges_meta.get('extracted_count', 0)} relationships
Built: {edges_meta.get('built_count', 0)} edges
Resolved: {edges_meta.get('resolved_count', 0)} edges
New: {edges_meta.get('new_count', 0)}
Merged: {edges_meta.get('merged_count', 0)}"""


def _format_episode(episode) -> str:
    """Format episode details."""
    if not episode:
        return "(no episode created)"

    content_preview = episode.content[:200] + "..." if len(episode.content) > 200 else episode.content

    return f"""UUID: {episode.uuid}
Name: {episode.name}
Group ID: {episode.group_id}
Valid At: {episode.valid_at.isoformat()}
Created At: {episode.created_at.isoformat()}
Source: {episode.source.value}
Content: {content_preview}"""


def on_extract(
    content: str,
    reference_time_str: str,
    episode_name: str,
    dedupe_enabled: bool,
):
    """Extract entities and relationships from journal entry text."""
    if not content or not content.strip():
        return (
            "(enter journal text to extract entities)",
            "(no statistics)",
            "(no UUID map)",
            "(no episode)",
            "(no edges extracted)",
            "(no edge statistics)",
            None,
            None,
            None,
        )

    reference_time = _parse_reference_time(reference_time_str)
    name = episode_name.strip() if episode_name and episode_name.strip() else None

    logger.info("Processing journal entry through pipeline")
    logger.info("Reference time: %s", reference_time)
    logger.info("Dedupe enabled: %s", dedupe_enabled)

    try:
        result = asyncio.run(
            add_journal(
                content=content,
                group_id=GROUP_ID,
                reference_time=reference_time,
                name=name,
            )
        )

        entity_list = _format_entity_list(result.nodes)
        stats = _format_stats(result.metadata)
        uuid_map = _format_uuid_map(result.uuid_map, result.nodes)
        episode_details = _format_episode(result.episode)
        edge_list = _format_edge_list(result.edges)
        edge_stats = _format_edge_stats(result.metadata)

        logger.info("Pipeline complete: %d nodes, %d edges", len(result.nodes), len(result.edges))

        return (
            entity_list,
            stats,
            uuid_map,
            episode_details,
            edge_list,
            edge_stats,
            result.episode,
            result.nodes,
            result.uuid_map,
        )

    except Exception as exc:
        logger.exception("Pipeline failed")
        error_msg = f"ERROR: {exc}"
        return (
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            None,
            None,
            None,
        )


def on_write_to_db(episode, nodes):
    """Write episode and nodes to database."""
    if not episode or not nodes:
        return "Run extraction first before writing to database.", "N/A"

    logger.info("Writing episode and %d nodes to database", len(nodes))

    try:
        result = asyncio.run(write_episode_and_nodes(episode, nodes))

        if "error" in result:
            return f"Write failed: {result['error']}", "N/A"

        stats = asyncio.run(get_db_stats())
        stats_str = f"Episodes: {stats['episodes']}, Entities: {stats['entities']}"

        success_msg = f"""Write successful!
Episode UUID: {result['episode_uuid']}
Nodes created: {result['nodes_created']}
Node UUIDs: {', '.join([uuid[:8] + '...' for uuid in result['node_uuids'][:5]])}
{"..." if len(result['node_uuids']) > 5 else ""}"""

        logger.info("Write complete: %d nodes created", result["nodes_created"])

        return success_msg, stats_str

    except Exception as exc:
        logger.exception("Write failed")
        return f"ERROR: {exc}", "N/A"


def on_reset_db():
    """Reset database (clear all data)."""
    logger.warning("Resetting database (clearing all data)")

    try:
        result = asyncio.run(reset_database())
        stats = asyncio.run(get_db_stats())
        stats_str = f"Episodes: {stats['episodes']}, Entities: {stats['entities']}"

        logger.info("Database reset complete")

        return result, stats_str

    except Exception as exc:
        logger.exception("Database reset failed")
        return f"ERROR: {exc}", "N/A"


def get_initial_stats():
    """Get initial database statistics."""
    try:
        stats = asyncio.run(get_db_stats())
        return f"Episodes: {stats['episodes']}, Entities: {stats['entities']}"
    except Exception as exc:
        logger.warning("Failed to get initial stats: %s", exc)
        return "Database unavailable"


with gr.Blocks(title="ExtractNodes Testing UI") as app:
    gr.Markdown("# ExtractNodes Testing UI")
    gr.Markdown(f"""
Test the entity extraction and resolution pipeline.

**Database**: `{DB_PATH}`
**Model**: `{DEFAULT_MODEL_PATH}`
**Group ID**: `{GROUP_ID}`
""")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Input")

            content_input = gr.Textbox(
                label="Journal Entry Text",
                placeholder="Enter journal entry text here...",
                lines=10,
            )

            with gr.Row():
                reference_time_input = gr.Textbox(
                    label="Reference Time (ISO8601, optional)",
                    placeholder="Defaults to now",
                    scale=2,
                )
                episode_name_input = gr.Textbox(
                    label="Episode Name (optional)",
                    placeholder="Auto-generated if empty",
                    scale=2,
                )

            dedupe_toggle = gr.Checkbox(
                label="Enable entity deduplication",
                value=True,
                info="Match extracted entities against existing graph entities",
            )

            extract_btn = gr.Button("Extract Entities", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("## Database Controls")

            db_stats_display = gr.Textbox(
                label="Database Statistics",
                value=get_initial_stats(),
                interactive=False,
                lines=2,
            )

            write_btn = gr.Button("Write to Database", variant="secondary")
            reset_btn = gr.Button("Reset Database", variant="stop")

            write_result_display = gr.Textbox(
                label="Write Result",
                interactive=False,
                lines=8,
            )

    gr.Markdown("## Output")

    with gr.Row():
        entity_list_output = gr.Textbox(
            label="Extracted Entities",
            interactive=False,
            lines=8,
        )

        stats_output = gr.Textbox(
            label="Extraction Statistics",
            interactive=False,
            lines=8,
        )

    with gr.Row():
        uuid_map_output = gr.Textbox(
            label="UUID Mapping (Provisional → Resolved)",
            interactive=False,
            lines=8,
        )

        episode_output = gr.Textbox(
            label="Episode Details",
            interactive=False,
            lines=8,
        )

    with gr.Row():
        edge_list_output = gr.Textbox(
            label="Extracted Edges",
            interactive=False,
            lines=12,
        )

        edge_stats_output = gr.Textbox(
            label="Edge Statistics",
            interactive=False,
            lines=8,
        )

    episode_state = gr.State(None)
    nodes_state = gr.State(None)
    uuid_map_state = gr.State(None)

    extract_btn.click(
        on_extract,
        inputs=[
            content_input,
            reference_time_input,
            episode_name_input,
            dedupe_toggle,
        ],
        outputs=[
            entity_list_output,
            stats_output,
            uuid_map_output,
            episode_output,
            edge_list_output,
            edge_stats_output,
            episode_state,
            nodes_state,
            uuid_map_state,
        ],
    )

    write_btn.click(
        on_write_to_db,
        inputs=[episode_state, nodes_state],
        outputs=[write_result_display, db_stats_display],
    )

    reset_btn.click(
        on_reset_db,
        inputs=[],
        outputs=[write_result_display, db_stats_display],
    )


if __name__ == "__main__":
    app.launch()
