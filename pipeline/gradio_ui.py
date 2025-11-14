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
from graphiti_core.nodes import EpisodicNode, EpisodeType
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

from pipeline import (
    ExtractNodes,
    ExtractEdges,
    ExtractAttributes,
    GenerateSummaries,
)
from pipeline.falkordblite_driver import (
    enable_tcp_server,
    fetch_entities_by_group,
    fetch_recent_episodes,
    get_db_stats,
    get_tcp_server_endpoint,
    get_tcp_server_password,
    persist_episode_and_nodes,
    reset_database,
)
from pipeline.entity_edge_models import entity_types
from settings import (
    DB_PATH,
    DEFAULT_MODEL_PATH,
    FALKORLITE_TCP_HOST,
    FALKORLITE_TCP_PASSWORD,
    FALKORLITE_TCP_PORT,
    GROUP_ID,
    MODEL_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Initializing ExtractNodes UI")
logger.info("Database: %s", DB_PATH)
logger.info("Model: %s", DEFAULT_MODEL_PATH)
logger.info("Model config: %s", MODEL_CONFIG)

enable_tcp_server(
    host=FALKORLITE_TCP_HOST,
    port=FALKORLITE_TCP_PORT,
    password=FALKORLITE_TCP_PASSWORD,
)
tcp_endpoint = get_tcp_server_endpoint()
if tcp_endpoint:
    host, port = tcp_endpoint
    password = get_tcp_server_password()
    if password:
        logger.info(
            "FalkorDB Lite TCP debug endpoint listening on %s:%d (password required)",
            host,
            port,
        )
    else:
        logger.info(
            "FalkorDB Lite TCP debug endpoint listening on %s:%d (no password)",
            host,
            port,
        )

lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)

logger.info("DSPy configured with OutlinesLM")


def _format_entity_list(nodes) -> str:
    """Format entity nodes as readable list."""
    if not nodes:
        return "(no entities extracted)"

    lines = []
    for node in nodes:
        labels = ", ".join(node.labels) if node.labels else "Entity"
        base_info = f"- {node.name} [{labels}]"
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


def _format_edge_list(edges, nodes=None) -> str:
    """Format entity edges as readable list."""
    if not edges:
        return "(no edges extracted)"

    uuid_to_name = {}
    if nodes:
        uuid_to_name = {node.uuid: node.name for node in nodes}

    lines = []
    for edge in edges:
        source_name = uuid_to_name.get(edge.source_node_uuid, edge.source_node_uuid[:8] + "...")
        target_name = uuid_to_name.get(edge.target_node_uuid, edge.target_node_uuid[:8] + "...")
        fact_preview = edge.fact[:100] + "..." if len(edge.fact) > 100 else edge.fact

        lines.append(
            f"- {source_name} → {target_name} [{edge.name}]\n"
            f"  {fact_preview}"
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


def _format_attributes_output(nodes) -> str:
    """Format extracted attributes by entity."""
    if not nodes:
        return "(no nodes to display attributes)"

    lines = []
    for node in nodes:
        if not node.attributes:
            continue

        entity_type = next((item for item in node.labels if item != 'Entity'), 'Entity')
        lines.append(f"- {node.name} [{entity_type}]")

        for key, value in node.attributes.items():
            lines.append(f"  {key}: {value}")

    if not lines:
        return "(no attributes extracted)"

    return "\n".join(lines)


def _format_attributes_stats(metadata: dict) -> str:
    """Format attribute extraction statistics."""
    attrs_meta = metadata.get("attributes", {})
    by_type = attrs_meta.get("attributes_extracted_by_type", {})

    by_type_str = "\n".join([f"  {typ}: {count}" for typ, count in by_type.items()])
    if not by_type_str:
        by_type_str = "  (none)"

    return f"""Processed: {attrs_meta.get('nodes_processed', 0)} nodes
Skipped: {attrs_meta.get('nodes_skipped', 0)} nodes
By type:
{by_type_str}"""


def _format_summaries_output(nodes) -> str:
    """Format generated summaries by entity."""
    if not nodes:
        return "(no nodes to display summaries)"

    lines = []
    for node in nodes:
        if not node.summary:
            continue

        entity_type = next((item for item in node.labels if item != 'Entity'), 'Entity')
        lines.append(f"- {node.name} [{entity_type}]")
        lines.append(f"  {node.summary}")
        lines.append("")

    if not lines:
        return "(no summaries generated)"

    return "\n".join(lines)


def _format_summaries_stats(metadata: dict) -> str:
    """Format summary generation statistics."""
    summ_meta = metadata.get("summaries", {})
    return f"""Processed: {summ_meta.get('nodes_processed', 0)} nodes
Avg length: {summ_meta.get('avg_summary_length', 0)} chars
Truncated: {summ_meta.get('truncated_count', 0)} summaries"""


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


def on_extract(content: str):
    """Extract entities and relationships from journal entry text with progressive updates."""
    if not content or not content.strip():
        empty_result = (
            "(enter journal text to extract entities)",
            "(no statistics)",
            "(no UUID map)",
            "(no episode)",
            "(no edges extracted)",
            "(no edge statistics)",
            "(no attributes extracted)",
            "(no attribute statistics)",
            "(no summaries generated)",
            "(no summary statistics)",
            None,
            None,
            None,
        )
        yield empty_result
        return

    logger.info("Processing journal entry through pipeline")

    try:
        reference_time = ensure_utc(utc_now())
        episode = EpisodicNode(
            name=f"journal_{reference_time.isoformat()}",
            group_id=GROUP_ID,
            labels=[],
            source=EpisodeType.text,
            content=content,
            source_description="Journal entry",
            created_at=utc_now(),
            valid_at=reference_time,
        )

        logger.info("Created episode %s", episode.uuid)

        previous_episodes = asyncio.run(
            fetch_recent_episodes(GROUP_ID, reference_time, limit=5)
        )

        logger.info("Starting Stage 1: Extract Nodes")
        extractor = ExtractNodes(group_id=GROUP_ID, dedupe_enabled=True)
        extract_result = asyncio.run(
            extractor(
                content=content,
                reference_time=reference_time,
                entity_types=entity_types,
            )
        )

        entity_list = _format_entity_list(extract_result.nodes)
        stats = _format_stats(extract_result.metadata)
        uuid_map = _format_uuid_map(extract_result.uuid_map, extract_result.nodes)
        episode_details = _format_episode(extract_result.episode)

        logger.info("Stage 1 complete: %d nodes extracted", len(extract_result.nodes))

        yield (
            entity_list,
            stats,
            uuid_map,
            episode_details,
            "(stage 2 running...)",
            "(stage 2 running...)",
            "(stage 3 pending...)",
            "(stage 3 pending...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            extract_result.episode,
            extract_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 2: Extract Edges")
        edge_extractor = ExtractEdges(group_id=GROUP_ID, dedupe_enabled=True)
        edges_result = asyncio.run(
            edge_extractor(
                episode=extract_result.episode,
                extracted_nodes=extract_result.extracted_nodes,
                resolved_nodes=extract_result.nodes,
                uuid_map=extract_result.uuid_map,
                previous_episodes=extract_result.previous_episodes,
                entity_types=entity_types,
            )
        )

        edge_list = _format_edge_list(edges_result.edges, extract_result.nodes)
        edge_stats = _format_edge_stats({"edges": edges_result.metadata})

        logger.info("Stage 2 complete: %d edges extracted", len(edges_result.edges))

        yield (
            entity_list,
            stats,
            uuid_map,
            episode_details,
            edge_list,
            edge_stats,
            "(stage 3 running...)",
            "(stage 3 running...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            extract_result.episode,
            extract_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 3: Extract Attributes")
        attribute_extractor = ExtractAttributes(group_id=GROUP_ID)
        attributes_result = asyncio.run(
            attribute_extractor(
                nodes=extract_result.nodes,
                episode=extract_result.episode,
                previous_episodes=extract_result.previous_episodes,
                entity_types=entity_types,
            )
        )

        attributes_output = _format_attributes_output(attributes_result.nodes)
        attributes_stats = _format_attributes_stats({"attributes": attributes_result.metadata})

        logger.info("Stage 3 complete: %d nodes processed", attributes_result.metadata['nodes_processed'])

        yield (
            entity_list,
            stats,
            uuid_map,
            episode_details,
            edge_list,
            edge_stats,
            attributes_output,
            attributes_stats,
            "(stage 4 running...)",
            "(stage 4 running...)",
            extract_result.episode,
            attributes_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 4: Generate Summaries")
        summary_generator = GenerateSummaries(group_id=GROUP_ID)
        summaries_result = asyncio.run(
            summary_generator(
                nodes=attributes_result.nodes,
                episode=extract_result.episode,
                previous_episodes=extract_result.previous_episodes,
            )
        )

        summaries_output = _format_summaries_output(summaries_result.nodes)
        summaries_stats = _format_summaries_stats({"summaries": summaries_result.metadata})

        logger.info("Stage 4 complete: %d nodes processed", summaries_result.metadata['nodes_processed'])

        yield (
            entity_list,
            stats,
            uuid_map,
            episode_details,
            edge_list,
            edge_stats,
            attributes_output,
            attributes_stats,
            summaries_output,
            summaries_stats,
            extract_result.episode,
            summaries_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Pipeline complete: %d nodes, %d edges", len(summaries_result.nodes), len(edges_result.edges))

    except Exception as exc:
        logger.exception("Pipeline failed")
        error_msg = f"ERROR: {exc}"
        yield (
            error_msg,
            error_msg,
            error_msg,
            error_msg,
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
        result = asyncio.run(persist_episode_and_nodes(episode, nodes))

        if "error" in result:
            return f"Write failed: {result['error']}", "N/A"

        stats = asyncio.run(get_db_stats())
        stats_str = f"Episodes: {stats['episodes']}, Entities: {stats['entities']}"

        success_msg = f"""Write successful!
Episode UUID: {result['episode_uuid']}
Nodes written: {result.get('nodes_written', 0)}
Node UUIDs: {', '.join([uuid[:8] + '...' for uuid in result.get('node_uuids', [])[:5]])}
{"..." if len(result.get('node_uuids', [])) > 5 else ""}"""

        logger.info("Write complete: %d nodes written", result.get("nodes_written", 0))

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


with gr.Blocks(title="Graphiti Pipeline Testing UI") as app:
    gr.Markdown("# Graphiti Pipeline Testing UI")
    gr.Markdown(f"""
Test the complete knowledge graph extraction pipeline with progressive stage updates.

**Database**: `{DB_PATH}`
**Model**: `{DEFAULT_MODEL_PATH}`
**Group ID**: `{GROUP_ID}`
""")

    gr.Markdown("## Input")

    content_input = gr.Textbox(
        label="Journal Entry Text",
        placeholder="Enter journal entry text here...",
        lines=10,
    )

    extract_btn = gr.Button("Extract Entities & Relationships", variant="primary", size="lg")

    gr.Markdown("## Output")

    gr.Markdown("### Stage 1: Entity Extraction")
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

    gr.Markdown("### Stage 2: Relationship Extraction")
    with gr.Row():
        edge_list_output = gr.Textbox(
            label="Extracted Relationships",
            interactive=False,
            lines=10,
        )

        edge_stats_output = gr.Textbox(
            label="Relationship Statistics",
            interactive=False,
            lines=8,
        )

    gr.Markdown("### Stage 3: Attribute Extraction")
    with gr.Row():
        attributes_output = gr.Textbox(
            label="Extracted Attributes",
            interactive=False,
            lines=10,
        )

        attributes_stats_output = gr.Textbox(
            label="Attribute Statistics",
            interactive=False,
            lines=8,
        )

    gr.Markdown("### Stage 4: Summary Generation")
    with gr.Row():
        summaries_output = gr.Textbox(
            label="Generated Summaries",
            interactive=False,
            lines=10,
        )

        summaries_stats_output = gr.Textbox(
            label="Summary Statistics",
            interactive=False,
            lines=8,
        )

    gr.Markdown("### Debug Information")
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

    gr.Markdown("## Database Controls")

    with gr.Row():
        with gr.Column(scale=2):
            db_stats_display = gr.Textbox(
                label="Database Statistics",
                value="Loading...",
                interactive=False,
                lines=2,
            )

        with gr.Column(scale=1):
            write_btn = gr.Button("Write to Database", variant="secondary")
            reset_btn = gr.Button("Reset Database", variant="stop")

    write_result_display = gr.Textbox(
        label="Write Result",
        interactive=False,
        lines=6,
    )

    episode_state = gr.State(None)
    nodes_state = gr.State(None)
    uuid_map_state = gr.State(None)

    extract_btn.click(
        on_extract,
        inputs=[content_input],
        outputs=[
            entity_list_output,
            stats_output,
            uuid_map_output,
            episode_output,
            edge_list_output,
            edge_stats_output,
            attributes_output,
            attributes_stats_output,
            summaries_output,
            summaries_stats_output,
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

    app.load(
        fn=get_initial_stats,
        inputs=None,
        outputs=db_stats_display,
    )


if __name__ == "__main__":
    app.launch()
