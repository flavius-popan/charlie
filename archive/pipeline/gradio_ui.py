"""Gradio UI for testing ExtractNodes module.

Provides interactive testing interface for entity extraction and resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add archive/ to sys.path for settings, pipeline, inference_runtime imports
_archive_dir = Path(__file__).resolve().parent.parent
if str(_archive_dir) not in sys.path:
    sys.path.insert(0, str(_archive_dir))

# Add repo root to sys.path for backend imports
_repo_root = _archive_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import asyncio
import json
import logging
import time

import _dspy_setup  # noqa: F401
import dspy
import gradio as gr
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

from inference_runtime import DspyLM
from pipeline import (
    ExtractAttributes,
    ExtractEdges,
    ExtractNodes,
    GenerateSummaries,
)
from pipeline.entity_edge_models import (
    edge_meta as DEFAULT_EDGE_META,
)
from pipeline.entity_edge_models import (
    edge_type_map as DEFAULT_EDGE_TYPE_MAP,
)
from pipeline.entity_edge_models import (
    edge_types as DEFAULT_EDGE_TYPES,
)
from pipeline.entity_edge_models import (
    entity_types,
)
from pipeline.falkordblite_driver import (
    enable_tcp_server,
    fetch_recent_episodes,
    get_db_stats,
    get_tcp_server_endpoint,
    get_tcp_server_password,
    persist_episode_and_nodes,
    reset_database,
)
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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

lm = DspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
adapter = dspy.ChatAdapter()
dspy.configure(lm=lm, adapter=adapter)

logger.info("DSPy configured with DspyLM")


def _format_entity_list(nodes, metadata=None) -> str:
    """Format entity nodes as readable list with NEW/MATCHED status."""
    if not nodes:
        return "(no entities extracted)"

    existing_set = set()
    if metadata:
        existing_set = set(metadata.get("existing_entity_uuids", []))

    lines = []
    for node in nodes:
        labels = ", ".join(node.labels) if node.labels else "Entity"
        status = "MATCHED" if node.uuid in existing_set else "NEW"
        base_info = f"- {node.name} [{labels}] [{status}]"
        lines.append(base_info)

    return "\n".join(lines)


def _format_stats(metadata: dict) -> str:
    """Format extraction statistics."""
    first_person = metadata.get("first_person_detected")
    pronoun_line = ""
    if first_person is not None:
        pronoun_line = f"\nFirst-person detected: {bool(first_person)}"
    return f"""Extracted: {metadata.get("extracted_count", 0)} entities
Resolved: {metadata.get("resolved_count", 0)} entities
Exact matches: {metadata.get("exact_matches", 0)}
Fuzzy matches: {metadata.get("fuzzy_matches", 0)}
New entities: {metadata.get("new_entities", 0)}{pronoun_line}"""


def _format_uuid_map(uuid_map: dict, nodes, metadata: dict | None = None) -> str:
    """Format UUID mapping table."""
    if not uuid_map:
        return "(no UUID mappings)"

    lines = ["Provisional UUID → Resolved UUID"]
    lines.append("-" * 50)

    node_names = {node.uuid: node.name for node in nodes}
    existing_set = set()
    if metadata:
        existing_set = set(metadata.get("existing_entity_uuids", []))

    for prov_uuid, res_uuid in uuid_map.items():
        name = node_names.get(res_uuid, "Unknown")
        status = "MATCHED" if res_uuid in existing_set else "NEW"
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
        source_name = uuid_to_name.get(
            edge.source_node_uuid, edge.source_node_uuid[:8] + "..."
        )
        target_name = uuid_to_name.get(
            edge.target_node_uuid, edge.target_node_uuid[:8] + "..."
        )
        fact_preview = edge.fact[:100] + "..." if len(edge.fact) > 100 else edge.fact

        edge_name = edge.name
        if edge.name.upper() in [k.upper() for k in DEFAULT_EDGE_META.keys()]:
            for proper_name in DEFAULT_EDGE_META.keys():
                if proper_name.upper() == edge.name.upper():
                    edge_name = proper_name
                    break

        lines.append(f"- {source_name} → {target_name} [{edge_name}]\n  {fact_preview}")

    return "\n\n".join(lines)


def _format_edge_stats(metadata: dict) -> str:
    """Format edge extraction statistics."""
    edges_meta = metadata.get("edges", {})
    return f"""Extracted: {edges_meta.get("extracted_count", 0)} relationships
Built: {edges_meta.get("built_count", 0)} edges
Resolved: {edges_meta.get("resolved_count", 0)} edges
New: {edges_meta.get("new_count", 0)}
Merged: {edges_meta.get("merged_count", 0)}"""


def _format_attributes_output(nodes) -> str:
    """Format extracted attributes by entity."""
    if not nodes:
        return "(no nodes to display attributes)"

    lines = []
    for node in nodes:
        if not node.attributes:
            continue

        entity_type = next((item for item in node.labels if item != "Entity"), "Entity")
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

    return f"""Processed: {attrs_meta.get("nodes_processed", 0)} nodes
Skipped: {attrs_meta.get("nodes_skipped", 0)} nodes
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

        entity_type = next((item for item in node.labels if item != "Entity"), "Entity")
        lines.append(f"- {node.name} [{entity_type}]")
        lines.append(f"  {node.summary}")
        lines.append("")

    if not lines:
        return "(no summaries generated)"

    return "\n".join(lines)


def _format_summaries_stats(metadata: dict) -> str:
    """Format summary generation statistics."""
    summ_meta = metadata.get("summaries", {})
    return f"""Processed: {summ_meta.get("nodes_processed", 0)} nodes
Avg length: {summ_meta.get("avg_summary_length", 0)} chars
Truncated: {summ_meta.get("truncated_count", 0)} summaries"""


def _format_episode(episode) -> str:
    """Format episode details."""
    if not episode:
        return "(no episode created)"

    content_preview = (
        episode.content[:200] + "..." if len(episode.content) > 200 else episode.content
    )

    return f"""UUID: {episode.uuid}
Name: {episode.name}
Group ID: {episode.group_id}
Valid At: {episode.valid_at.isoformat()}
Created At: {episode.created_at.isoformat()}
Source: {episode.source.value}
Content: {content_preview}"""


def _format_raw_data(data) -> str:
    """Format raw data as JSON for inspection."""
    if data is None:
        return "(no data)"

    try:
        if hasattr(data, "model_dump"):
            return json.dumps(data.model_dump(), indent=2, default=str)
        elif hasattr(data, "__dict__"):
            return json.dumps(data.__dict__, indent=2, default=str)
        else:
            return str(data)
    except Exception as e:
        return f"(error formatting data: {e})"


def _format_timing(timings: dict) -> str:
    """Format timing information."""
    if not timings:
        return "(no timing data)"

    lines = []
    for stage, duration in timings.items():
        lines.append(f"{stage}: {duration:.2f}s")

    if "total" in timings:
        lines.append(f"\nTotal: {timings['total']:.2f}s")

    return "\n".join(lines)


def on_extract(content: str):
    """Extract entities and relationships from journal entry text with progressive updates."""
    if not content or not content.strip():
        empty_result = (
            "(enter journal text to extract entities)",
            "(no statistics)",
            "(no episode)",
            "(no edges extracted)",
            "(no edge statistics)",
            "(no attributes extracted)",
            "(no attribute statistics)",
            "(no summaries generated)",
            "(no summary statistics)",
            "(no LLM output)",
            "(no processed data)",
            "(no LLM output)",
            "(no processed data)",
            "(no LLM output)",
            "(no processed data)",
            "(no LLM output)",
            "(no processed data)",
            "(no timing data)",
            None,
            None,
            None,
        )
        yield empty_result
        return

    logger.info("Processing journal entry through pipeline")

    timings = {}
    pipeline_start = time.time()

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

        logger.info("Starting Stage 1: Extract Nodes")
        stage_start = time.time()
        extractor = ExtractNodes(group_id=GROUP_ID, dedupe_enabled=True)
        extract_result = asyncio.run(
            extractor(
                content=content,
                reference_time=reference_time,
                entity_types=entity_types,
            )
        )
        timings["Stage 1: Extract Nodes"] = time.time() - stage_start

        entity_list = _format_entity_list(extract_result.nodes, extract_result.metadata)
        stats = _format_stats(extract_result.metadata)
        episode_details = _format_episode(extract_result.episode)

        # Format raw LLM output and processed output separately
        llm_stage1 = json.dumps(
            extract_result.raw_llm_output.model_dump() if extract_result.raw_llm_output else None,
            indent=2,
            default=str
        )
        raw_stage1 = json.dumps({
            "nodes": [node.model_dump() if hasattr(node, "model_dump") else str(node) for node in extract_result.nodes],
            "metadata": extract_result.metadata,
        }, indent=2, default=str)

        logger.info("Stage 1 complete: %d nodes extracted", len(extract_result.nodes))

        yield (
            entity_list,
            stats,
            episode_details,
            "(stage 2 running...)",
            "(stage 2 running...)",
            "(stage 3 pending...)",
            "(stage 3 pending...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            llm_stage1,
            raw_stage1,
            "(stage 2 pending...)",
            "(stage 2 pending...)",
            "(stage 3 pending...)",
            "(stage 3 pending...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            _format_timing(timings),
            extract_result.episode,
            extract_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 2: Extract Edges")
        stage_start = time.time()
        edge_extractor = ExtractEdges(
            group_id=GROUP_ID,
            dedupe_enabled=True,
            edge_types=DEFAULT_EDGE_TYPES,
            edge_type_map=DEFAULT_EDGE_TYPE_MAP,
            edge_meta=DEFAULT_EDGE_META,
        )
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
        timings["Stage 2: Extract Edges"] = time.time() - stage_start

        edge_list = _format_edge_list(edges_result.edges, extract_result.nodes)
        edge_stats = _format_edge_stats({"edges": edges_result.metadata})

        # Format raw LLM output and processed output separately
        llm_stage2 = json.dumps(
            edges_result.raw_llm_output.model_dump() if edges_result.raw_llm_output else None,
            indent=2,
            default=str
        )
        raw_stage2 = json.dumps({
            "edges": [edge.model_dump() if hasattr(edge, "model_dump") else str(edge) for edge in edges_result.edges],
            "metadata": edges_result.metadata,
        }, indent=2, default=str)

        logger.info("Stage 2 complete: %d edges extracted", len(edges_result.edges))

        yield (
            entity_list,
            stats,
            episode_details,
            edge_list,
            edge_stats,
            "(stage 3 running...)",
            "(stage 3 running...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            llm_stage1,
            raw_stage1,
            llm_stage2,
            raw_stage2,
            "(stage 3 pending...)",
            "(stage 3 pending...)",
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            _format_timing(timings),
            extract_result.episode,
            extract_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 3: Extract Attributes")
        stage_start = time.time()
        attribute_extractor = ExtractAttributes(group_id=GROUP_ID)
        attributes_result = asyncio.run(
            attribute_extractor(
                nodes=extract_result.nodes,
                episode=extract_result.episode,
                previous_episodes=extract_result.previous_episodes,
                entity_types=entity_types,
            )
        )
        timings["Stage 3: Extract Attributes"] = time.time() - stage_start

        attributes_output = _format_attributes_output(attributes_result.nodes)
        attributes_stats = _format_attributes_stats(
            {"attributes": attributes_result.metadata}
        )

        # Format all raw LLM outputs (one per entity) and processed output separately
        llm_stage3 = json.dumps(
            attributes_result.raw_llm_outputs if attributes_result.raw_llm_outputs else [],
            indent=2,
            default=str
        )
        raw_stage3 = json.dumps({
            "nodes": [node.model_dump() if hasattr(node, "model_dump") else str(node) for node in attributes_result.nodes],
            "metadata": attributes_result.metadata,
        }, indent=2, default=str)

        logger.info(
            "Stage 3 complete: %d nodes processed",
            attributes_result.metadata["nodes_processed"],
        )

        yield (
            entity_list,
            stats,
            episode_details,
            edge_list,
            edge_stats,
            attributes_output,
            attributes_stats,
            "(stage 4 running...)",
            "(stage 4 running...)",
            llm_stage1,
            raw_stage1,
            llm_stage2,
            raw_stage2,
            llm_stage3,
            raw_stage3,
            "(stage 4 pending...)",
            "(stage 4 pending...)",
            _format_timing(timings),
            extract_result.episode,
            attributes_result.nodes,
            extract_result.uuid_map,
        )

        logger.info("Starting Stage 4: Generate Summaries")
        stage_start = time.time()
        summary_generator = GenerateSummaries(group_id=GROUP_ID)
        summaries_result = asyncio.run(
            summary_generator(
                nodes=attributes_result.nodes,
                episode=extract_result.episode,
                previous_episodes=extract_result.previous_episodes,
            )
        )
        timings["Stage 4: Generate Summaries"] = time.time() - stage_start

        summaries_output = _format_summaries_output(summaries_result.nodes)
        summaries_stats = _format_summaries_stats(
            {"summaries": summaries_result.metadata}
        )

        # Format all raw LLM outputs (one per entity) and final processed output separately
        llm_stage4 = json.dumps(
            summaries_result.raw_llm_outputs if summaries_result.raw_llm_outputs else [],
            indent=2,
            default=str
        )
        raw_stage4 = json.dumps({
            "nodes": [node.model_dump() if hasattr(node, "model_dump") else str(node) for node in summaries_result.nodes],
            "metadata": summaries_result.metadata,
        }, indent=2, default=str)

        timings["total"] = time.time() - pipeline_start

        logger.info(
            "Stage 4 complete: %d nodes processed",
            summaries_result.metadata["nodes_processed"],
        )

        yield (
            entity_list,
            stats,
            episode_details,
            edge_list,
            edge_stats,
            attributes_output,
            attributes_stats,
            summaries_output,
            summaries_stats,
            llm_stage1,
            raw_stage1,
            llm_stage2,
            raw_stage2,
            llm_stage3,
            raw_stage3,
            llm_stage4,
            raw_stage4,
            _format_timing(timings),
            extract_result.episode,
            summaries_result.nodes,
            extract_result.uuid_map,
        )

        logger.info(
            "Pipeline complete: %d nodes, %d edges in %.2fs",
            len(summaries_result.nodes),
            len(edges_result.edges),
            timings["total"],
        )

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
Episode UUID: {result["episode_uuid"]}
Nodes written: {result.get("nodes_written", 0)}
Node UUIDs: {", ".join([uuid[:8] + "..." for uuid in result.get("node_uuids", [])[:5]])}
{"..." if len(result.get("node_uuids", [])) > 5 else ""}"""

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


with gr.Blocks(title="Charlie Pipeline UI") as app:
    gr.Markdown("# Charlie Pipeline UI")
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

    extract_btn = gr.Button(
        "Extract Entities & Relationships", variant="primary", size="lg"
    )

    gr.Markdown("## Output")

    gr.Markdown("### Stage 1: Entity Extraction")
    with gr.Row():
        with gr.Column():
            entity_list_output = gr.Textbox(
                label="Extracted Entities",
                interactive=False,
                lines=8,
            )

        with gr.Column():
            stats_output = gr.Textbox(
                label="Extraction Statistics",
                interactive=False,
                lines=8,
            )

    llm_stage1_output = gr.Textbox(
        label="Raw LLM Output (Stage 1)",
        interactive=False,
        lines=10,
        max_lines=20,
    )

    raw_stage1_output = gr.Textbox(
        label="Processed Output → Stage 2 (JSON)",
        interactive=False,
        lines=10,
        max_lines=20,
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

    llm_stage2_output = gr.Textbox(
        label="Raw LLM Output (Stage 2)",
        interactive=False,
        lines=10,
        max_lines=20,
    )

    raw_stage2_output = gr.Textbox(
        label="Processed Output → Stage 3 (JSON)",
        interactive=False,
        lines=10,
        max_lines=20,
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

    llm_stage3_output = gr.Textbox(
        label="Raw LLM Outputs (Stage 3 - All Entities)",
        interactive=False,
        lines=10,
        max_lines=20,
    )

    raw_stage3_output = gr.Textbox(
        label="Processed Output → Stage 4 (JSON)",
        interactive=False,
        lines=10,
        max_lines=20,
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

    llm_stage4_output = gr.Textbox(
        label="Raw LLM Outputs (Stage 4 - All Entities)",
        interactive=False,
        lines=10,
        max_lines=20,
    )

    raw_stage4_output = gr.Textbox(
        label="Final Processed Output (JSON)",
        interactive=False,
        lines=10,
        max_lines=20,
    )

    gr.Markdown("## Performance & Debug")

    with gr.Row():
        timing_output = gr.Textbox(
            label="Pipeline Timing",
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
        value="(waiting for write operation)",
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
            episode_output,
            edge_list_output,
            edge_stats_output,
            attributes_output,
            attributes_stats_output,
            summaries_output,
            summaries_stats_output,
            llm_stage1_output,
            raw_stage1_output,
            llm_stage2_output,
            raw_stage2_output,
            llm_stage3_output,
            raw_stage3_output,
            llm_stage4_output,
            raw_stage4_output,
            timing_output,
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
