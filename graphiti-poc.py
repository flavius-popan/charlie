"""Phase 1 PoC: Text → Graph in FalkorDBLite with Gradio UI."""
import os
# Fix tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import gradio as gr
import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM
from settings import MODEL_CONFIG, DB_PATH
from falkordb_utils import get_db_stats, reset_database, write_entities_and_edges
from distilbert_ner import predict_entities
from entity_utils import deduplicate_entities, build_entity_nodes, build_entity_edges, build_episodic_node, build_episodic_edges
from signatures import FactExtractionSignature, RelationshipSignature
from graphviz_utils import load_written_entities, render_graph_from_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Starting Graphiti PoC with MODEL_CONFIG: {MODEL_CONFIG}")
logger.info(f"Database path: {DB_PATH}")

# Configure DSPy once at module level
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)


# Stage 1: NER processing function
def process_ner(text: str, persons_only: bool):
    """
    Stage 1: Extract entities using NER.

    Returns: (entity_names_list, raw_ner_output, display_string)
    """
    if not text.strip():
        return [], None, ""

    # Run NER
    raw_entities = predict_entities(text)

    # Filter by type if requested
    if persons_only:
        filtered = [e for e in raw_entities if e["label"] == "PER"]
    else:
        filtered = raw_entities

    # Extract entity names and deduplicate
    entity_names = [e["text"] for e in filtered]
    unique_names = deduplicate_entities(entity_names)

    logger.info(f"Stage 1: Extracted {len(unique_names)} unique entities")

    # Format for display
    display = "\n".join(unique_names) if unique_names else "(no entities found)"

    return unique_names, raw_entities, display


# Stage 2: Fact extraction function
def extract_facts(text: str, entity_names: list[str]):
    """
    Stage 2: Extract facts using DSPy.

    Returns: (Facts object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        fact_predictor = dspy.Predict(FactExtractionSignature)
        facts = fact_predictor(text=text, entities=entity_names).facts

        logger.info(f"Stage 2: Extracted {len(facts.items)} facts")

        # Convert to JSON for display
        facts_json = {
            "items": [
                {"entity": f.entity, "text": f.text}
                for f in facts.items
            ]
        }

        return facts, facts_json
    except Exception as e:
        return None, {"error": str(e)}


# Stage 3: Relationship inference function
def infer_relationships(text: str, facts, entity_names: list[str]):
    """
    Stage 3: Infer relationships using DSPy.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not facts or not entity_names:
        return None, {"error": "Need text, facts, and entities"}

    try:
        rel_predictor = dspy.Predict(RelationshipSignature)
        relationships = rel_predictor(
            text=text,
            facts=facts,
            entities=entity_names
        ).relationships

        logger.info(f"Stage 3: Inferred {len(relationships.items)} relationships")

        # Convert to JSON for display
        rels_json = {
            "items": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "context": r.context
                }
                for r in relationships.items
            ]
        }

        return relationships, rels_json
    except Exception as e:
        return None, {"error": str(e)}


# Stage 4: Graphiti object builder function
def build_graphiti_objects(input_text, entity_names: list[str], relationships, reference_time):
    """
    Stage 4: Build EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects.

    Now mirrors graphiti.py:413-436 (_process_episode_data).
    """
    if not entity_names or not relationships:
        return None, [], [], [], {
            "error": "Need entities and relationships"
        }

    try:
        # 1. Create EpisodicNode (mirrors graphiti.py:706-720)
        episode = build_episodic_node(
            content=input_text,
            reference_time=reference_time
        )

        # 2. Build EntityNodes
        entity_nodes, entity_map = build_entity_nodes(entity_names)

        # 3. Build EntityEdges with episode UUID (mirrors edge_operations.py:220-230)
        entity_edges = build_entity_edges(
            relationships,
            entity_map,
            episode_uuid=episode.uuid  # Pass episode UUID
        )

        # 4. Link episode to entity edges (mirrors graphiti.py:422)
        episode.entity_edges = [edge.uuid for edge in entity_edges]

        # 5. Build EpisodicEdges (MENTIONS) (mirrors edge_operations.py:51-68)
        episodic_edges = build_episodic_edges(episode, entity_nodes)

        logger.info(
            f"Stage 4: Built episode {episode.uuid}, "
            f"{len(entity_nodes)} nodes, "
            f"{len(entity_edges)} entity edges, "
            f"{len(episodic_edges)} episodic edges"
        )

        # Convert to JSON for display
        graphiti_json = {
            "episode": episode.model_dump(),
            "nodes": [n.model_dump() for n in entity_nodes],
            "entity_edges": [e.model_dump() for e in entity_edges],
            "episodic_edges": [e.model_dump() for e in episodic_edges]
        }

        return episode, entity_nodes, entity_edges, episodic_edges, graphiti_json

    except Exception as e:
        return None, [], [], [], {"error": str(e)}


# Stage 5: FalkorDB write function
def write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges):
    """
    Stage 5: Write EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects to FalkorDB.

    Returns: Write result dict (with UUIDs)
    """
    if not entity_nodes:
        return {"error": "No entities to write"}

    try:
        logger.info(f"Stage 5: Writing episode {episode.uuid}, {len(entity_nodes)} nodes, {len(entity_edges)} entity edges, {len(episodic_edges)} episodic edges to FalkorDB")
        result = write_entities_and_edges(episode, entity_nodes, entity_edges, episodic_edges)
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Stage 6: Graphviz rendering function
def render_verification_graph(write_result):
    """
    Stage 6: Verify FalkorDB write by querying and rendering graph with episode.

    Returns: Path to PNG file, or None on error
    """
    if not write_result or "error" in write_result:
        return None

    # Load episode and entities from DB using UUIDs
    episode_uuid = write_result.get("episode_uuid")
    node_uuids = write_result.get("node_uuids", [])
    edge_uuids = write_result.get("edge_uuids", [])

    if not node_uuids:
        return None

    db_data = load_written_entities(node_uuids, edge_uuids, episode_uuid)
    return render_graph_from_db(db_data)


# Build Gradio interface
with gr.Blocks(title="Phase 1 PoC: Graphiti Pipeline") as app:
    gr.Markdown("# Phase 1 PoC: Text → Graph in FalkorDBLite")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    # Database stats display
    with gr.Row():
        db_stats_display = gr.Textbox(
            label="Database Stats",
            value=lambda: str(get_db_stats()),
            interactive=False
        )
        reset_btn = gr.Button("Reset Database", variant="stop")

    # Example text section
    gr.Markdown("### Example Text")
    example_text = gr.Textbox(
        value="Alice works at Microsoft in Seattle. She reports to Bob, who manages the engineering team.",
        interactive=False,
        show_label=False
    )
    load_example_btn = gr.Button("Load Example", size="sm")

    # Stage 0: Input
    gr.Markdown("## Stage 0: Input Text")
    input_text = gr.Textbox(
        label="Journal Entry",
        placeholder="Enter text here...",
        lines=5
    )

    # Stage 1: NER
    gr.Markdown("## Stage 1: NER Entities")
    ner_output = gr.Textbox(label="Entity Names", interactive=False)
    persons_only_filter = gr.Checkbox(label="Persons only", value=False)

    # Stage 2: Facts
    gr.Markdown("## Stage 2: Fact Extraction")
    run_facts_btn = gr.Button("Run Facts", variant="primary")
    facts_output = gr.JSON(label="Extracted Facts (or error)")

    # Stage 3: Relationships
    gr.Markdown("## Stage 3: Relationship Inference")
    run_relationships_btn = gr.Button("Run Relationships", variant="primary")
    relationships_output = gr.JSON(label="Inferred Relationships (or error)")

    # Stage 4: Graphiti Objects
    gr.Markdown("## Stage 4: Build Graphiti Objects")
    build_graphiti_btn = gr.Button("Build Graphiti Objects", variant="primary")
    graphiti_output = gr.JSON(label="EntityNode + EntityEdge Objects (or error)")

    # Stage 5: FalkorDB Write
    gr.Markdown("## Stage 5: Write to FalkorDB")
    write_falkor_btn = gr.Button("Write to Falkor", variant="primary")
    write_output = gr.JSON(label="Write Confirmation (or error with traceback)")

    # Stage 6: Graphviz Preview
    gr.Markdown("## Stage 6: Graphviz Verification")
    graphviz_output = gr.Image(label="Graph Visualization")

    # State management
    ner_raw_state = gr.State(None)
    entity_names_state = gr.State([])
    facts_state = gr.State(None)
    relationships_state = gr.State(None)
    episode_state = gr.State(None)             # EpisodicNode
    entity_nodes_state = gr.State([])
    entity_edges_state = gr.State([])
    episodic_edges_state = gr.State([])        # EpisodicEdge list
    write_result_state = gr.State(None)

    # Event handlers

    # Stage 1: NER (automatic on text change)
    def on_text_change(text, persons_only):
        entity_names, raw_ner, display = process_ner(text, persons_only)
        return display, entity_names, raw_ner

    input_text.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state],
        trigger_mode="always_last"
    )

    persons_only_filter.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state]
    )

    # Stage 2: Fact Extraction
    def on_run_facts(text, entity_names):
        facts, facts_json = extract_facts(text, entity_names)
        return facts_json, facts

    run_facts_btn.click(
        on_run_facts,
        inputs=[input_text, entity_names_state],
        outputs=[facts_output, facts_state]
    )

    # Stage 3: Relationship Inference
    def on_run_relationships(text, facts, entity_names):
        relationships, rels_json = infer_relationships(text, facts, entity_names)
        return rels_json, relationships

    run_relationships_btn.click(
        on_run_relationships,
        inputs=[input_text, facts_state, entity_names_state],
        outputs=[relationships_output, relationships_state]
    )

    # Stage 4: Build Graphiti Objects
    def on_build_graphiti(text, entity_names, relationships):
        from datetime import datetime
        reference_time = datetime.now()

        episode, nodes, entity_edges, episodic_edges, json_output = build_graphiti_objects(
            input_text=text,
            entity_names=entity_names,
            relationships=relationships,
            reference_time=reference_time
        )
        return json_output, episode, nodes, entity_edges, episodic_edges

    build_graphiti_btn.click(
        on_build_graphiti,
        inputs=[input_text, entity_names_state, relationships_state],
        outputs=[
            graphiti_output,
            episode_state,           # New
            entity_nodes_state,
            entity_edges_state,
            episodic_edges_state     # New
        ]
    )

    # Stage 5: Write to FalkorDB (triggers Stage 6)
    def on_write_falkor(episode, entity_nodes, entity_edges, episodic_edges):
        result = write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges)
        # Update database stats
        new_stats = str(get_db_stats())
        # Render verification graph
        graph_img = render_verification_graph(result)
        return result, result, new_stats, graph_img

    write_falkor_btn.click(
        on_write_falkor,
        inputs=[episode_state, entity_nodes_state, entity_edges_state, episodic_edges_state],
        outputs=[write_output, write_result_state, db_stats_display, graphviz_output]
    )

    def on_reset_db():
        msg = reset_database()
        return str(get_db_stats())

    reset_btn.click(
        on_reset_db,
        outputs=[db_stats_display]
    )

    # Load example handler
    def on_load_example():
        return example_text.value

    load_example_btn.click(
        on_load_example,
        outputs=[input_text]
    )

if __name__ == "__main__":
    app.launch()
