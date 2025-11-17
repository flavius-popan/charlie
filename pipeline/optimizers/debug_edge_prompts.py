"""Diagnostic script to inspect what prompts are being sent to the EdgeExtractor.

This helps us understand why the tiny model is echoing schemas instead of generating data.
"""

import dspy
from pipeline.extract_edges import EdgeExtractor
from pipeline.optimizers.extract_edges_optimizer import build_trainset, configure_dspy

def inspect_prompts():
    """Inspect the actual prompts sent to the model."""

    # Configure DSPy with the same settings as the optimizer
    configure_dspy()

    # Build the dataset
    trainset, valset = build_trainset()

    # Create a baseline EdgeExtractor
    baseline = EdgeExtractor()

    # Get the first validation example
    example = valset[0]

    # Get the adapter (ChatAdapter is the default)
    adapter = dspy.ChatAdapter()

    print("=" * 80)
    print("INSPECTING PROMPTS FOR EDGE EXTRACTION")
    print("=" * 80)
    print()

    # Access the signature from the ChainOfThought module
    # ChainOfThought has a 'predict' attribute that contains the Predict module
    signature = baseline.extractor.predict.signature

    # Format the messages that would be sent to the LM
    messages = adapter.format(
        signature=signature,
        demos=[],  # No demonstrations for now
        inputs={
            'episode_content': example.episode_content,
            'entities_json': example.entities_json,
            'reference_time': example.reference_time,
            'edge_type_context': example.edge_type_context,
            'previous_episodes_json': example.previous_episodes_json,
        }
    )

    # Print each message clearly
    for i, msg in enumerate(messages):
        print("=" * 80)
        print(f"MESSAGE {i+1}: {msg['role'].upper()}")
        print("=" * 80)
        print(msg['content'])
        print()

    # System message was already printed above in MESSAGE 1

    # Print example data for reference
    print("\n" + "=" * 80)
    print("EXAMPLE INPUT DATA")
    print("=" * 80)
    print(f"\nEpisode content: {example.episode_content[:200]}...")
    print(f"\nEntities JSON (first 500 chars): {example.entities_json[:500]}...")
    print(f"\nEdge type context (first 500 chars): {example.edge_type_context[:500]}...")
    print(f"\nReference time: {example.reference_time}")

    # Show what we expect as output
    print("\n" + "=" * 80)
    print("EXPECTED OUTPUT (from training data)")
    print("=" * 80)
    print(f"Number of edges: {len(example.edges.edges)}")
    if len(example.edges.edges) > 0:
        print(f"\nFirst edge example:")
        first_edge = example.edges.edges[0]
        print(f"  source_entity_id: {first_edge.source_entity_id}")
        print(f"  target_entity_id: {first_edge.target_entity_id}")
        print(f"  relation_type: {first_edge.relation_type}")
        print(f"  fact: {first_edge.fact}")

if __name__ == "__main__":
    inspect_prompts()
