"""Benchmark parallel inference using multiprocessing model pool.

This benchmark tests true parallel inference by running multiple MLX models
in separate processes. Unlike threading (which is serialized by MLX_LOCK),
multiprocessing gives each process its own 4GB model instance.

Run with: python benchmarks/benchmark_model_pool.py

Configuration:
- MAX_PROCESSES: Maximum number of parallel processes (default: 3)
- PROMPTS_PER_PROCESS: Number of prompts each process handles (default: 5)

Expected results with 64GB RAM:
- Serial (1 model): X prompts/sec, 4GB RAM
- Parallel (3 models): ~3X prompts/sec, 12GB RAM
"""

import multiprocessing as mp
import time
import json
from typing import List, Tuple

import dspy
from pydantic import BaseModel, Field

from dspy_outlines import OutlinesLM, OutlinesAdapter


# Configuration
MAX_PROCESSES = 3  # Max parallel processes (adjust based on available RAM)
PROMPTS_PER_PROCESS = 5  # Prompts per process


# Test schema (same as main app)
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Name of the entity")
    properties: dict = Field(default_factory=dict, description="Additional attributes")


class Edge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    label: str = Field(description="Type of relationship")
    properties: dict = Field(default_factory=dict, description="Additional attributes")


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of entities (people)")
    edges: List[Edge] = Field(description="List of relationships between entities")


class ExtractKnowledgeGraph(dspy.Signature):
    """Extract a knowledge graph of people and their relationships from text."""
    text: str = dspy.InputField(desc="Text to extract entities and relationships from")
    graph: KnowledgeGraph = dspy.OutputField(
        desc="Knowledge graph with people as nodes and relationships as edges"
    )


# Sample prompts for benchmarking
SAMPLE_PROMPTS = [
    "Alice works with Bob. Bob reports to Charlie. Charlie mentors Alice.",
    "David and Eve are colleagues. Eve manages Frank. Frank collaborates with David.",
    "Grace is friends with Henry. Henry knows Isaac. Isaac works with Grace.",
    "Jack teaches Kate. Kate studies with Leo. Leo learns from Jack.",
    "Maria leads Noah. Noah supports Olivia. Olivia reports to Maria.",
    "Paul partners with Quinn. Quinn assists Ruby. Ruby collaborates with Paul.",
    "Sam coaches Tina. Tina trains Uma. Uma works with Sam.",
    "Victor knows Wendy. Wendy mentors Xavier. Xavier works with Victor.",
    "Yara manages Zack. Zack supports Yara in project planning.",
    "Anna and Ben are team members. Ben coordinates with Anna on tasks.",
]


def worker_process(process_id: int, prompts: List[str], result_queue: mp.Queue):
    """Worker process that loads its own model and processes prompts.

    Each process:
    1. Loads a separate 4GB MLX model instance
    2. Processes its assigned prompts
    3. Returns timing and result metadata

    Args:
        process_id: Worker ID for logging
        prompts: List of prompts to process
        result_queue: Queue to send results back to main process
    """
    try:
        # Each process loads its own model (separate 4GB instance)
        lm = OutlinesLM()
        dspy.configure(lm=lm, adapter=OutlinesAdapter())
        predictor = dspy.Predict(ExtractKnowledgeGraph)

        start_time = time.time()
        results = []

        for i, prompt in enumerate(prompts):
            prompt_start = time.time()
            result = predictor(text=prompt)
            prompt_duration = time.time() - prompt_start

            results.append({
                'prompt_idx': i,
                'duration': prompt_duration,
                'node_count': len(result.graph.nodes),
                'edge_count': len(result.graph.edges),
            })

        total_duration = time.time() - start_time

        result_queue.put({
            'process_id': process_id,
            'total_duration': total_duration,
            'prompt_count': len(prompts),
            'results': results,
            'success': True
        })
    except Exception as e:
        result_queue.put({
            'process_id': process_id,
            'error': str(e),
            'success': False
        })


def benchmark_serial(prompts: List[str]) -> dict:
    """Benchmark serial inference with a single model.

    Args:
        prompts: List of prompts to process

    Returns:
        dict: Timing and throughput metrics
    """
    lm = OutlinesLM()
    dspy.configure(lm=lm, adapter=OutlinesAdapter())
    predictor = dspy.Predict(ExtractKnowledgeGraph)

    start_time = time.time()
    results = []

    for i, prompt in enumerate(prompts):
        prompt_start = time.time()
        result = predictor(text=prompt)
        prompt_duration = time.time() - prompt_start

        results.append({
            'prompt_idx': i,
            'duration': prompt_duration,
            'node_count': len(result.graph.nodes),
            'edge_count': len(result.graph.edges),
        })

    total_duration = time.time() - start_time

    return {
        'mode': 'serial',
        'total_duration': total_duration,
        'prompt_count': len(prompts),
        'throughput': len(prompts) / total_duration,
        'avg_latency': total_duration / len(prompts),
        'results': results
    }


def benchmark_parallel(num_processes: int, prompts_per_process: int) -> dict:
    """Benchmark parallel inference with multiple processes.

    Args:
        num_processes: Number of parallel worker processes
        prompts_per_process: Number of prompts for each process

    Returns:
        dict: Timing and throughput metrics
    """
    # Generate prompts for each process
    all_prompts = []
    for i in range(num_processes):
        process_prompts = [
            SAMPLE_PROMPTS[j % len(SAMPLE_PROMPTS)]
            for j in range(i * prompts_per_process, (i + 1) * prompts_per_process)
        ]
        all_prompts.append(process_prompts)

    # Create result queue
    result_queue = mp.Queue()

    # Start worker processes
    processes = []
    start_time = time.time()

    for i in range(num_processes):
        p = mp.Process(
            target=worker_process,
            args=(i, all_prompts[i], result_queue)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    total_duration = time.time() - start_time

    # Collect results
    worker_results = []
    for _ in range(num_processes):
        worker_results.append(result_queue.get())

    # Check for errors
    errors = [r for r in worker_results if not r.get('success', False)]
    if errors:
        raise RuntimeError(f"Worker errors: {errors}")

    # Aggregate metrics
    total_prompts = sum(r['prompt_count'] for r in worker_results)

    return {
        'mode': 'parallel',
        'num_processes': num_processes,
        'total_duration': total_duration,
        'prompt_count': total_prompts,
        'throughput': total_prompts / total_duration,
        'avg_latency': total_duration / num_processes,  # Per-process avg
        'worker_results': worker_results
    }


def print_benchmark_results(serial_result: dict, parallel_results: List[dict]):
    """Pretty-print benchmark comparison.

    Args:
        serial_result: Serial benchmark metrics
        parallel_results: List of parallel benchmark metrics
    """
    print("\n" + "=" * 70)
    print("MODEL POOL BENCHMARK RESULTS")
    print("=" * 70)

    # Serial baseline
    print(f"\nSERIAL (1 model):")
    print(f"  Total time: {serial_result['total_duration']:.2f}s")
    print(f"  Prompts: {serial_result['prompt_count']}")
    print(f"  Throughput: {serial_result['throughput']:.2f} prompts/sec")
    print(f"  Avg latency: {serial_result['avg_latency']:.2f}s per prompt")

    # Parallel results
    for result in parallel_results:
        n = result['num_processes']
        speedup = result['throughput'] / serial_result['throughput']
        efficiency = speedup / n * 100

        print(f"\nPARALLEL ({n} processes):")
        print(f"  Total time: {result['total_duration']:.2f}s")
        print(f"  Prompts: {result['prompt_count']}")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")
        print(f"  Avg latency: {result['avg_latency']:.2f}s per process")
        print(f"  Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")
        print(f"  Est. RAM: ~{n * 4}GB ({n} models × 4GB)")

    print("\n" + "=" * 70 + "\n")


def run_full_benchmark():
    """Run full benchmark: serial vs parallel with 1-3 processes."""
    print("\n🚀 Starting model pool benchmark...")
    print(f"Configuration: MAX_PROCESSES={MAX_PROCESSES}, PROMPTS_PER_PROCESS={PROMPTS_PER_PROCESS}")

    # Serial baseline
    print("\n📊 Running serial baseline...")
    total_prompts = MAX_PROCESSES * PROMPTS_PER_PROCESS
    serial_prompts = SAMPLE_PROMPTS[:total_prompts]
    serial_result = benchmark_serial(serial_prompts)

    # Parallel benchmarks
    parallel_results = []
    for n in range(2, MAX_PROCESSES + 1):
        print(f"\n📊 Running parallel benchmark with {n} processes...")
        result = benchmark_parallel(n, PROMPTS_PER_PROCESS)
        parallel_results.append(result)

    # Print comparison
    print_benchmark_results(serial_result, parallel_results)


def run_sanity_check():
    """Sanity check: single worker process should complete without errors."""
    print("\n🔍 Running single worker sanity check...")

    result_queue = mp.Queue()
    prompts = SAMPLE_PROMPTS[:3]  # Just 3 prompts for quick test

    p = mp.Process(target=worker_process, args=(0, prompts, result_queue))
    p.start()
    p.join()

    result = result_queue.get()

    print(f"\nWorker result: {json.dumps(result, indent=2)}")

    if not result['success']:
        print(f"❌ Worker failed: {result.get('error')}")
        return False

    print(f"✅ Sanity check passed!")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--sanity":
        run_sanity_check()
    else:
        run_full_benchmark()
