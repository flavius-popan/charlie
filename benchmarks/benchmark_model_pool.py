"""Benchmark parallel inference using multiprocessing model pool.

This benchmark tests true parallel inference by running multiple MLX models
in separate processes. Unlike threading (which is serialized by MLX_LOCK),
multiprocessing gives each process its own 4GB model instance.

Run with: python benchmarks/benchmark_model_pool.py

Configuration:
- MAX_WORKERS: Maximum number of parallel worker processes (default: 3)
- PROMPTS_PER_WORKER: Number of prompts each worker handles (default: 3)

Expected results with 64GB RAM:
- Serial (1 model): X prompts/sec, 4GB RAM
- Parallel (3 workers): ~3X prompts/sec, 12GB RAM

Note: Total process count = 1 main process + N worker processes
"""

import multiprocessing as mp
import time
import json
import os
from typing import List, Tuple

import dspy
from pydantic import BaseModel, Field

from dspy_outlines import OutlinesLM, OutlinesAdapter

# Disable tokenizers parallelism warning (we're using multiprocessing, not threading)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configuration
MAX_WORKERS = 3  # Max parallel worker processes (total processes = 1 main + N workers)
PROMPTS_PER_WORKER = 3  # Prompts per worker (reduced for faster benchmarking)


# Simple test schema - just extract a sentiment classification
class SentimentResult(BaseModel):
    """Simple sentiment classification."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: str = Field(description="high, medium, or low")


class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a text."""
    text: str = dspy.InputField(desc="Text to classify")
    result: SentimentResult = dspy.OutputField(desc="Sentiment classification")


# Sample prompts for benchmarking (simple, fast to process)
SAMPLE_PROMPTS = [
    "I love this product! It's amazing and works perfectly.",
    "This is terrible. Worst purchase ever.",
    "It's okay, nothing special.",
    "Absolutely fantastic experience! Highly recommend.",
    "Disappointed with the quality. Not worth the money.",
    "Pretty good overall, but could be better.",
    "Outstanding service! Will definitely come back.",
    "Waste of time and money. Avoid at all costs.",
    "Decent product for the price.",
    "Exceeded all my expectations! Five stars!",
    "Mediocre at best. Expected more.",
    "Can't complain, does what it's supposed to do.",
]


def worker_process(process_id: int, prompts: List[str], result_queue: mp.Queue):
    """Worker process that loads its own model and processes prompts.

    Timing includes:
    - Model loading (4GB MLX model load time)
    - All inference time (including any gaps)
    - Total wall-clock time from start to finish

    Args:
        process_id: Worker ID for logging
        prompts: List of prompts to process
        result_queue: Queue to send results back to main process
    """
    try:
        start_time = time.time()  # Start timing BEFORE model load

        # Each process loads its own model (separate 4GB instance)
        lm = OutlinesLM()
        dspy.configure(lm=lm, adapter=OutlinesAdapter())
        predictor = dspy.Predict(ClassifySentiment)

        results = []

        for i, prompt in enumerate(prompts):
            prompt_start = time.time()
            result = predictor(text=prompt)
            prompt_duration = time.time() - prompt_start

            results.append({
                'prompt_idx': i,
                'duration': prompt_duration,
                'sentiment': result.result.sentiment,
                'confidence': result.result.confidence,
            })

        total_duration = time.time() - start_time  # Wall-clock time

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

    Wall-clock timing includes model loading + all inference time.

    Args:
        prompts: List of prompts to process

    Returns:
        dict: Timing and throughput metrics
    """
    start_time = time.time()  # Start timing BEFORE model load

    lm = OutlinesLM()
    dspy.configure(lm=lm, adapter=OutlinesAdapter())
    predictor = dspy.Predict(ClassifySentiment)

    results = []

    for i, prompt in enumerate(prompts):
        prompt_start = time.time()
        result = predictor(text=prompt)
        prompt_duration = time.time() - prompt_start

        results.append({
            'prompt_idx': i,
            'duration': prompt_duration,
            'sentiment': result.result.sentiment,
            'confidence': result.result.confidence,
        })

    total_duration = time.time() - start_time  # Wall-clock time

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
        n_workers = result['num_processes']
        n_total = n_workers + 1  # +1 for main process
        speedup = result['throughput'] / serial_result['throughput']
        efficiency = speedup / n_workers * 100

        print(f"\nPARALLEL ({n_workers} workers, {n_total} total processes):")
        print(f"  Total time: {result['total_duration']:.2f}s")
        print(f"  Prompts: {result['prompt_count']}")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")
        print(f"  Avg latency: {result['avg_latency']:.2f}s per worker")
        print(f"  Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")
        print(f"  Est. RAM: ~{n_workers * 4}GB ({n_workers} models × 4GB)")

    print("\n" + "=" * 70 + "\n")


def run_full_benchmark():
    """Run full benchmark: serial vs parallel with 2-3 worker processes."""
    print(f"\nBenchmark config: MAX_WORKERS={MAX_WORKERS}, PROMPTS_PER_WORKER={PROMPTS_PER_WORKER}")
    print("Note: Timing includes model loading + inference (wall-clock)")
    print("      Process count = 1 main + N workers\n")

    # Serial baseline - same total workload as largest parallel test
    print("Running serial baseline (1 model, 1 process)...")
    total_prompts = MAX_WORKERS * PROMPTS_PER_WORKER
    serial_prompts = SAMPLE_PROMPTS[:total_prompts]
    serial_result = benchmark_serial(serial_prompts)

    # Parallel benchmarks with 2, 3 worker processes
    parallel_results = []
    for n in range(2, MAX_WORKERS + 1):
        print(f"Running parallel ({n} workers, {n+1} total processes)...")
        result = benchmark_parallel(n, PROMPTS_PER_WORKER)
        parallel_results.append(result)

    # Print comparison
    print_benchmark_results(serial_result, parallel_results)


def run_sanity_check():
    """Sanity check: single worker process should complete without errors."""
    print("\nRunning single worker sanity check (3 prompts)...")

    result_queue = mp.Queue()
    prompts = SAMPLE_PROMPTS[:3]  # Just 3 prompts for quick test

    p = mp.Process(target=worker_process, args=(0, prompts, result_queue))
    p.start()
    p.join()

    result = result_queue.get()

    if not result['success']:
        print(f"FAILED: {result.get('error')}")
        return False

    print(f"SUCCESS: Processed {result['prompt_count']} prompts in {result['total_duration']:.1f}s")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--sanity":
        run_sanity_check()
    else:
        run_full_benchmark()
