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
import os
from typing import List

import dspy
from pydantic import BaseModel, Field

from dspy_outlines import OutlinesLM, OutlinesAdapter

# Disable tokenizers parallelism warning (we're using multiprocessing, not threading)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Default configuration (can be overridden via CLI)
DEFAULT_MAX_WORKERS = 3  # Max parallel worker processes to test
DEFAULT_WORKLOAD = 24  # Fixed total prompts for ALL tests (serial and parallel)


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
# Expanded to support scaling up to 8+ workers
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
    "Brilliant! Best decision I've made this year.",
    "Not impressed. Had better experiences elsewhere.",
    "Solid choice, would buy again.",
    "Completely useless. Total waste of money.",
    "Works as advertised. No complaints.",
    "Mind-blowing quality! Worth every penny.",
    "Meh. It's just average.",
    "Horrible customer service. Never coming back.",
    "Great value for money. Very satisfied.",
    "Regret this purchase. Should have read reviews.",
    "Perfect! Exactly what I was looking for.",
    "Below expectations. Quite disappointing.",
    "Reliable and efficient. Does the job.",
    "Awful experience from start to finish.",
    "Impressed with the attention to detail.",
    "Nothing to write home about. Pretty basic.",
    "Exceptional! Goes above and beyond.",
    "Frustrating and confusing. Not user-friendly.",
    "Well-made and durable. Happy with purchase.",
    "Worst service I've ever encountered.",
    "Delighted with this purchase! Superb quality.",
    "Underwhelming. Expected much more.",
    "Good enough for the price point.",
    "Terrible quality. Broke after one use.",
    "Wonderful experience! Will recommend to friends.",
    "Just okay. Neither good nor bad.",
    "Amazing! Can't believe how well it works.",
    "Poor design. Very disappointed.",
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

            results.append(
                {
                    "prompt_idx": i,
                    "duration": prompt_duration,
                    "sentiment": result.result.sentiment,
                    "confidence": result.result.confidence,
                }
            )

        total_duration = time.time() - start_time

        result_queue.put(
            {
                "process_id": process_id,
                "total_duration": total_duration,
                "prompt_count": len(prompts),
                "results": results,
                "success": True,
            }
        )
    except Exception as e:
        result_queue.put({"process_id": process_id, "error": str(e), "success": False})


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

        results.append(
            {
                "prompt_idx": i,
                "duration": prompt_duration,
                "sentiment": result.result.sentiment,
                "confidence": result.result.confidence,
            }
        )

    total_duration = time.time() - start_time

    return {
        "mode": "serial",
        "total_duration": total_duration,
        "prompt_count": len(prompts),
        "throughput": len(prompts) / total_duration,
        "avg_latency": total_duration / len(prompts),
        "results": results,
    }


def benchmark_parallel(num_total_processes: int, prompts_per_process: int) -> dict:
    """Benchmark parallel inference with N total processes (including main).

    Strategy: Main process does work too! No idle processes.
    - Total processes = num_total_processes (e.g., 8)
    - Worker processes spawned = num_total_processes - 1 (e.g., 7)
    - Main process handles its own chunk (e.g., 1)
    → All processes work in parallel

    Args:
        num_total_processes: Total number of processes working (including main)
        prompts_per_process: Number of prompts for each process

    Returns:
        dict: Timing and throughput metrics
    """
    # Generate prompts for each process
    all_prompts = []
    for i in range(num_total_processes):
        process_prompts = [
            SAMPLE_PROMPTS[j % len(SAMPLE_PROMPTS)]
            for j in range(i * prompts_per_process, (i + 1) * prompts_per_process)
        ]
        all_prompts.append(process_prompts)

    # Create result queue
    result_queue = mp.Queue()

    # Start worker processes (N-1 workers, main will do the Nth chunk)
    processes = []
    start_time = time.time()

    num_workers = num_total_processes - 1
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, all_prompts[i], result_queue))
        processes.append(p)
        p.start()

    # Main process does work too (handles last chunk)
    main_prompts = all_prompts[num_workers]
    main_result = {
        "process_id": num_workers,
        "total_duration": 0,  # Will be set below
        "prompt_count": len(main_prompts),
        "results": [],
        "success": True,
    }

    # Main process loads model and processes its chunk
    main_start = time.time()
    lm = OutlinesLM()
    dspy.configure(lm=lm, adapter=OutlinesAdapter())
    predictor = dspy.Predict(ClassifySentiment)

    for i, prompt in enumerate(main_prompts):
        prompt_start = time.time()
        result = predictor(text=prompt)
        prompt_duration = time.time() - prompt_start
        main_result["results"].append(
            {
                "prompt_idx": i,
                "duration": prompt_duration,
                "sentiment": result.result.sentiment,
                "confidence": result.result.confidence,
            }
        )
    main_result["total_duration"] = time.time() - main_start

    # Wait for all worker processes to complete
    for p in processes:
        p.join()

    total_duration = time.time() - start_time

    # Collect results from workers
    worker_results = []
    for _ in range(num_workers):
        worker_results.append(result_queue.get())

    # Add main process result
    worker_results.append(main_result)

    # Check for errors
    errors = [r for r in worker_results if not r.get("success", False)]
    if errors:
        raise RuntimeError(f"Worker errors: {errors}")

    # Aggregate metrics (includes main process work)
    total_prompts = sum(r["prompt_count"] for r in worker_results)

    return {
        "mode": "parallel",
        "num_processes": num_total_processes,
        "total_duration": total_duration,
        "prompt_count": total_prompts,
        "throughput": total_prompts / total_duration,
        "avg_latency": total_duration / num_total_processes,  # Per-process avg
        "worker_results": worker_results,
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
    print("\nSERIAL (1 model):")
    print(f"  Total time: {serial_result['total_duration']:.2f}s")
    print(f"  Throughput: {serial_result['throughput']:.2f} prompts/sec")
    print(f"  Avg latency: {serial_result['avg_latency']:.2f}s per prompt")

    # Parallel results
    for result in parallel_results:
        n_total = result["num_processes"]
        speedup = result["throughput"] / serial_result["throughput"]
        efficiency = speedup / n_total * 100

        print(f"\nPARALLEL ({n_total} processes):")
        print(f"  Total time: {result['total_duration']:.2f}s")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")
        print(f"  Avg latency: {result['avg_latency']:.2f}s per process")
        print(f"  Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")

    print("\n" + "=" * 70 + "\n")


def run_full_benchmark(
    max_workers: int = DEFAULT_MAX_WORKERS, total_workload: int = DEFAULT_WORKLOAD
):
    """Run full benchmark: serial vs parallel with SAME total workload.

    Fair benchmark design:
    - Serial: processes total_workload prompts sequentially
    - Parallel (N processes): distributes same total_workload across N processes (all working)
    - All tests process the EXACT SAME number of total prompts
    - Main process works too - no idle processes!

    Workload is automatically adjusted to ensure even distribution:
    - Finds largest multiple of max_workers ≤ total_workload
    - Ensures no remainder/uneven splits

    Args:
        max_workers: Maximum number of parallel processes to test (all working)
        total_workload: Target prompts - will be adjusted to be evenly divisible
    """
    # Adjust workload to be evenly divisible by max_workers (no remainder)
    adjusted_workload = (total_workload // max_workers) * max_workers
    print(
        f"\nBenchmark config: max_workers={max_workers}, workload={adjusted_workload} prompts"
    )

    # Generate fixed workload by cycling through SAMPLE_PROMPTS
    workload_prompts = [
        SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(adjusted_workload)
    ]

    # Serial baseline - processes ALL prompts
    print("Running serial baseline")
    serial_result = benchmark_serial(workload_prompts)

    # Parallel benchmarks - distribute SAME workload evenly across N processes
    parallel_results = []
    for n in range(2, max_workers + 1):
        prompts_per_process = adjusted_workload // n

        print(f"Running {n} in parallel")

        result = benchmark_parallel(n, prompts_per_process)
        parallel_results.append(result)

        # Show result immediately after each run
        speedup = result["throughput"] / serial_result["throughput"]
        efficiency = speedup / n * 100
        print(
            f"  ✓ Completed in {result['total_duration']:.1f}s - Speedup: {speedup:.2f}x ({efficiency:.1f}% efficiency)\n"
        )

    # Print final comparison
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

    if not result["success"]:
        print(f"FAILED: {result.get('error')}")
        return False

    print(
        f"SUCCESS: Processed {result['prompt_count']} prompts in {result['total_duration']:.1f}s"
    )
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark parallel MLX model inference using multiprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python benchmarks/benchmark_model_pool.py --sanity
  python benchmarks/benchmark_model_pool.py
  python benchmarks/benchmark_model_pool.py --workers 6
  python benchmarks/benchmark_model_pool.py --workers 8 --workload 40

Fair Benchmark Design:
  ALL tests (serial and parallel) process the SAME total workload.
  This measures true speedup by comparing time to complete identical work.

  Main process does work too - NO idle processes!
  --workers N = N total processes all working (not N workers + 1 main)

  Default: {DEFAULT_WORKLOAD} prompts total (auto-adjusted to be evenly divisible)
  - Serial: 1 process handles all prompts
  - Parallel: N processes each handle (total/N) prompts

  Example with --workers 6 --workload 24:
    Serial:      1 process × 24 prompts = 24 total
    2 processes: 2 × 12 prompts = 24 total
    3 processes: 3 × 8 prompts = 24 total
    6 processes: 6 × 4 prompts = 24 total
        """,
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run quick sanity check (1 worker, 3 prompts)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        metavar="N",
        help=f"Max parallel processes (all working, including main) - tests 1 to N (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--workload",
        type=int,
        default=DEFAULT_WORKLOAD,
        metavar="N",
        help=f"Total prompts for ALL tests (auto-adjusted for even distribution) (default: {DEFAULT_WORKLOAD})",
    )

    args = parser.parse_args()

    if args.sanity:
        run_sanity_check()
    else:
        run_full_benchmark(max_workers=args.workers, total_workload=args.workload)
