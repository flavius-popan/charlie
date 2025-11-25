"""Run all optimizers in parallel with prefixed logs.

Usage: python -m backend.optimizers.run_all
"""
from __future__ import annotations

import asyncio
import re
import sys
from asyncio.subprocess import PIPE
from dataclasses import dataclass

OPTIMIZERS = {
    "extract_nodes": "backend.optimizers.extract_nodes_optimizer",
    # Add more optimizers here as they're created
}

IMPROVEMENT_RE = re.compile(r"Improvement:\s*([0-9.]+)\s*->\s*([0-9.]+)\s*\(\+?([0-9.-]+)\)")


@dataclass
class Result:
    status: str = "pending"
    baseline: float | None = None
    optimized: float | None = None
    delta: float | None = None
    returncode: int | None = None


async def stream_output(reader, name: str, stream: str, result: Result):
    while line := await reader.readline():
        text = line.decode(errors="replace").rstrip()
        print(f"[{name}][{stream}] {text}", flush=True)
        if match := IMPROVEMENT_RE.search(text):
            result.baseline = float(match.group(1))
            result.optimized = float(match.group(2))
            result.delta = float(match.group(3))


async def run_optimizer(name: str, module: str, result: Result):
    print(f"[{name}] Starting {module}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", module, stdout=PIPE, stderr=PIPE
    )
    result.status = "running"

    await asyncio.gather(
        stream_output(proc.stdout, name, "stdout", result),
        stream_output(proc.stderr, name, "stderr", result),
    )

    result.returncode = await proc.wait()
    result.status = "success" if result.returncode == 0 else "failed"
    icon = "OK" if result.returncode == 0 else "FAIL"
    print(f"[{name}] {icon} (exit {result.returncode})", flush=True)


async def main():
    results = {name: Result() for name in OPTIMIZERS}
    tasks = [
        asyncio.create_task(run_optimizer(name, module, results[name]))
        for name, module in OPTIMIZERS.items()
    ]
    await asyncio.gather(*tasks)

    print("\n=== Summary ===", flush=True)
    for name in OPTIMIZERS:
        r = results[name]
        if r.status == "success" and r.delta is not None:
            print(f"  {name}: {r.baseline:.3f} -> {r.optimized:.3f} (+{r.delta:.3f})")
        elif r.status == "success":
            print(f"  {name}: SUCCESS (no improvement data)")
        else:
            print(f"  {name}: FAILED (exit {r.returncode})")


if __name__ == "__main__":
    asyncio.run(main())
