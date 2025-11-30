"""Run every pipeline optimizer in parallel with prefixed logs and summary."""

from __future__ import annotations

import asyncio
import re
import sys
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from typing import Dict


OPTIMIZERS: Dict[str, str] = {
    "extract_nodes": "pipeline.optimizers.extract_nodes_optimizer",
    "extract_edges": "pipeline.optimizers.extract_edges_optimizer",
    "extract_attributes": "pipeline.optimizers.extract_attributes_optimizer",
    "generate_summaries": "pipeline.optimizers.generate_summaries_optimizer",
}

IMPROVEMENT_RE = re.compile(
    r"Improvement:\s*([0-9.]+)\s*(?:→|->)\s*([0-9.]+)\s*\(([-+0-9.]+)\)"
)


@dataclass
class StageResult:
    status: str = "pending"
    baseline: float | None = None
    optimized: float | None = None
    delta: float | None = None
    returncode: int | None = None


def _maybe_parse_improvement(line: str, result: StageResult) -> None:
    match = IMPROVEMENT_RE.search(line)
    if not match:
        return

    try:
        result.baseline = float(match.group(1))
        result.optimized = float(match.group(2))
        result.delta = float(match.group(3))
    except ValueError:
        return


async def _stream_output(reader, stage: str, stream_name: str, result: StageResult) -> None:
    while True:
        line = await reader.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        print(f"[{stage}][{stream_name}] {text}", flush=True)
        _maybe_parse_improvement(text, result)


async def _run_stage(stage: str, module: str, result: StageResult) -> None:
    print(f"[{stage}] Launching {module}", flush=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        module,
        stdout=PIPE,
        stderr=PIPE,
    )
    result.status = "running"

    stdout_task = asyncio.create_task(_stream_output(process.stdout, stage, "stdout", result))
    stderr_task = asyncio.create_task(_stream_output(process.stderr, stage, "stderr", result))

    returncode = await process.wait()
    await asyncio.gather(stdout_task, stderr_task)

    result.returncode = returncode
    result.status = "success" if returncode == 0 else "failed"
    icon = "✅" if returncode == 0 else "❌"
    print(f"[{stage}] {icon} exited with code {returncode}", flush=True)


def _format_summary(result: StageResult) -> str:
    if (
        result.baseline is not None
        and result.optimized is not None
        and result.delta is not None
    ):
        return f"{result.baseline:.3f} → {result.optimized:.3f} ({result.delta:+.3f})"
    return "no improvement data"


async def main():
    stage_results = {stage: StageResult() for stage in OPTIMIZERS}
    tasks = [
        asyncio.create_task(_run_stage(stage, module, stage_results[stage]))
        for stage, module in OPTIMIZERS.items()
    ]
    await asyncio.gather(*tasks)

    print("\n=== Optimization Summary ===", flush=True)
    for stage in OPTIMIZERS:
        result = stage_results[stage]
        if result.status == "success":
            print(f"- {stage}: SUCCESS {_format_summary(result)}", flush=True)
        else:
            code = result.returncode if result.returncode is not None else "unknown"
            print(f"- {stage}: FAILED (exit code {code})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
