from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from dspy_outlines import KGExtractionModule, OutlinesAdapter, OutlinesLM
import dspy

# Reduce noisy loggers - keep INFO to see HTTP requests, but hide DEBUG noise
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Configure Outlines+MLX hybrid LM
lm = OutlinesLM()
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = Path("prompts/kg_extraction_optimized.json")


def build_extractor(prompts_path: Optional[Path]) -> KGExtractionModule:
    """Instantiate the KG extraction module and optionally load optimized prompts."""
    module = KGExtractionModule()
    if prompts_path and prompts_path.exists():
        try:
            module.load_prompts(str(prompts_path))
            logger.info("Loaded optimized prompts from %s", prompts_path)
        except Exception as exc:  # noqa: BLE001 - CLI should keep running
            logger.warning("Failed to load optimized prompts: %s", exc)
    elif prompts_path:
        logger.warning("Prompts path %s does not exist; continuing without.", prompts_path)
    return module


def extract_graph(
    module: KGExtractionModule, text: str, known_entities: Optional[list[str]] = None
) -> dict[str, Any]:
    """Run the KG extractor and return the prediction payload."""
    prediction = module(text=text, known_entities=known_entities)
    graph = prediction.graph.model_dump()
    metadata = {
        "adapter_used": getattr(adapter, "last_adapter_used", "unknown"),
    }
    return {"graph": graph, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI playground for the knowledge graph extractor."
    )
    parser.add_argument(
        "--text",
        help="Inline text to process (overrides interactive mode).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a UTF-8 text file to process (overrides interactive mode).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS if DEFAULT_PROMPTS.exists() else None,
        help="Optional path to optimized prompts exported via DSPy.",
    )
    parser.add_argument(
        "--entities",
        type=Path,
        help="Optional path to JSON file containing known entities hints.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the extracted graph (default: print compact JSON).",
    )
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> Optional[str]:
    if args.text:
        return args.text
    if args.file:
        try:
            return args.file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read %s: %s", args.file, exc)
            raise
    return None


def load_known_entities(args: argparse.Namespace) -> Optional[list[str]]:
    if not args.entities:
        return None
    try:
        data = json.loads(args.entities.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load known entities from %s: %s", args.entities, exc)
        raise
    if isinstance(data, list):
        return data
    logger.warning("Expected list in %s; ignoring entity hints.", args.entities)
    return None


def interactive_loop(module: KGExtractionModule) -> None:
    print("Knowledge Graph Extractor (Ctrl+D or Ctrl+C to exit)")
    print("=" * 50)
    print("\nPaste your text and press Enter twice:\n")

    while True:
        try:
            lines: list[str] = []
            print("> ", end="", flush=True)
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            text = "\n".join(lines)
            if not text.strip():
                continue

            run_extraction(module, text=text, known_entities=None, pretty=True)
        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def run_extraction(
    module: KGExtractionModule,
    text: str,
    known_entities: Optional[list[str]],
    pretty: bool,
) -> None:
    print("\nExtracting knowledge graph...")
    payload = extract_graph(module, text=text, known_entities=known_entities)
    graph_json = payload["graph"]
    adapter_used = payload["metadata"]["adapter_used"]
    print(f"\nAdapter used: {adapter_used}")
    if pretty:
        print("\nExtracted Knowledge Graph:")
        print(json.dumps(graph_json, indent=2))
    else:
        print(json.dumps(graph_json))
    print("\n" + "=" * 50 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    module = build_extractor(args.prompts)
    known_entities = load_known_entities(args)
    text = load_text(args)

    if text is not None:
        run_extraction(module, text=text, known_entities=known_entities, pretty=args.pretty)
        return

    interactive_loop(module)


if __name__ == "__main__":
    main()
