"""
Journal Loader for Charlie

Loads journal entries from JSON files into the Graphiti knowledge graph.
Incorporates FTS extension setup and supports step-by-step verification.
"""

import asyncio
import json
import logging
import argparse
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
import kuzu

from graphiti_core import Graphiti
import graphiti_core.edges
import graphiti_core.nodes
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType

# Monkey-patch to fix Kuzu naive datetime issue
# See: https://github.com/getzep/graphiti/issues/893
import graphiti_core.helpers
from neo4j import time as neo4j_time

from app import settings
from app.llm.schema_patches import apply_all_patches


#################################################
# MONKEY-PATCH FOR KUZU DATETIME
#################################################
def patched_parse_db_date(
    input_date: neo4j_time.DateTime | str | datetime | None,
) -> datetime | None:
    """
    Patched version of parse_db_date that ensures all datetimes are timezone-aware.
    This fixes the issue where Kuzu returns naive datetimes but graphiti expects offset-aware ones.
    """
    if isinstance(input_date, neo4j_time.DateTime):
        return input_date.to_native()

    if isinstance(input_date, str):
        dt = datetime.fromisoformat(input_date)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    if isinstance(input_date, datetime):
        if input_date.tzinfo is None:
            return input_date.replace(tzinfo=timezone.utc)
        return input_date

    return input_date


# Apply the monkey-patch to both modules
graphiti_core.helpers.parse_db_date = patched_parse_db_date  # type: ignore[assignment]
graphiti_core.edges.parse_db_date = patched_parse_db_date  # type: ignore[attr-defined]
graphiti_core.nodes.parse_db_date = patched_parse_db_date  # type: ignore[attr-defined]


#################################################
# LOGGING CONFIGURATION
#################################################
logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

#################################################
# SCHEMA PATCHES FOR ENTITY EXTRACTION LIMITS
#################################################
# Apply schema constraints to prevent runaway entity generation
apply_all_patches()
logging.info("Schema patches applied: entity extraction limits enabled")


#################################################
# FTS EXTENSION SETUP
#################################################
def setup_fts_extension(db_path: str) -> None:
    """
    Install and load the FTS (Full-Text Search) extension for Kuzu.
    This must be done before graphiti builds indices.
    """
    logger.info(f"Setting up FTS extension for database: {db_path}")

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    try:
        logger.info("Installing FTS extension...")
        conn.execute("INSTALL FTS")
        logger.info("FTS extension installed successfully")

        logger.info("Loading FTS extension...")
        conn.execute("LOAD FTS")
        logger.info("FTS extension loaded successfully")
    except Exception as e:
        logger.error(f"Error setting up FTS extension: {e}")
        raise
    finally:
        # Connection is automatically closed when it goes out of scope
        pass


#################################################
# JOURNAL LOADING
#################################################
async def load_journals(
    journal_file_path: str, db_path: str, skip_verification: bool = False
) -> None:
    """
    Load journal entries from a JSON file into the Graphiti knowledge graph.

    Args:
        journal_file_path: Path to the journal JSON file
        db_path: Path to the Kuzu database
        skip_verification: If True, load all entries without pausing for verification
    """
    # Read and parse journal file
    logger.info(f"Reading journal file: {journal_file_path}")

    if not os.path.exists(journal_file_path):
        logger.error(f"Journal file not found: {journal_file_path}")
        raise FileNotFoundError(f"Journal file not found: {journal_file_path}")

    with open(journal_file_path, "r", encoding="utf-8") as f:
        journal_data = json.load(f)

    # Validate structure
    if "entries" not in journal_data:
        logger.error("Invalid journal structure: missing 'entries' key")
        raise ValueError("Invalid journal structure: missing 'entries' key")

    entries = journal_data["entries"]
    total_entries = len(entries)
    logger.info(f"Found {total_entries} journal entries")

    # Initialize Graphiti with MLX backend
    logger.info(f"Initializing Graphiti with database: {db_path}")

    # Initialize MLX LLM model and tokenizer
    # Note: Tested explicit Qwen config (eos_token, trust_remote_code) vs defaults.
    # Result: Model defaults are correct - uses <|im_end|> as EOS (ID: 151645)
    # which matches the chat template. No explicit config needed.
    # See: test_qwen_config.py for full comparison
    logger.info("Loading MLX LLM model and tokenizer...")
    import mlx_lm

    mlx_model, mlx_tokenizer = mlx_lm.load(settings.MLX_MODEL_NAME)
    logger.info("MLX LLM model loaded successfully")

    # Initialize MLX embedding model and tokenizer
    logger.info("Loading MLX embedding model and tokenizer...")
    mlx_embedding_model, mlx_embedding_tokenizer = mlx_lm.load(
        settings.MLX_EMBEDDING_MODEL_NAME
    )
    logger.info("MLX embedding model loaded successfully")

    # Create Outlines model wrapper
    logger.info("Initializing Outlines structured generation...")
    import outlines

    outlines_model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)
    logger.info("Outlines model ready")

    # Initialize bridge components
    from app.llm.client import GraphitiLM
    from app.llm.embedder import MLXEmbedder

    llm_client = GraphitiLM(outlines_model, mlx_tokenizer)
    embedder = MLXEmbedder(
        mlx_embedding_model,
        mlx_embedding_tokenizer,
        embedding_dim=settings.MLX_EMBEDDING_DIM,
    )
    logger.info(
        f"LLM client and embedder initialized with embedding dimension: {settings.MLX_EMBEDDING_DIM}"
    )

    # Initialize Graphiti with local MLX backend
    kuzu_driver = KuzuDriver(db=db_path)
    graphiti = Graphiti(
        graph_driver=kuzu_driver, llm_client=llm_client, embedder=embedder
    )

    try:
        # Build indices and constraints
        logger.info("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()
        logger.info("Indices and constraints built successfully")

        # Process each journal entry
        loaded_count = 0
        skipped_count = 0

        for i, entry in enumerate(entries, 1):
            try:
                # Extract required fields
                if "text" not in entry:
                    logger.warning(f"Entry {i} missing 'text' field, skipping")
                    skipped_count += 1
                    continue

                if "creationDate" not in entry:
                    logger.warning(f"Entry {i} missing 'creationDate' field, skipping")
                    skipped_count += 1
                    continue

                text = entry["text"]
                creation_date_str = entry["creationDate"]
                uuid = entry.get("uuid", f"entry_{i}")

                # Parse creation date
                # Handle ISO format with Z suffix
                creation_date_str = creation_date_str.replace("Z", "+00:00")
                reference_time = datetime.fromisoformat(creation_date_str)

                # Ensure timezone-aware
                if reference_time.tzinfo is None:
                    reference_time = reference_time.replace(tzinfo=timezone.utc)

                # Create episode name
                episode_name = f"journal_{uuid}"

                # Add episode to knowledge graph
                await graphiti.add_episode(
                    name=episode_name,
                    episode_body=text,
                    source=EpisodeType.text,
                    source_description=settings.SOURCE_DESCRIPTION,
                    reference_time=reference_time,
                )

                loaded_count += 1

                # Display progress
                print(f"\n{'=' * 60}")
                print(f"✓ Loaded entry {loaded_count}/{total_entries}")
                print(f"  UUID: {uuid}")
                print(f"  Date: {reference_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"  Length: {len(text)} characters")

                # Show preview
                preview = text[:200].replace("\n", " ")
                if len(text) > 200:
                    preview += "..."
                print(f"  Preview: {preview}")
                print(f"{'=' * 60}")

                # Wait for user confirmation unless skipping
                if not skip_verification:
                    try:
                        input("\nPress Enter to load next entry (Ctrl+C to abort)...")
                    except KeyboardInterrupt:
                        print("\n\nLoading aborted by user")
                        logger.info(
                            f"Loading aborted. Loaded {loaded_count} entries, skipped {skipped_count}"
                        )
                        break

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error processing entry {i}: {e}")
                skipped_count += 1
                continue

        # Final summary
        print(f"\n{'=' * 60}")
        print("Loading complete!")
        print(f"  Successfully loaded: {loaded_count} entries")
        print(f"  Skipped: {skipped_count} entries")
        print(f"  Total processed: {total_entries} entries")
        print(f"{'=' * 60}\n")

        logger.info(
            f"Journal loading complete: {loaded_count} loaded, {skipped_count} skipped"
        )

    finally:
        # Close graphiti connection
        await graphiti.close()
        logger.info("Graphiti connection closed")


#################################################
# COMMUNITY BUILDING
#################################################
async def build_communities(db_path: str) -> None:
    """
    Build communities in the knowledge graph using the Leiden algorithm.

    Communities represent groups of strongly connected entity nodes.
    This will remove any existing communities before creating new ones.

    Args:
        db_path: Path to the Kuzu database
    """
    logger.info(f"Building communities for database: {db_path}")

    # Ensure FTS extension is loaded
    logger.info("Ensuring FTS extension is loaded...")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    try:
        conn.execute("LOAD FTS")
        logger.info("FTS extension loaded")
    except Exception as e:
        logger.warning(f"FTS extension may already be loaded: {e}")

    # Initialize Graphiti
    kuzu_driver = KuzuDriver(db=db_path)
    graphiti = Graphiti(graph_driver=kuzu_driver)

    try:
        print("\n" + "=" * 60)
        print("Building Communities")
        print("=" * 60)
        print("This will:")
        print("  • Remove any existing communities")
        print("  • Analyze entity connections using Leiden algorithm")
        print("  • Create new community groupings")
        print("  • Generate summaries for each community")
        print("=" * 60 + "\n")

        logger.info("Starting community building...")
        await graphiti.build_communities()
        logger.info("Communities built successfully")

        print("\n" + "=" * 60)
        print("✓ Community building complete!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Error building communities: {e}")
        raise
    finally:
        await graphiti.close()
        logger.info("Graphiti connection closed")


#################################################
# MAIN
#################################################
def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Load journal entries into Charlie's knowledge graph"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Load command
    load_parser = subparsers.add_parser(
        "load", help="Load journal entries into the graph"
    )
    load_parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Load all entries without pausing for verification between entries",
    )
    load_parser.add_argument(
        "--journal-file",
        default=settings.JOURNAL_FILE_PATH,
        help=f"Path to journal JSON file (default: {settings.JOURNAL_FILE_PATH})",
    )
    load_parser.add_argument(
        "--db-path",
        default=settings.DB_PATH,
        help=f"Path to Kuzu database (default: {settings.DB_PATH})",
    )

    # Build communities command
    communities_parser = subparsers.add_parser(
        "build-communities", help="Build entity communities using Leiden algorithm"
    )
    communities_parser.add_argument(
        "--db-path",
        default=settings.DB_PATH,
        help=f"Path to Kuzu database (default: {settings.DB_PATH})",
    )

    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return

    # Load environment variables
    load_dotenv()

    try:
        if args.command == "load":
            print("\n" + "=" * 60)
            print("Charlie Journal Loader")
            print("=" * 60)
            print(f"Journal file: {args.journal_file}")
            print(f"Database: {args.db_path}")
            print(
                f"Verification mode: {'DISABLED' if args.skip_verification else 'ENABLED'}"
            )
            print("=" * 60 + "\n")

            # Setup FTS extension
            setup_fts_extension(args.db_path)

            # Load journals
            asyncio.run(
                load_journals(
                    journal_file_path=args.journal_file,
                    db_path=args.db_path,
                    skip_verification=args.skip_verification,
                )
            )

        elif args.command == "build-communities":
            print("\n" + "=" * 60)
            print("Charlie Community Builder")
            print("=" * 60)
            print(f"Database: {args.db_path}")
            print("=" * 60 + "\n")

            # Build communities
            asyncio.run(build_communities(db_path=args.db_path))

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
