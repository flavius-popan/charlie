#!/usr/bin/env python3
"""
Charlie - Interactive Graph Explorer Startup Script

Run this script to start the FastAPI development server.
"""

import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_database():
    """Check if the Kuzu database exists."""
    from app import settings

    if not os.path.exists(settings.DB_PATH):
        logger.warning(f"Database not found at {settings.DB_PATH}")
        logger.info("You may need to load journal entries first:")
        logger.info("  python load_journals.py load --skip-verification")
        return False
    return True


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Charlie - Interactive Graph Explorer"
    )
    parser.add_argument(
        "--db",
        default="charlie.kuzu",
        help="Database filename in brain/ directory (default: charlie.kuzu)",
    )
    args = parser.parse_args()

    # Configure database path
    from app import settings

    settings.DB_PATH = f"{settings.BRAIN_DIR}/{args.db}"

    logger.info("=" * 60)
    logger.info("Charlie - Interactive Graph Explorer")
    logger.info("=" * 60)
    logger.info(f"Database: {settings.DB_PATH}")

    # Check database
    check_database()

    # Start the server
    logger.info("\nStarting development server...")
    logger.info("Server will be available at: http://localhost:8080")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60 + "\n")

    try:
        import uvicorn

        uvicorn.run(
            "app.main:app", host="0.0.0.0", port=8080, reload=True, log_level="info"
        )
    except ImportError:
        logger.error("uvicorn not installed. Install dependencies with:")
        logger.error("  uv pip install -e .")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
