from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# Configure logging
logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Kuzu connection using KuzuDriver
kuzu_driver = KuzuDriver(db="brain/charlie.kuzu")

# Configure LM Studio client
llm_config = LLMConfig(
    api_key="abc",  # LM Studio doesn't require a real API key
    model="qwen/qwen3-30b-a3b-2507",
    small_model="qwen/qwen3-4b-2507",
    base_url="http://127.0.0.1:1234/v1",
)

llm_client = OpenAIClient(config=llm_config)

graphiti = Graphiti(
    graph_driver=kuzu_driver,
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="abc",
            embedding_model="text-embedding-nomic-embed-text-v1.5@f32",
            embedding_dim=768,
            base_url="http://127.0.0.1:1234/v1",
        )
    ),
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
)


async def main():
    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                "content": "Kamala Harris is the Attorney General of California. She was previously "
                "the district attorney for San Francisco.",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
                "type": EpisodeType.text,
                "description": "podcast transcript",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "state": "California",
                    "previous_role": "Lieutenant Governor",
                    "previous_location": "San Francisco",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
            {
                "content": {
                    "name": "Gavin Newsom",
                    "position": "Governor",
                    "term_start": "January 7, 2019",
                    "term_end": "Present",
                },
                "type": EpisodeType.json,
                "description": "podcast metadata",
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f"Freakonomics Radio {i}",
                episode_body=episode["content"]
                if isinstance(episode["content"], str)
                else json.dumps(episode["content"]),
                source=episode["type"],
                source_description=episode["description"],
                reference_time=datetime.now(timezone.utc),
            )
            print(f"Added episode: Freakonomics Radio {i} ({episode['type'].value})")

    finally:
        # Close the connection
        await graphiti.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(main())
