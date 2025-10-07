from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# Kuzu connection using KuzuDriver
kuzu_driver = KuzuDriver(db="brain/charlie.kuzu")

# Configure LM Studio client
llm_config = LLMConfig(
    api_key="abc",  # LM Studio doesn't require a real API key
    model="qwen/qwen3-30b-a3b-2507",
    small_model="qwen/qwen3-4b-2507",
    base_url="http://127.0.0.1:1234",
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
            base_url="http://127.0.0.1:1234",
        )
    ),
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
)
