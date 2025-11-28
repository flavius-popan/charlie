# Stability Fixes
- Any other systems in need of an FSM for safety?
- BACKUPS! Need to think of ways to always have backups.

## Minor but important

- Remove statemachine visualization dep and requirements to refresh. KISS.
- Automate test runs in CI
- Hide "d/up/down" labels in connections pane if no connections found
- Improve readability/UI when no connections found/other valid statuses.
- Add wc/token/journal metrics at bottom of md viewer

## Technical
- Discover relative time references and auto-map them to time-bounded queries
- Add toggle for switching between perma-ban/journal delete for entities
- Implement [Contextual Retreival](https://www.anthropic.com/engineering/contextual-retrieval)
- Use [SBERT](https://sbert.net/docs/quickstart.html) with embedding & reranking/cross-encoder models
  - https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ
  - https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
- Handle new model downloads to avoid blocking TUI

## Non-Technical

- LICENSE??

## Nice-to-haves

- [Count input tokens properly](https://www.perplexity.ai/search/does-qwen3-use-tiktoken-for-ca-APRBFEnXRPawkHgtv5EryQ#0)
