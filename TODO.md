## Technical
- ADD ORGS & LOC INTO NODE EXTRACTOR!
- Would entity extraction work better by summarizing the text and extracting that?
- Check if you can use MLX batch()
- Evaluate quantized ONNX models after collecting sample data - test if smaller versions (int8: 63MB, q4: 112MB) are sufficient vs full model (249MB)
- Constrain extraction prompts significantly, focus only on people first.
- Implement [Contextual Retreival](https://www.anthropic.com/engineering/contextual-retrieval)
- Use [SBERT](https://sbert.net/docs/quickstart.html) with embedding & reranking/cross-encoder models
  - https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ
  - https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls (Requires mlx-lm conversion!)

## Non-Technical

- LICENSE??
- Actual trademarkable project name (talk to Joe about this, maybe Andrew M too)

## Nice-to-haves

- Upgrade to python 3.14
- [Count input tokens properly](https://www.perplexity.ai/search/does-qwen3-use-tiktoken-for-ca-APRBFEnXRPawkHgtv5EryQ#0)
- Might have to use all-LLM entity extraction for non English/German due to Distilbert-NER only being trained on those two
- Confirm facts/extractions with "Let's reflect...., you felt ___?". Humanize and use mirrored reflection from psychology to clean up data
