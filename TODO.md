# Technical

- Full graph traversal support first
- Full summarization viewer next, all summary levels need to be clear
- Cleanup `kuzu_service.py` errors

- Manual import UI, full debug of extraction before loading
- Figure out how to test each prompt extraction manually
- FTS bar in header
- Add graph search query support
- Check if viewing old posts filters everything by time of post.
- Figure out how to do time-series rollups

## Questions

- Git-submodule graphiti or use as lib?? How much control does the lib give you?

## Non-Technical

- LICENSE??
- Actual trademarkable project name (talk to Joe about this, maybe Andrew M too)
- Put in a real repo

## Local Model Support

- CRITICAL: Need to add `outlines` support for JSON constraints
  - Need to add `LM Studio` support to `outlines` first, or go with MLX-LM directly
    <https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221-2HFnIWuyIw7fUZOrF6JJCeF9gm_V0_E%22%5D,%22action%22:%22open%22,%22userId%22:%22116565776749566603200%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing>
- CRITICAL: DSPy
- Constrain extraction prompts significantly, focus only on people first.
- Go straight to MLX-LM and configure everything in code early?
- Need to fix janky Kuzu timezone/datetime support, or keep using monkey-patches

## Nice-to-haves

- Constrain communities to clusters of 5+
- Use journal-centric language instead of EpisodicNodes
- Upgrade to python 3.14
- Load via .zip instead of .json
