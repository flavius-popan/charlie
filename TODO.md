# NEXT STEPS

- Extremely detailed debugging in load_journals.py
- Expose all prompts for customization
- Remove janky Kuzu timezone/datetime monkey-patch
- Move models to default location, out of project dir
- Is batch generation actually being done right?
See https://github.com/ml-explore/mlx-lm/blob/367d6d76860499767f62b0bc34408b51c9ed916b/mlx_lm/examples/batch_generate_response.py
- Enable prompt caching? 
- Check if Qwen models need `eos_token` application and trust_remote_code

## Technical

- Full graph traversal support first
- Full summarization viewer next, all summary levels need to be clear
- Cleanup `kuzu_service.py` errors

- Manual import UI, full debug of extraction before loading
- Figure out how to test each prompt extraction manually
- FTS bar in header
- Add graph search query support
- Check if viewing old posts filters everything by time of post.
- Figure out how to do time-series rollups

## Non-Technical

- LICENSE??
- Actual trademarkable project name (talk to Joe about this, maybe Andrew M too)
- Put in a real repo

## Local Model Support

- CRITICAL: DSPy
- Constrain extraction prompts significantly, focus only on people first.

## Nice-to-haves

- Constrain communities to clusters of 5+
- Use journal-centric language instead of EpisodicNodes
- Upgrade to python 3.14
- Load via .zip instead of .json
