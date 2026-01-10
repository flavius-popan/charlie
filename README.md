# Charlie 🧶

The most over-engineered journaling app in the world.

![[home_screen.png]]

Charlie is an experimental, local-first journal that synthesizes a knowledge graph of your memories for you to explore. Write naturally about your life, and a local LLM finds and connects people, places, groups, and activities so you can explore your memories through connections between concepts. It automatically turns your freeform journal entries into Wikipedia-like pages full of links to click on and dig deeper.

That's the idea at least, it's kinda half-baked at the moment.

![[pepe_silva.png]]

This ain't another RAG project; it's way more convoluted and less effective *BUT* it has potential. Here's how it currently works:

1. You write a journal entry (or import a bunch of old ones).
2. A local LLM (Qwen3-4B-Instruct to be exact) will read it, then extract "connections" (people, groups, places, and activities) that you can click on, or see a sidebar pane of.
![[reader.png]]
3. Clicking a connection shows direct quotes, related connections, dates mentioned, and a few other tidbits of information to give you a higher-level overview of the entity.
![[browser.png]]
4. That's all it does. For now.

The extraction isn't perfect and it's slow, but the foundations work. Relationship extraction between entities (Alice → Knows → Bob) is planned but not implemented yet. 

## Architecture

**Everything runs in a single Python process. No external dependencies. No cloud. No containers.**

![[architecture.png]]

- **Embedded graph database** (FalkorDB Lite) - Redis + OpenCypher queries, fully in-process
- **Local LLM inference** (llama.cpp) - auto-downloads model (~4.5GB), no Ollama/LM Studio needed
- **In-process task queue** (Huey) - background extraction without external Redis/Celery
- **Terminal UI** (Textual) - keyboard-driven, SSH-friendly
- **Memory efficient** - runs a quantized 4B model (Q8_0) in under 900MB of RAM (see [backend/settings.py](backend/settings.py) for config)

## Genetic Prompt Optimization

Instead of writing prompts by hand like a pleb from 2023, ~~I over-engineered the hell out of this so I could justify paying $100/mo for a Claude Pro plan~~ I used the most advanced technology available in the Fall of 2025 to design an automated pipeline that uses genetic algorithms to find the best prompt for the task at hand. Here's how it currently works:

![[GEPA.png]]

1. I created a "gold standard" set of [examples showing ideal entity extraction](backend/optimizers/data/extract_nodes_examples.json) to feed into a [local optimizer script](backend/optimizers/extract_nodes_optimizer.py).
2. On HuggingFace's Hosted Inference service, I spin up the same model (Qwen3-4B-Instruct) on an L4 GPU. This gets around llama.cpp's serialization limit and lets me run 20-30 evaluations in parallel instead of waiting hours for serial local inference.
3. DSPy's GEPA optimizer orchestrates two models working together over ~128 iterations:
   - **Qwen3-4B** (on HuggingFace GPU): Does the actual entity extraction with different prompt variants
   - **GPT-4o-mini** (OpenAI API): Acts as the "coach" - analyzes what Qwen3 got wrong, then rewrites the prompt to fix those mistakes

   It's like evolution - bad prompts die, good ones survive and get mutated by GPT-4o-mini based on performance feedback.
4. After an hour of automated trial-and-error, GEPA spits out the best-performing prompt which gets saved to [`backend/prompts/extract_nodes.json`](backend/prompts/extract_nodes.json) and loaded at runtime.

For just a few dollars, this pipeline spits out prompts that work way better than anything I'd write manually, optimized specifically for extracting entities from journal entries. Neat huh? See [`backend/optimizers/README.md`](backend/optimizers/README.md) for the implementation details.

### Smart Model Management

Charlie waits for you to finish writing before loading the model in memory to keep things snappy. When all the work is finished, it unloads the model to avoid hogging up all your precious RAM. All the orchestration is handled for you, including queuing entries, processing your most recent entry before tending to the backlog of jobs, and many other conveniences. **Just focus on writing and exploring, everything else is automated.**

## Quick Start

**System Requirements:**
- Linux (x86_64, glibc 2.39+) — check with `ldd --version`
- macOS 15+ (Sequoia) on Apple Silicon (M-series)

```bash
./setup.sh
```

The setup script handles everything:
- Installs `uv` package manager (if needed)
- Installs Python 3.12+ (if needed)
- Creates virtual environment and installs dependencies
- Downloads the language model (~4.5GB, cached)
- Initializes the database
- Offers to import existing journals

## Importing Journals

Setup prompts you to import, or run importers manually:

```bash
# Day One export
python importers/dayone.py ~/Downloads/DayOneExport.zip

# Text files directory
python importers/files.py ~/journals/ --recursive

# Dry run (preview without importing)
python importers/files.py ~/journals/ --dry-run
```

See [importers/IMPORTING.md](importers/IMPORTING.md) for all options.

## Running

```bash
./charlie.sh
```

Or manually activate the virtual environment:

```bash
source .venv/bin/activate
python charlie.py
```
