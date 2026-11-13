# Woof LLM Streaming

Streams (or outputs) speech from a local Underdog Woof TTS model in response to OpenAI-compatible requests.

## Setup

First, clone this repository.

It's recommended to use a virtual environment, like `uv` but you can also manually set it up.

### UV

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
source .venv/bin/activate
hf download ConwayResearch/Underdog-Woof --local-dir underdog-woof
```

### Manual

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the Underdog Woof model:

```bash
hf download ConwayResearch/Underdog-Woof --local-dir underdog-woof
```

If you don't have `hf` installed, you can install it with:

```bash
pip install hf
```

## Usage

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` in `.env`, then run:

```bash
woof-llm-streaming "Tell me a short joke." --output response.wav
```

(or `python main.py "Tell me a short joke." --output response.wav` if you don't have the script installed)

Use `--voice mira` to select the Mira voice, or `--no-play` to save audio without playing it.
