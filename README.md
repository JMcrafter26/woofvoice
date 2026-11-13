# Woof LLM Streaming

Streams (or outputs) speech from a local Underdog Woof TTS model in response to OpenAI-compatible, Ollama, or Anthropic requests.

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

Set the provider settings in `.env`, then run:

```bash
woof-llm-streaming "Tell me a short joke." --output response.wav
```

(or `python main.py "Tell me a short joke." --output response.wav` if you don't have the script installed)

Select a provider explicitly when needed:

```bash
woof-llm-streaming --provider ollama --model llama3.2 "Tell me a short joke."
woof-llm-streaming --provider anthropic --model claude-sonnet-4-5 "Tell me a short joke."
```

Use `--voice mira` to select the Mira voice, or `--no-play` to save audio without playing it.

To route live audio to VB-CABLE or another virtual device, list available outputs:

```bash
woof-llm-streaming --list-audio-devices
woof-llm-streaming --audio-device "VB-CABLE" "Tell me a short joke."
```

You can also set `AUDIO_DEVICE` in `.env`. Use the device name or numeric index shown by
`--list-audio-devices`; the WAV file is still saved normally.

## Use as a library

```python
from underdog_woof import Woof
from woof_tts import OllamaChatBackend, stream_and_speak

backend = OllamaChatBackend("http://localhost:11434", "llama3.2")
stream_and_speak(
    backend.stream([{"role": "user", "content": "Hello"}]),
    Woof(voice="juno"),
    "response.wav",
    audio_device="VB-CABLE",
)
```
