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

To expose each spoken segment as soon as it is generated, write chunks to a directory:

```bash
woof-llm-streaming --chunk-dir ./audio-chunks "Tell me a short joke."
```

Each file is a complete WAV named with a unique run ID, sequence number, and text hash,
for example `a1b2c3d4e5f6-000001-9f2a1c4d8e77.wav`. An external player can watch this
directory and play files in sequence. Files are written atomically, so only complete
audio files appear with the `.wav` extension. The same option is available through
`AUDIO_CHUNK_DIR` in `.env` and as `chunk_dir` in `stream_and_speak()`.

For in-process notifications, pass `on_chunk`. It receives an `AudioChunkEvent`
after the file is ready, including `run_id`, `index`, `text`, and `path`:

```python
from woof_tts import AudioChunkEvent

def new_chunk(event: AudioChunkEvent) -> None:
    print(f"Play section {event.index}: {event.text} ({event.path})")

stream_and_speak(tokens, tts, "response.wav", chunk_dir="audio-chunks", on_chunk=new_chunk)
```

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
    chunk_dir="audio-chunks",
)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

NOTE: This license does not cover the Underdog Woof model and/or their content, which is subject to its own license. Please refer to the model's license for usage terms.

Woof is free for non-commercial use, with credit. Commercial use needs a license. The full terms are the
[Underdog Non-Commercial License 1.0](https://huggingface.co/ConwayResearch/Underdog-Woof/blob/main/LICENSE).

The audio decoder is adapted from Kyutai's Mimi, and the training data includes LibriTTS-R and Emilia-YODAS, all CC-BY
4.0. The attributions are in [NOTICE](https://huggingface.co/ConwayResearch/Underdog-Woof/blob/main/NOTICE).
