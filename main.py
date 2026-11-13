"""Stream an OpenAI-compatible response through local Woof TTS."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "underdog-woof"))

from underdog_woof import Woof
from woof_tts import (
    AnthropicChatBackend,
    ChatBackend,
    OllamaChatBackend,
    OpenAIChatBackend,
    stream_and_speak,
)


def load_dotenv(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE settings from a dotenv file."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return {}

    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid .env entry on line {line_number}: {line!r}")
        try:
            parsed = shlex.split(raw_value, comments=True)
        except ValueError as exc:
            raise ValueError(f"Invalid .env value on line {line_number}") from exc
        values[key.strip()] = " ".join(parsed)
    return values


def create_backend(args: argparse.Namespace, config: dict[str, str]) -> ChatBackend:
    """Create the selected provider client from CLI arguments and environment settings."""
    provider = args.provider
    if provider == "ollama":
        base_url = args.base_url or config.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaChatBackend(base_url, args.model or config.get("OLLAMA_MODEL", "llama3.2"))
    if provider == "anthropic":
        api_key = args.api_key or config.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic requires --api-key or ANTHROPIC_API_KEY in .env")
        base_url = args.base_url or config.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = args.model or config.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        return AnthropicChatBackend(api_key, model, base_url, args.max_tokens)
    base_url = args.base_url or config.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model = args.model or config.get("OPENAI_MODEL", "custom")
    return OpenAIChatBackend(base_url, args.api_key or config.get("OPENAI_API_KEY"), model)


def parse_audio_device(value: str | None) -> int | str | None:
    """Accept either a sounddevice device index or a name."""
    if not value:
        return None
    return int(value) if value.isdecimal() else value


def main() -> None:
    config = load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the chat backend")
    parser.add_argument("--provider", choices=("openai", "ollama", "anthropic"), default=config.get("CHAT_PROVIDER", "openai"))
    parser.add_argument("--base-url", help="Provider API base URL")
    parser.add_argument("--api-key", help="Provider API key")
    parser.add_argument("--model", help="Provider model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum response tokens for Anthropic")
    parser.add_argument("--voice", choices=("juno", "mira"), default="juno")
    parser.add_argument("--output", default="response.wav")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--audio-device", help="Output device name or index, for example VB-CABLE")
    parser.add_argument("--list-audio-devices", action="store_true", help="List available audio output devices and exit")
    parser.add_argument("--chunk-dir", help="Write each spoken chunk as a WAV file in this directory")
    parser.add_argument("--no-play", action="store_true", help="Save the WAV without playing it")
    args = parser.parse_args()

    if args.list_audio_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    backend = create_backend(args, config)
    tts = Woof(voice=args.voice)
    audio_device = parse_audio_device(args.audio_device or config.get("AUDIO_DEVICE"))
    chunk_dir = args.chunk_dir or config.get("AUDIO_CHUNK_DIR")

    def handle_response(messages: list[dict[str, str]], output_path: str) -> str:
        response = stream_and_speak(
            backend.stream(messages),
            tts,
            output_path,
            seed=args.seed,
            play=not args.no_play,
            audio_device=audio_device,
            on_text=lambda chunk: print(chunk, flush=True),
            chunk_dir=chunk_dir,
        )
        if response:
            print(f"\nSaved audio to {output_path}")
        return response

    if args.prompt:
        handle_response([{"role": "user", "content": args.prompt}], args.output)
        return

    messages: list[dict[str, str]] = []
    print("Interactive chat. Type 'exit' or press Ctrl-D to quit.")
    turn = 0
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": prompt})
        turn += 1
        output_path = str(Path(args.output).with_name(f"{Path(args.output).stem}-{turn}{Path(args.output).suffix}"))
        response = handle_response(messages, output_path)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
