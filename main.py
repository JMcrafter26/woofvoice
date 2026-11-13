"""Stream an OpenAI-compatible response through local Woof TTS.

The chat response, speech generation, playback, and WAV saving are pipelined:
the model can keep receiving tokens while earlier sentences are being spoken.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shlex
import sys
import threading
import unicodedata
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "underdog-woof"))

from underdog_woof import Woof


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


class OpenAIChatBackend:
    """Small dependency-free client for an OpenAI-style streaming endpoint."""

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
        ).encode()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")

        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid SSE event from chat backend: {data!r}") from exc
                choices = event.get("choices")
                if choices and choices[0].get("delta", {}).get("content"):
                    yield choices[0]["delta"]["content"]


def speakable_chunks(tokens: Iterator[str], max_chars: int = 240) -> Iterator[str]:
    """Yield short sentence-like chunks without waiting for the whole answer."""
    pending = ""
    for token in tokens:
        pending += token
        while True:
            boundary = next(
                (
                    index + 1
                    for index, character in enumerate(pending)
                    if character in ".!?\n" and (index + 1 == len(pending) or pending[index + 1].isspace())
                ),
                None,
            )
            if boundary is None and len(pending) < max_chars:
                break
            if boundary is None:
                boundary = pending.rfind(" ", 0, max_chars)
                if boundary <= 0:
                    boundary = max_chars
            chunk, pending = pending[:boundary].strip(), pending[boundary:].lstrip()
            if chunk:
                yield chunk
    if pending.strip():
        yield pending.strip()


def clean_for_speech(text: str) -> str:
    """Remove presentation markup and symbols that should not be read aloud."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```[\w+-]*", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", "", text)
    text = re.sub(r"[*_~`]+", "", text)
    text = re.sub(r"<[^>\n]+>", "", text)

    text = "".join(
        character
        for character in text
        if unicodedata.category(character) not in {"So", "Sk", "Sm", "Cf"}
        and not 0xFE00 <= ord(character) <= 0xFE0F
    )
    return re.sub(r"\s+", " ", text).strip()


def stream_and_speak(
    backend: OpenAIChatBackend,
    messages: list[dict[str, str]],
    tts: Woof,
    output_path: str,
    seed: int | None,
    play: bool,
) -> str:
    text_queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    generated: list[np.ndarray] = []
    errors: list[BaseException] = []

    def generate_audio() -> None:
        try:
            first_chunk = True
            while not stop_event.is_set():
                try:
                    chunk = text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                audio = tts(chunk, seed=seed if first_chunk else None)
                first_chunk = False
                generated.append(audio)
                if play:
                    while not stop_event.is_set():
                        try:
                            audio_queue.put(audio, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:
            errors.append(exc)
        finally:
            if play:
                while not stop_event.is_set():
                    try:
                        audio_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def play_audio() -> None:
        try:
            import sounddevice as sd

            with sd.OutputStream(samplerate=tts.sr, channels=1, dtype="float32") as stream:
                while not stop_event.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if audio is None:
                        break
                    for start in range(0, len(audio), 1024):
                        if stop_event.is_set():
                            break
                        stream.write(audio[start : start + 1024].reshape(-1, 1))
        except BaseException as exc:
            errors.append(exc)

    generator = threading.Thread(target=generate_audio, name="tts-generator")
    player = threading.Thread(target=play_audio, name="audio-player") if play else None
    generator.start()
    if player:
        player.start()

    response_chunks: list[str] = []
    interrupted = False

    def join_worker(worker: threading.Thread) -> None:
        nonlocal interrupted
        while worker.is_alive():
            try:
                worker.join(timeout=0.1)
            except KeyboardInterrupt:
                if not interrupted:
                    print("\nInterrupted.")
                interrupted = True
                stop_event.set()

    try:
        for chunk in speakable_chunks(backend.stream(messages)):
            if stop_event.is_set():
                break
            print(chunk, flush=True)
            response_chunks.append(chunk)
            spoken_chunk = clean_for_speech(chunk)
            if spoken_chunk:
                while not stop_event.is_set():
                    try:
                        text_queue.put(spoken_chunk, timeout=0.1)
                        break
                    except queue.Full:
                        continue
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        print("\nInterrupted.")
    finally:
        if interrupted:
            stop_event.set()
        else:
            while not interrupted:
                try:
                    text_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
                except KeyboardInterrupt:
                    print("\nInterrupted.")
                    interrupted = True
                    stop_event.set()
        join_worker(generator)
        if player:
            join_worker(player)

    if errors:
        raise RuntimeError("Streaming voice pipeline failed") from errors[0]
    if interrupted:
        return " ".join(response_chunks)
    if not generated:
        print("\nNo speakable audio was generated for this response.")
        return " ".join(response_chunks)
    tts.save(output_path, np.concatenate(generated))
    print(f"\nSaved audio to {output_path}")
    return " ".join(response_chunks)


def main() -> None:
    config = load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the chat backend")
    parser.add_argument("--base-url", default=config.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=config.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default=config.get("OPENAI_MODEL", "custom"))
    parser.add_argument("--voice", choices=("juno", "mira"), default="juno")
    parser.add_argument("--output", default="response.wav")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-play", action="store_true", help="Save the WAV without playing it")
    args = parser.parse_args()

    backend = OpenAIChatBackend(args.base_url, args.api_key, args.model)
    tts = Woof(voice=args.voice)
    if args.prompt:
        stream_and_speak(
            backend,
            [{"role": "user", "content": args.prompt}],
            tts,
            args.output,
            args.seed,
            not args.no_play,
        )
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
        response = stream_and_speak(backend, messages, tts, output_path, args.seed, not args.no_play)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
