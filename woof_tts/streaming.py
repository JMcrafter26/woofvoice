"""Provider-neutral text-to-speech streaming pipeline."""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextToSpeech(Protocol):
    sr: int

    def __call__(self, text: str, seed: int | None = None) -> np.ndarray:
        ...

    @staticmethod
    def save(path: str | Path, wav: np.ndarray) -> None:
        ...


@dataclass(frozen=True)
class AudioChunkEvent:
    """Metadata for a newly-created external audio chunk."""

    run_id: str
    index: int
    text: str
    path: Path


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
    tokens: Iterator[str],
    tts: TextToSpeech,
    output_path: str | Path,
    seed: int | None = None,
    play: bool = True,
    audio_device: int | str | None = None,
    max_chars: int = 240,
    on_text: Callable[[str], None] | None = None,
    chunk_dir: str | Path | None = None,
    on_chunk: Callable[[AudioChunkEvent], None] | None = None,
) -> str:
    """Convert a token stream to pipelined audio and return the full response text.

    `audio_device` may be a sounddevice index or name, such as a VB-CABLE output.
    When `chunk_dir` is set, each generated segment is saved there immediately as
    an atomically-created WAV file for external playback.
    `on_chunk` is called after each chunk file is atomically renamed and
    receives the spoken section text and its sequence number.
    """
    text_queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    generated: list[np.ndarray] = []
    errors: list[BaseException] = []
    chunk_output_dir = Path(chunk_dir) if chunk_dir else None
    run_id = uuid.uuid4().hex[:12]

    def generate_audio() -> None:
        try:
            first_chunk = True
            chunk_index = 0
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
                if chunk_output_dir:
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                    chunk_index += 1
                    text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                    chunk_path = chunk_output_dir / f"{run_id}-{chunk_index:06d}-{text_hash}.wav"
                    temporary_path = chunk_path.with_suffix(".tmp")
                    tts.save(temporary_path, audio)
                    temporary_path.replace(chunk_path)
                    if on_chunk:
                        on_chunk(AudioChunkEvent(run_id, chunk_index, chunk, chunk_path))
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

            with sd.OutputStream(
                samplerate=tts.sr,
                channels=1,
                dtype="float32",
                device=audio_device,
            ) as stream:
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
                interrupted = True
                stop_event.set()

    try:
        for chunk in speakable_chunks(tokens, max_chars=max_chars):
            if stop_event.is_set():
                break
            response_chunks.append(chunk)
            if on_text:
                on_text(chunk)
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
                    interrupted = True
                    stop_event.set()
        join_worker(generator)
        if player:
            join_worker(player)

    if errors:
        raise RuntimeError("Streaming voice pipeline failed") from errors[0]
    if interrupted:
        return " ".join(response_chunks)
    if generated:
        tts.save(output_path, np.concatenate(generated))
    return " ".join(response_chunks)
