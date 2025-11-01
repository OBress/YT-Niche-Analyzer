"""Audio transcription helpers using the ElevenLabs API."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests

from .config import Config
from .downloader import VideoMetadata
from .utils import save_json


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    title: str
    upload_date: str
    duration_sec: Optional[int]
    segments: List[TranscriptSegment]
    full_text: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "upload_date": self.upload_date,
            "duration_sec": self.duration_sec,
            "segments": [segment.to_dict() for segment in self.segments],
            "full_text": self.full_text,
        }


ELEVEN_TRANSCRIPTION_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def transcribe_audio(
    audio_path: Path,
    video: VideoMetadata,
    config: Config,
    *,
    session: Optional[requests.Session] = None,
) -> Transcript:
    """Send *audio_path* to ElevenLabs for transcription."""
    sess = session or requests.Session()
    headers = {
        "xi-api-key": config.eleven_api_key,
    }
    data = {
        "model_id": config.eleven_model_id,
        "timestamps": "true",
        "response_format": "verbose_json",
    }

    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, "audio/mpeg")}
        response = sess.post(
            ELEVEN_TRANSCRIPTION_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=config.request_timeout,
        )
    response.raise_for_status()
    payload = response.json()

    segments = [
        TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=segment.get("text", "").strip(),
        )
        for segment in payload.get("segments", [])
    ]
    full_text = payload.get("text") or " ".join(segment.text for segment in segments)

    transcript = Transcript(
        title=video.title,
        upload_date=video.upload_date,
        duration_sec=video.duration,
        segments=segments,
        full_text=full_text.strip(),
    )
    return transcript


def save_transcript(transcript: Transcript, destination: Path) -> None:
    """Persist a transcript as JSON."""
    save_json(destination, transcript.to_dict())


def load_transcript(path: Path) -> Transcript:
    """Load a transcript from disk."""
    from .utils import load_json

    payload = load_json(path)
    segments = [
        TranscriptSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")),
        )
        for seg in payload.get("segments", [])
    ]
    return Transcript(
        title=payload.get("title", ""),
        upload_date=payload.get("upload_date", ""),
        duration_sec=payload.get("duration_sec"),
        segments=segments,
        full_text=payload.get("full_text", ""),
    )
