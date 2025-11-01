"""Audio transcription helpers using the ElevenLabs API."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import sys

import requests

# Handle both relative imports (when used as module) and direct execution
try:
    from .config import Config
    from .downloader import VideoMetadata
    from .utils import save_json
except ImportError:
    # Running as a script, set up path for absolute imports
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from dev.config import Config
    from dev.downloader import VideoMetadata
    from dev.utils import save_json


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
    try:
        from .utils import load_json
    except ImportError:
        from dev.utils import load_json

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


if __name__ == "__main__":
    # ==== TEST CONFIGURATION ====
    # Put your audio file path here
    TEST_AUDIO_PATH = "test_downloads/2025-11-01_growing-up-there-was-one-rule-never-open-the-door-in-the-basement.m4a"
    # ============================
    
    audio_path = Path(TEST_AUDIO_PATH)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    print(f"Testing transcription with: {audio_path.name}")
    print("Loading config...")
    
    # Load config from environment variables
    config = Config.from_env()
    
    # Create minimal video metadata for testing
    video_metadata = VideoMetadata(
        video_id="test_id",
        title="Test Audio",
        upload_date="2025-11-01",
        url="https://example.com/test",
        duration=None,
    )
    
    print("Sending audio to ElevenLabs for transcription...")
    print("This may take a while depending on audio length...")
    
    try:
        transcript = transcribe_audio(audio_path, video_metadata, config)
        
        print("\n" + "="*50)
        print("TRANSCRIPTION SUCCESSFUL!")
        print("="*50)
        print(f"\nTitle: {transcript.title}")
        print(f"Duration: {transcript.duration_sec} seconds")
        print(f"Number of segments: {len(transcript.segments)}")
        print(f"\nFull text length: {len(transcript.full_text)} characters")
        print(f"\nFirst 500 characters of transcript:\n{transcript.full_text[:500]}...")
        
        if transcript.segments:
            print(f"\nFirst segment:")
            seg = transcript.segments[0]
            print(f"  Time: {seg.start:.2f}s - {seg.end:.2f}s")
            print(f"  Text: {seg.text}")
        
        # Optionally save the transcript
        output_path = audio_path.parent / f"{audio_path.stem}_transcript.json"
        save_transcript(transcript, output_path)
        print(f"\nTranscript saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)