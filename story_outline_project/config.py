"""Configuration management for the story outline pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv


@dataclass
class Config:
    """Runtime configuration for the pipeline."""

    channel_url: str
    eleven_api_key: str
    eleven_model_id: str
    openai_api_key: str
    openai_model: str
    num_videos: int
    batch_size: int
    temp_audio_dir: Path
    transcript_dir: Path
    outline_dir: Path
    merged_dir: Path
    video_index_path: Path
    logs_dir: Path
    log_file: Path
    request_timeout: int = 600

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration values from environment variables and defaults."""
        load_dotenv()

        project_root = Path(__file__).resolve().parent
        data_dir = project_root / "data"
        logs_dir = project_root / "logs"

        channel_url = os.getenv("YOUTUBE_CHANNEL_URL", "")
        if not channel_url:
            raise ValueError("YOUTUBE_CHANNEL_URL environment variable is required")

        eleven_api_key = os.getenv("ELEVEN_API_KEY", "")
        if not eleven_api_key:
            raise ValueError("ELEVEN_API_KEY environment variable is required")

        eleven_model_id = os.getenv("ELEVEN_MODEL_ID", "eleven_monolingual_v1")

        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        num_videos = int(os.getenv("NUM_VIDEOS", "50"))
        batch_size = int(os.getenv("BATCH_SIZE", "5"))

        temp_audio_dir = data_dir / "tmp_audio"
        transcript_dir = data_dir / "transcripts"
        outline_dir = data_dir / "outlines"
        merged_dir = data_dir / "merged"
        video_index_path = data_dir / "video_index.json"
        log_file = logs_dir / "pipeline.log"

        return cls(
            channel_url=channel_url,
            eleven_api_key=eleven_api_key,
            eleven_model_id=eleven_model_id,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            num_videos=num_videos,
            batch_size=batch_size,
            temp_audio_dir=temp_audio_dir,
            transcript_dir=transcript_dir,
            outline_dir=outline_dir,
            merged_dir=merged_dir,
            video_index_path=video_index_path,
            logs_dir=logs_dir,
            log_file=log_file,
        )

    def ensure_directories(self) -> None:
        """Ensure all runtime directories exist."""
        for path in self.directories:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def directories(self) -> List[Path]:
        """Return all directories managed by the configuration."""
        return [
            self.temp_audio_dir,
            self.transcript_dir,
            self.outline_dir,
            self.merged_dir,
            self.logs_dir,
        ]
