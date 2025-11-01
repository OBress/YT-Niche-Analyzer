"""Video metadata retrieval and audio download utilities."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from yt_dlp import YoutubeDL

from .utils import save_json, slugify


@dataclass
class VideoMetadata:
    """Metadata describing a single YouTube video."""

    video_id: str
    title: str
    upload_date: str
    url: str
    duration: int | None = None

    @property
    def slug(self) -> str:
        return slugify(f"{self.upload_date}_{self.title}")

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_date(raw_date: str) -> str:
    return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")


def fetch_channel_videos(channel_url: str, limit: int) -> List[VideoMetadata]:
    """Return metadata for the most recent *limit* videos on a channel."""
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "extract_flat": "in_playlist",
    }
    videos: List[VideoMetadata] = []
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        entries: Iterable[dict] = info.get("entries", [])
        sorted_entries = sorted(
            [entry for entry in entries if entry.get("upload_date")],
            key=lambda item: item.get("upload_date"),
            reverse=True,
        )
        for entry in sorted_entries:
            if len(videos) >= limit:
                break
            upload_date = entry.get("upload_date")
            video = VideoMetadata(
                video_id=entry.get("id", ""),
                title=entry.get("title", "Untitled"),
                upload_date=_normalize_date(upload_date),
                url=f"https://www.youtube.com/watch?v={entry.get('id')}",
                duration=entry.get("duration"),
            )
            videos.append(video)
    return videos


def save_video_index(path: Path, videos: Iterable[VideoMetadata]) -> None:
    """Persist the video metadata index for later reuse."""
    payload = [video.to_dict() for video in videos]
    save_json(path, payload)


def load_video_index(path: Path) -> List[VideoMetadata]:
    """Load existing video metadata from disk."""
    from .utils import load_json

    payload = load_json(path)
    videos: List[VideoMetadata] = []
    for item in payload:
        videos.append(
            VideoMetadata(
                video_id=item["video_id"],
                title=item["title"],
                upload_date=item["upload_date"],
                url=item["url"],
                duration=item.get("duration"),
            )
        )
    return videos


def download_audio(video: VideoMetadata, destination_dir: Path) -> Path:
    """Download the audio track for *video* and return the resulting file path."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename_base = video.slug
    output_template = str(destination_dir / f"{filename_base}.%(ext)s")

    for stale in destination_dir.glob(f"{filename_base}.*"):
        try:
            stale.unlink()
        except FileNotFoundError:
            continue

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video.url, download=True)
    downloaded_path = destination_dir / f"{filename_base}.mp3"
    if not downloaded_path.exists():
        # yt-dlp occasionally returns a different extension; fall back to detected filename
        actual_path = Path(result.get("requested_downloads", [{}])[0].get("_filename", ""))
        if actual_path and actual_path.exists():
            return actual_path
        raise FileNotFoundError(f"Audio download failed for video {video.video_id}")
    return downloaded_path
