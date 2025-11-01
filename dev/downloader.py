"""Video metadata retrieval and audio download utilities."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

from yt_dlp import YoutubeDL

try:
    from .utils import save_json, slugify
except ImportError:
    # Handle running as script (not as module)
    from utils import save_json, slugify


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
        return slugify(f"{self.video_id}_{self.title}")

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_channel_videos(channel_url: str, limit: int) -> List[VideoMetadata]:
    """Return metadata for the most recent *limit* videos on a channel."""
    # Ensure the URL points to the videos tab
    if not channel_url.endswith('/videos'):
        channel_url = channel_url.rstrip('/') + '/videos'
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": limit,  # Only fetch the number we need
    }
    videos: List[VideoMetadata] = []
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        
        if not info:
            return videos
            
        entries: Iterable[dict] = info.get("entries", [])
        entry_list = list(entries)
        
        # Just take the first `limit` videos (they're already in chronological order)
        for entry in entry_list[:limit]:
            if not entry or not entry.get("id"):
                continue
            
            video = VideoMetadata(
                video_id=entry.get("id", ""),
                title=entry.get("title", "Untitled"),
                upload_date="",  # Not used
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
    try:
        from .utils import load_json
    except ImportError:
        from utils import load_json

    payload = load_json(path)
    videos: List[VideoMetadata] = []
    for item in payload:
        videos.append(
            VideoMetadata(
                video_id=item["video_id"],
                title=item["title"],
                upload_date=item.get("upload_date", ""),
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
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video.url, download=True)
    
    # Check for m4a file first
    downloaded_path = destination_dir / f"{filename_base}.m4a"
    if not downloaded_path.exists():
        # Fall back to other possible extensions
        for ext in [".opus", ".webm", ".mp3"]:
            alt_path = destination_dir / f"{filename_base}{ext}"
            if alt_path.exists():
                return alt_path
        # Last resort: check yt-dlp's reported filename
        actual_path = Path(result.get("requested_downloads", [{}])[0].get("_filename", ""))
        if actual_path and actual_path.exists():
            return actual_path
        raise FileNotFoundError(f"Audio download failed for video {video.video_id}")
    return downloaded_path


if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests
    
    try:
        from .transcriber import transcribe_audio, save_transcript
        from .config import Config
    except ImportError:
        from transcriber import transcribe_audio, save_transcript
        from config import Config
    
    # Load config
    try:
        config = Config.from_env()
    except ValueError as e:
        print(f"Error loading config: {e}")
        print("Please ensure .env file is configured with required API keys.")
        sys.exit(1)
    
    # Test with a channel URL
    if len(sys.argv) > 1:
        channel_url = sys.argv[1]
    else:
        # Use channel from config
        channel_url = config.channel_url
        print(f"No channel URL provided. Using channel from config: {channel_url}\n")
    
    print("Fetching channel videos...")
    videos = fetch_channel_videos(channel_url, limit=5)
    
    print(f"Found {len(videos)} videos to download and transcribe:\n")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video.title}")
        if video.duration:
            print(f"   Duration: {video.duration}s")
        print()
    
    print("Processing videos in parallel (3 concurrent operations)...\n")
    print("Each video will be: downloaded → transcribed → audio deleted\n")
    
    temp_audio_dir = Path("test_downloads")
    transcript_dir = Path("test_transcripts")
    
    # Ensure directories exist
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to download, transcribe, and delete audio for a single video
    def process_single_video(video_tuple):
        i, total, video = video_tuple
        audio_path = None
        session = requests.Session()
        
        try:
            # Step 1: Download audio
            audio_path = download_audio(video, temp_audio_dir)
            size_mb = audio_path.stat().st_size / (1024*1024)
            
            # Step 2: Transcribe audio
            transcript = transcribe_audio(audio_path, video, config, session=session)
            
            # Step 3: Save transcript
            transcript_filename = f"{video.slug}.json"
            transcript_path = transcript_dir / transcript_filename
            save_transcript(transcript, transcript_path)
            
            # Step 4: Delete audio file
            audio_path.unlink()
            
            return (True, i, video.title, transcript_path, size_mb, None)
        except Exception as e:
            return (False, i, video.title, None, None, str(e))
        finally:
            session.close()
    
    # Use ThreadPoolExecutor for parallel processing (max 3 concurrent)
    max_workers = min(3, len(videos))
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing tasks
        futures = {
            executor.submit(process_single_video, (i, len(videos), video)): video
            for i, video in enumerate(videos, 1)
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            completed += 1
            success, i, title, transcript_path, size_mb, error = future.result()
            if success:
                print(f"✓ [{completed}/{len(videos)}] {title}")
                print(f"  Audio: {size_mb:.2f} MB (downloaded & deleted)")
                print(f"  Transcript: {transcript_path.name}\n")
            else:
                print(f"✗ [{completed}/{len(videos)}] Error processing {title}: {error}\n")
    
    print(f"✓ Processing complete!")
    print(f"  Audio files: Downloaded and deleted")
    print(f"  Transcripts saved to: {transcript_dir}")