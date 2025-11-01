"""Pipeline orchestrator for parallel YouTube story processing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from requests import Session

from dev.config import Config
from dev.downloader import (
    VideoMetadata,
    download_audio,
    fetch_channel_videos,
    load_video_index,
    save_video_index,
)
from dev.summarizer import generate_creative_framework, save_outline
from dev.transcriber import save_transcript, transcribe_audio
from dev.utils import remove_file, setup_logger, slugify


class PipelineError(Exception):
    """Raised when the pipeline encounters a recoverable error."""


def _load_or_create_index(config: Config, logger) -> List[VideoMetadata]:
    if config.video_index_path.exists():
        logger.info("Loading existing video index from %s", config.video_index_path)
        videos = load_video_index(config.video_index_path)
    else:
        logger.info(
            "Fetching metadata for the most recent %s videos from %s",
            config.num_videos,
            config.channel_url,
        )
        videos = fetch_channel_videos(config.channel_url, config.num_videos)
        save_video_index(config.video_index_path, videos)
        logger.info("Saved video index to %s", config.video_index_path)
    return videos


def _transcript_path(config: Config, video: VideoMetadata) -> Path:
    return config.transcript_dir / f"{video.slug}.json"


def process_video(
    video: VideoMetadata,
    config: Config,
    logger,
    *,
    session: Session | None = None,
) -> None:
    """Process a single video: download, transcribe, generate outline, and cleanup."""
    logger.info("Processing video: %s (%s)", video.title, video.video_id)

    transcript_path = _transcript_path(config, video)
    if transcript_path.exists():
        logger.info("Transcript already exists for %s, skipping.", video.video_id)
        return

    # Create a session for this worker if not provided
    if session is None:
        session = Session()
        own_session = True
    else:
        own_session = False

    audio_path: Path | None = None
    try:
        audio_path = download_audio(video, config.temp_audio_dir)
        logger.info("Downloaded audio to %s", audio_path)

        transcript = transcribe_audio(audio_path, video, config, session=session)
        save_transcript(transcript, transcript_path)
        if transcript_path.stat().st_size == 0:
            raise PipelineError("Transcript file is empty")
        logger.info("Saved transcript to %s", transcript_path)
    except Exception:
        logger.exception("Failed processing %s", video.video_id)
        raise
    finally:
        if audio_path:
            remove_file(audio_path)
            logger.info("Deleted temporary audio %s", audio_path)
        if own_session:
            session.close()

def main() -> int:
    try:
        config = Config.from_env()
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return 1

    config.ensure_directories()
    logger = setup_logger(config.log_file)

    try:
        videos = _load_or_create_index(config, logger)
    except Exception:
        logger.exception("Failed to load or create video index")
        return 1

    if not videos:
        logger.error("No videos found. Please check the channel URL.")
        return 1

    logger.info("Processing %d videos in parallel (batch size: %d)", len(videos), config.batch_size)
    
    retry_queue: List[VideoMetadata] = []
    completed = 0
    
    # Process videos in parallel using ThreadPoolExecutor
    max_workers = min(config.batch_size, len(videos))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks
        futures = {
            executor.submit(process_video, video, config, logger): video
            for video in videos
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            video = futures[future]
            completed += 1
            try:
                future.result()
                logger.info("✓ [%d/%d] Successfully processed %s", completed, len(videos), video.title)
            except Exception:
                logger.error("✗ [%d/%d] Failed to process %s", completed, len(videos), video.title)
                retry_queue.append(video)

    if retry_queue:
        logger.warning("Retry queue populated with %d videos", len(retry_queue))
        save_video_index(
            config.video_index_path.with_name(
                f"retry_{slugify(retry_queue[0].upload_date)}.json"
            ),
            retry_queue,
        )

    transcript_paths = sorted(config.transcript_dir.glob("*.json"))
    if transcript_paths:
        logger.info(
            "Generating creative framework from %d transcripts", len(transcript_paths)
        )
        creative_framework = generate_creative_framework(transcript_paths, config)
        framework_path = config.merged_dir / "creative_framework.txt"
        save_outline(creative_framework, framework_path)
        if framework_path.stat().st_size == 0:
            raise PipelineError("Creative framework file is empty")
        logger.info("Saved creative framework to %s", framework_path)
    else:
        logger.warning("No transcripts found to analyze for a creative framework.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
