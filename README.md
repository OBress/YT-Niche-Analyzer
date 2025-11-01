# YT-Niche-Analyzer

Creates in depth outlines of a YouTube channel's story/video format using a sequential, disk-friendly pipeline.

## Project Structure

```
story_outline_project/
├── main.py                # Orchestrator for the entire workflow
├── downloader.py          # YouTube metadata fetching and audio downloading helpers
├── transcriber.py         # ElevenLabs transcription interface
├── summarizer.py          # Outline generation and hierarchical merging
├── config.py              # Environment-aware configuration loader
├── utils.py               # Logging, serialization, and helper utilities
├── requirements.txt       # Python dependencies
├── .env.example           # Template for required API keys and configuration
├── data/
│   ├── transcripts/       # JSON transcripts (created at runtime)
│   ├── outlines/          # Story outlines (created at runtime)
│   ├── merged/            # Hierarchical merged summaries (created at runtime)
│   └── tmp_audio/         # Temporary audio downloads
└── logs/                  # Pipeline logs
```

## Prerequisites

1. Python 3.10+
2. `ffmpeg` available on your `PATH` (required by `yt-dlp` to extract audio)
3. ElevenLabs Speech-to-Text access and an OpenAI API key.

Install dependencies with:

```bash
pip install -r story_outline_project/requirements.txt
```

## Configuration

Copy the example environment file and populate it with your credentials:

```bash
cp story_outline_project/.env.example .env
```

Update the values for:

- `ELEVEN_API_KEY`
- `ELEVEN_MODEL_ID` (defaults to `eleven_monolingual_v1`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
- `YOUTUBE_CHANNEL_URL`

Optional overrides:

- `NUM_VIDEOS` – number of videos to process (default 50)
- `BATCH_SIZE` – number of outlines to merge per round (default 5)

## Running the Pipeline

Execute the orchestrator module to process the configured channel sequentially:

```bash
python -m story_outline_project.main
```

The pipeline performs the following steps for each video:

1. Download the audio track and save it to `data/tmp_audio/`.
2. Transcribe the audio with ElevenLabs and save the JSON transcript to `data/transcripts/`.
3. Delete the temporary audio file to conserve disk space.
4. Generate a structured outline via OpenAI and save it to `data/outlines/`.

After all transcripts/outlines are generated, the pipeline merges outlines in rounds (merge-sort style) until a final `final_outline.txt` is produced in `data/merged/`.

Progress, file validation, and cleanup are logged to `logs/pipeline.log`. If any step fails, the affected video metadata is saved to a retry file for later processing.
