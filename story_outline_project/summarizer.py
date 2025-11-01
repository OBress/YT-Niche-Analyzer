"""Summarization and hierarchical merging utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI

from .config import Config
from .transcriber import Transcript, load_transcript
from .utils import chunk_iterable


def _client(config: Config) -> OpenAI:
    return OpenAI(api_key=config.openai_api_key)


def _collect_text_from_response(response) -> str:
    pieces: List[str] = []
    for output in response.output:
        if output.type != "message":
            continue
        for content in output.message.content:
            if content.type == "text":
                pieces.append(content.text)
    return "".join(pieces).strip()


def generate_outline(
    transcript_path: Path,
    config: Config,
    *,
    client: Optional[OpenAI] = None,
) -> str:
    """Create a structured outline for a transcript."""
    transcript: Transcript = load_transcript(transcript_path)
    llm = client or _client(config)

    prompt = (
        "You are an expert narrative analyst. Given the transcript of a story, "
        "produce a structured outline with the following sections:\n\n"
        "1. Title\n"
        "2. Main Characters\n"
        "3. Setting\n"
        "4. Narrative Structure (exposition, rising action, climax, falling action, resolution)\n"
        "5. Key Plot Points with timestamp ranges\n"
        "6. Emotional Arc\n"
        "7. Themes and Motifs\n\n"
        "Respond in markdown, using bullet lists where appropriate."
    )

    user_content = (
        f"Title: {transcript.title}\n"
        f"Upload Date: {transcript.upload_date}\n"
        f"Duration (seconds): {transcript.duration_sec}\n"
        "Transcript:\n" + transcript.full_text
    )

    response = llm.responses.create(
        model=config.openai_model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content,
                    }
                ],
            },
        ],
    )
    outline_text = _collect_text_from_response(response)
    return outline_text


def save_outline(outline_text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(outline_text, encoding="utf-8")


def hierarchical_merge_outlines(
    outline_paths: Iterable[Path],
    config: Config,
    merged_dir: Path,
    *,
    client: Optional[OpenAI] = None,
) -> Optional[Path]:
    """Merge outlines hierarchically in a merge-sort fashion."""
    outlines = [Path(path) for path in outline_paths]
    if not outlines:
        return None

    merged_dir.mkdir(parents=True, exist_ok=True)
    outlines.sort()
    llm = client or _client(config)
    round_number = 1
    current_paths = outlines

    while len(current_paths) > 1:
        next_round: List[Path] = []
        for batch_index, batch in enumerate(chunk_iterable(current_paths, config.batch_size), start=1):
            combined_outline = "\n\n".join(path.read_text(encoding="utf-8") for path in batch)
            merge_prompt = (
                "You will receive multiple story outlines. Merge them into a single meta-outline "
                "that highlights overlapping themes, motifs, archetypes, emotional arcs, and "
                "distinct outliers. Summarize common structures and notable differences. "
                "Respond in markdown with the following sections:\n\n"
                "1. Major Recurring Motifs\n"
                "2. Common Narrative Structures\n"
                "3. Emotional and Tonal Progression\n"
                "4. Distinct Outliers or Innovations\n"
                "5. Opportunities for Future Storytelling"
            )

            response = llm.responses.create(
                model=config.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": merge_prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": combined_outline,
                            }
                        ],
                    },
                ],
            )
            merged_text = _collect_text_from_response(response)
            output_path = merged_dir / f"round{round_number}_batch{batch_index}.txt"
            output_path.write_text(merged_text, encoding="utf-8")
            next_round.append(output_path)

        current_paths = next_round
        round_number += 1

    final_path = current_paths[0]
    if final_path != merged_dir / "final_outline.txt":
        final_destination = merged_dir / "final_outline.txt"
        final_destination.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")
        final_path = final_destination
    return final_path
