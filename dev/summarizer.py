"""Programmatic story framework generation for narrative corpora."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from .config import Config
from .transcriber import Transcript, load_transcript

_WORD_PATTERN = re.compile(r"\b[\w']+\b")
@dataclass(frozen=True)
class StoryArcSection:
    """Template describing a repeating narrative section."""

    name: str
    purpose: str
    tone: str
    style: str
    proportion: float


@dataclass(frozen=True)
class StoryArcFormat:
    """Definition of a narrative format detected across the corpus."""

    identifier: str
    name: str
    overview: str
    keywords: Dict[str, int]
    sections: Sequence[StoryArcSection]


@dataclass(frozen=True)
class MotifDefinition:
    """Keyword-driven motif detection template."""

    name: str
    keywords: Sequence[str]
    description: str
    tone: str


@dataclass
class TranscriptProfile:
    """Pre-computed features for a transcript used in corpus analysis."""

    path: Path
    transcript: Transcript
    tokens: List[str]
    word_counter: Counter[str]
    sentences: List[str]
    sentences_lower: List[str]
    sentence_tokens: List[List[str]]
    word_count: int
    duration_minutes: float
    arc_format: StoryArcFormat
    section_word_counts: List[int]
    section_midpoints: List[float]
    peak_section_midpoint: float


def _tokenize(text: str) -> List[str]:
    return _WORD_PATTERN.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned or ([text] if text else [])


def _round_to_five(minutes: float) -> int:
    if minutes <= 0:
        return 0
    rounded = int(round(minutes / 5.0) * 5)
    return max(5, rounded)


def _format_time_range(values: Sequence[float]) -> str:
    durations = [value for value in values if value > 0]
    if not durations:
        return "≈0 minutes"
    rounded = [_round_to_five(value) for value in durations]
    minimum = min(rounded)
    maximum = max(rounded)
    if minimum == maximum:
        return f"≈{minimum} minutes"
    return f"≈{minimum}–{maximum} minutes"


SETTING_DESCRIPTIONS = {
    "forest": "dense forests",
    "woods": "remote woods",
    "mountain": "mountain ranges",
    "mountains": "mountain ranges",
    "island": "isolated islands",
    "town": "insular frontier towns",
    "village": "secluded communities",
    "house": "haunted dwellings",
    "shelter": "abandoned shelters",
    "bunker": "buried bunkers",
    "factory": "industrial ruins",
    "camp": "makeshift camps",
}

ROLE_DESCRIPTIONS = {
    "ranger": "park rangers and wilderness guardians",
    "soldier": "soldiers and veterans",
    "guide": "guides and fixers for dangerous terrain",
    "guard": "night guards and sentries",
    "lookout": "lookouts stationed at lonely posts",
    "worker": "shift workers bound by ominous rules",
    "investigator": "field investigators and researchers",
    "caretaker": "caretakers of cursed properties",
    "patrol": "patrol leaders and scouts",
    "pilot": "pilots and transport escorts",
}

TONE_DESCRIPTIONS = {
    "scream": "echoing distress calls",
    "whisper": "hushed whispers",
    "blood": "visceral dread",
    "shadow": "encroaching shadow imagery",
    "night": "perpetual nightfall",
    "fear": "palpable fear",
    "strange": "language of the uncanny",
    "ritual": "ritualistic repetition",
    "echo": "echoing anomalies",
}

STAGE_WINDOWS = [
    (0.2, "Orientation & Setup"),
    (0.4, "Foreshadowing & Early Disturbances"),
    (0.7, "Escalation & Investigation"),
    (0.9, "Crisis & Confrontation"),
    (1.01, "Aftermath & Reckoning"),
]


_STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "it",
    "that",
    "is",
    "was",
    "for",
    "on",
    "with",
    "as",
    "at",
    "my",
    "we",
    "but",
    "had",
    "our",
    "they",
    "from",
    "this",
    "be",
    "by",
    "about",
    "were",
    "an",
    "are",
    "me",
    "their",
    "all",
    "have",
}


STORY_ARC_FORMATS: Sequence[StoryArcFormat] = (
    StoryArcFormat(
        identifier="rule_protocol",
        name="Rule-Driven Survival Protocol",
        overview=(
            "Narrators inherit strict lists of prohibitions and recount the spiralling cost of even a single"
            " broken rule."
        ),
        keywords={
            "rule": 6,
            "rules": 6,
            "instruction": 4,
            "instructions": 4,
            "list": 3,
            "forbidden": 4,
            "must": 2,
            "dont": 2,
            "don't": 2,
            "penalty": 3,
            "compliance": 2,
            "punishment": 3,
        },
        sections=(
            StoryArcSection(
                name="Orientation and Mandate Briefing",
                purpose="Introduce the narrator, their duty, and the unnerving rule-set they must obey.",
                tone="Guarded curiosity laced with apprehension.",
                style="Measured exposition blending job details with ominous hints.",
                proportion=0.18,
            ),
            StoryArcSection(
                name="Rule Ledger and Foreshadowing",
                purpose="Lay out each prohibition while hinting at the consequences behind them.",
                tone="Cautionary and procedural, emphasizing looming dread.",
                style="Enumerated, list-driven delivery with clipped sentences.",
                proportion=0.25,
            ),
            StoryArcSection(
                name="Rule Friction and Escalation",
                purpose="Show mounting pressure as reality strains against the written code.",
                tone="Tense vigilance giving way to urgency.",
                style="Sequential incidents stitched with anxious introspection.",
                proportion=0.22,
            ),
            StoryArcSection(
                name="Confronting the Breach",
                purpose="Depict the inevitable violation and direct confrontation with the threat.",
                tone="Panicked, adrenaline-fueled survivalism.",
                style="Fast-paced action and sensory detail.",
                proportion=0.2,
            ),
            StoryArcSection(
                name="Aftermath and New Protocols",
                purpose="Establish the cost, lingering damage, and revised rules for the future.",
                tone="Somber resolve with fatalistic acceptance.",
                style="Reflective debrief mixing lessons learned with ominous warnings.",
                proportion=0.15,
            ),
        ),
    ),
    StoryArcFormat(
        identifier="field_descent",
        name="Field Expedition into Hostile Territory",
        overview=(
            "Investigators, rangers, and soldiers chart strange terrain, logging anomalies until the"
            " landscape reveals its predatory intelligence."
        ),
        keywords={
            "patrol": 4,
            "patrols": 4,
            "mission": 4,
            "team": 3,
            "search": 4,
            "expedition": 4,
            "assignment": 3,
            "ranger": 5,
            "lookout": 4,
            "camp": 3,
            "camping": 3,
            "forest": 4,
            "woods": 4,
            "night": 2,
            "watch": 3,
            "station": 3,
            "outpost": 3,
            "tour": 2,
            "guide": 3,
            "scout": 3,
        },
        sections=(
            StoryArcSection(
                name="Mission Brief and Arrival",
                purpose="Explain the assignment, landscape, and operating constraints.",
                tone="Professional focus with creeping unease.",
                style="Field-report exposition rich in sensory geography.",
                proportion=0.17,
            ),
            StoryArcSection(
                name="First Anomalies",
                purpose="Document the earliest signs that the terrain is hostile or intelligent.",
                tone="Prickling curiosity turning into dread.",
                style="Observational notes punctuated by clipped dialogue.",
                proportion=0.23,
            ),
            StoryArcSection(
                name="Encirclement and Stakes",
                purpose="Show the protagonists understanding the scale of the threat.",
                tone="Claustrophobic escalation.",
                style="Logbook-like sequencing of strange events.",
                proportion=0.28,
            ),
            StoryArcSection(
                name="Breaking the Perimeter",
                purpose="Narrate the direct clash or escape attempt.",
                tone="Urgent and tactical.",
                style="Action-heavy descriptions with sensory overload.",
                proportion=0.2,
            ),
            StoryArcSection(
                name="Extraction Debrief",
                purpose="Outline the fallout, casualties, and unresolved mysteries.",
                tone="Haunted pragmatism.",
                style="Measured reporting mixed with introspective regret.",
                proportion=0.12,
            ),
        ),
    ),
    StoryArcFormat(
        identifier="haunted_haven",
        name="Haunted Haven Entrapment",
        overview=(
            "Caretakers of inherited homes or temporary shelters uncover secret rooms and resident"
            " entities that refuse to stay buried."
        ),
        keywords={
            "house": 6,
            "home": 4,
            "basement": 5,
            "attic": 4,
            "floorboard": 4,
            "floorboards": 4,
            "apartment": 4,
            "door": 3,
            "room": 2,
            "hallway": 2,
            "tenant": 2,
            "inherit": 3,
            "inherited": 3,
            "stairs": 2,
        },
        sections=(
            StoryArcSection(
                name="Relocation and Domestic Setup",
                purpose="Describe the move-in circumstances and the building's first impressions.",
                tone="Hopeful curiosity undercut by subtle menace.",
                style="Atmospheric scene-setting with domestic detail.",
                proportion=0.2,
            ),
            StoryArcSection(
                name="Unsettling Disturbances",
                purpose="Catalogue early noises, smells, or impossibilities in the dwelling.",
                tone="Creeping paranoia.",
                style="Sensory-driven, focusing on tactile anomalies.",
                proportion=0.24,
            ),
            StoryArcSection(
                name="Discovery of the Hidden Wing",
                purpose="Reveal the architectural secret or sealed chamber.",
                tone="Morbid fascination tipping into fear.",
                style="Exploratory prose with deliberate pacing.",
                proportion=0.22,
            ),
            StoryArcSection(
                name="Manifestation and Siege",
                purpose="Depict the entity's full emergence and the desperate attempt to survive.",
                tone="Panicked claustrophobia.",
                style="Staccato sentences and visceral imagery.",
                proportion=0.18,
            ),
            StoryArcSection(
                name="Exit Strategy and Lingering Haunt",
                purpose="Resolve the immediate danger while foreshadowing future hauntings.",
                tone="Resigned dread.",
                style="Reflective coda with unresolved questions.",
                proportion=0.16,
            ),
        ),
    ),
)

_DEFAULT_FORMAT = STORY_ARC_FORMATS[1]

MOTIF_DEFINITIONS: Sequence[MotifDefinition] = (
    MotifDefinition(
        name="Rulebooks and Non-Negotiable Directives",
        keywords=("rule", "rules", "instruction", "guideline"),
        description="Symbols of fragile order; every mandate foreshadows the chaos waiting to erupt.",
        tone="Intensifies fatalistic vigilance and paranoia.",
    ),
    MotifDefinition(
        name="Encroaching Wilderness and Predatory Terrain",
        keywords=("forest", "woods", "tree", "trail", "clearing", "underbrush"),
        description="Nature behaves as a sentient antagonist that herds characters into traps.",
        tone="Amplifies the feeling of isolation and surveillance.",
    ),
    MotifDefinition(
        name="Nocturnal Surveillance Tools",
        keywords=("flashlight", "torch", "radio", "walkie", "night vision", "scanner"),
        description="Technology stands in for courage, flickering at the worst possible moment.",
        tone="Highlights brittle confidence and dependency on failing tools.",
    ),
    MotifDefinition(
        name="Hidden Doorways and Sealed Thresholds",
        keywords=("door", "hatch", "basement", "attic", "floorboard", "trapdoor"),
        description="Boundaries between normalcy and horror literally splinter open.",
        tone="Reinforces the transition from curiosity to panic.",
    ),
    MotifDefinition(
        name="Missing Persons and Echoed Screams",
        keywords=("missing", "disappear", "vanish", "scream", "echo"),
        description="Lives slip silently away, leaving only sound as evidence.",
        tone="Sustains dread through absence and unanswered pleas.",
    ),
    MotifDefinition(
        name="Mimicry and False Voices",
        keywords=("voice", "whisper", "mimic", "call", "lure"),
        description="Supernatural predators weaponize familiar voices to bait victims.",
        tone="Undermines trust and enforces psychological terror.",
    ),
    MotifDefinition(
        name="Protective Circles and Perimeters",
        keywords=("perimeter", "boundary", "circle", "line", "ward"),
        description="Drawn lines and wards offer brief reprieve before collapsing.",
        tone="Stresses the temporary nature of safety.",
    ),
    MotifDefinition(
        name="Cursed Heirlooms and Binding Pacts",
        keywords=("promise", "oath", "curse", "inherit", "legacy"),
        description="Characters inherit obligations that tether them to the anomaly.",
        tone="Adds inevitability and generational doom.",
    ),
)


def _choose_descriptor(counter: Counter[str], mapping: Dict[str, str], default: str) -> str:
    best_word = None
    best_count = 0
    for word, description in mapping.items():
        count = counter.get(word, 0)
        if count > best_count:
            best_word = word
            best_count = count
    return mapping.get(best_word, default)


def _format_story_count(count: int) -> str:
    return f"{count} story" if count == 1 else f"{count} stories"


def _classify_format(counter: Counter[str]) -> StoryArcFormat:
    best_format = _DEFAULT_FORMAT
    best_score = -1
    for story_format in STORY_ARC_FORMATS:
        score = 0
        for keyword, weight in story_format.keywords.items():
            score += counter.get(keyword, 0) * weight
        if story_format.identifier == "rule_protocol":
            rule_hits = counter.get("rule", 0) + counter.get("rules", 0)
            if rule_hits >= 3:
                score += 20 + rule_hits * 2
        elif story_format.identifier == "haunted_haven":
            home_hits = (
                counter.get("house", 0)
                + counter.get("home", 0)
                + counter.get("basement", 0)
                + counter.get("attic", 0)
            )
            if home_hits >= 3:
                score += 15 + home_hits * 2
        elif story_format.identifier == "field_descent":
            terrain_hits = (
                counter.get("forest", 0)
                + counter.get("woods", 0)
                + counter.get("patrol", 0)
                + counter.get("patrols", 0)
                + counter.get("mission", 0)
            )
            score += terrain_hits
        if score > best_score:
            best_score = score
            best_format = story_format
    return best_format


def _build_profile(path: Path, transcript: Transcript) -> TranscriptProfile | None:
    sentences = _split_sentences(transcript.full_text)
    tokens = _tokenize(transcript.full_text)
    if not sentences or not tokens:
        return None

    sentence_tokens = [_tokenize(sentence) for sentence in sentences]
    sentences_lower = [sentence.lower() for sentence in sentences]
    word_counter = Counter(token for token in tokens if token not in _STOPWORDS)
    word_count = sum(len(tokens_) for tokens_ in sentence_tokens)

    if word_count == 0:
        return None

    duration_minutes = (
        float(transcript.duration_sec) / 60.0 if transcript.duration_sec else word_count / 120.0
    )

    arc_format = _classify_format(word_counter)
    total_sentences = len(sentences)
    section_ranges: List[tuple[int, int]] = []
    section_word_counts: List[int] = []
    section_midpoints: List[float] = []

    current_index = 0
    for idx, section in enumerate(arc_format.sections):
        remaining_sections = len(arc_format.sections) - idx
        remaining_sentences = total_sentences - current_index
        if remaining_sentences <= 0:
            start = end = total_sentences
        else:
            ideal_length = int(round(section.proportion * total_sentences))
            minimum_length = 1 if remaining_sections > 0 else remaining_sentences
            maximum_length = max(1, remaining_sentences - (remaining_sections - 1))
            length = max(minimum_length, min(maximum_length, ideal_length))
            if idx == len(arc_format.sections) - 1:
                length = remaining_sentences
            start = current_index
            end = min(total_sentences, start + length)
            current_index = end
        section_ranges.append((start, end))
        words = sum(len(sentence_tokens[i]) for i in range(start, end)) if end > start else 0
        section_word_counts.append(words)
        if total_sentences:
            midpoint = ((start + end) / 2) / total_sentences
        else:
            midpoint = 0.0
        section_midpoints.append(midpoint)

    if total_sentences and current_index < total_sentences:
        # assign any leftover sentences to the final section
        last_range_start, last_range_end = section_ranges[-1]
        section_ranges[-1] = (last_range_start, total_sentences)
        additional_words = sum(
            len(sentence_tokens[i]) for i in range(current_index, total_sentences)
        )
        section_word_counts[-1] += additional_words

    total_words = sum(section_word_counts)
    if total_words == 0:
        return None

    peak_index = max(range(len(section_word_counts)), key=section_word_counts.__getitem__)
    peak_midpoint = section_midpoints[peak_index] if section_midpoints else 0.0

    return TranscriptProfile(
        path=path,
        transcript=transcript,
        tokens=tokens,
        word_counter=word_counter,
        sentences=sentences,
        sentences_lower=sentences_lower,
        sentence_tokens=sentence_tokens,
        word_count=total_words,
        duration_minutes=duration_minutes,
        arc_format=arc_format,
        section_word_counts=section_word_counts,
        section_midpoints=section_midpoints,
        peak_section_midpoint=peak_midpoint,
    )


def _analyse_motifs(profiles: Sequence[TranscriptProfile]) -> List[dict]:
    if not profiles:
        return []

    threshold = 1 if len(profiles) <= 2 else max(2, math.ceil(len(profiles) * 0.25))
    results: List[dict] = []

    for motif in MOTIF_DEFINITIONS:
        total_hits = 0
        story_paths: set[Path] = set()
        stage_counter: Counter[str] = Counter()

        for profile in profiles:
            total_sentences = len(profile.sentences)
            for sentence_index, sentence in enumerate(profile.sentences_lower):
                hit_count = 0
                for keyword in motif.keywords:
                    pattern = fr"\b{re.escape(keyword)}\b"
                    hits = re.findall(pattern, sentence)
                    hit_count += len(hits)
                if hit_count:
                    total_hits += hit_count
                    story_paths.add(profile.path)
                    if total_sentences:
                        ratio = (sentence_index + 0.5) / total_sentences
                    else:
                        ratio = 0.0
                    for boundary, label in STAGE_WINDOWS:
                        if ratio <= boundary:
                            stage_counter[label] += hit_count
                            break

        if len(story_paths) >= threshold and total_hits >= len(story_paths):
            dominant_stage = (
                stage_counter.most_common(1)[0][0] if stage_counter else "Across multiple phases"
            )
            results.append(
                {
                    "name": motif.name,
                    "description": motif.description,
                    "tone": motif.tone,
                    "story_count": len(story_paths),
                    "total_hits": total_hits,
                    "dominant_stage": dominant_stage,
                }
            )

    return results


def generate_creative_framework(
    transcript_paths: Iterable[Path],
    config: Config | None = None,
) -> str:
    """Generate a creative framework summary across *transcript_paths*."""

    profiles: List[TranscriptProfile] = []
    seen_paths: set[Path] = set()
    for path in transcript_paths:
        resolved = Path(path)
        if resolved in seen_paths or not resolved.exists():
            continue
        seen_paths.add(resolved)
        transcript = load_transcript(resolved)
        profile = _build_profile(resolved, transcript)
        if profile:
            profiles.append(profile)

    if not profiles:
        return "No transcripts available for analysis."

    word_counter: Counter[str] = Counter()
    durations = []
    format_buckets: dict[str, List[TranscriptProfile]] = defaultdict(list)
    peak_positions: List[float] = []
    mid_section_shares: List[float] = []

    for profile in profiles:
        word_counter.update(profile.word_counter)
        durations.append(profile.duration_minutes)
        format_buckets[profile.arc_format.identifier].append(profile)
        peak_positions.append(profile.peak_section_midpoint)
        if len(profile.section_word_counts) >= 3:
            total_words = sum(profile.section_word_counts)
            if total_words:
                mid_section_shares.append(profile.section_word_counts[2] / total_words)

    story_count = len(profiles)
    avg_words = int(round(mean(profile.word_count for profile in profiles)))
    avg_minutes_value = mean(durations) if durations else 0.0
    avg_minutes = _round_to_five(avg_minutes_value)

    dominant_setting = _choose_descriptor(word_counter, SETTING_DESCRIPTIONS, "isolated, untamed locations")
    dominant_role = _choose_descriptor(word_counter, ROLE_DESCRIPTIONS, "isolated caretakers and scouts")
    dominant_tone = _choose_descriptor(word_counter, TONE_DESCRIPTIONS, "persistent dread")

    pacing_statement: str
    if mid_section_shares:
        mid_share = mean(mid_section_shares)
        if mid_share >= 0.33:
            pacing_statement = (
                f"a slow-burn escalation that devotes roughly {int(round(mid_share * 100))}% of the runtime"
                " to mounting anomalies"
            )
        elif mid_share <= 0.26:
            pacing_statement = (
                f"a brisk escalation with mid-story action occupying about {int(round(mid_share * 100))}%"
                " of each narrative"
            )
        else:
            pacing_statement = (
                f"a balanced pace where build-up and payoff split the runtime at about {int(round(mid_share * 100))}%"
            )
    else:
        pacing_statement = "a measured escalation toward each climax"

    general_overview = (
        f"{story_count} long-form, first-person horror narratives anchor this corpus, most of them unfolding"
        f" in {dominant_setting} and told by {dominant_role}. The language leans into {dominant_tone},"
        f" and the pacing favors {pacing_statement}. On average they run about {avg_minutes} minutes"
        f" (≈{avg_words} words)."
    )

    formats_in_use = [fmt for fmt in STORY_ARC_FORMATS if format_buckets.get(fmt.identifier)]
    if not formats_in_use:
        formats_in_use = list(STORY_ARC_FORMATS)

    output_lines: List[str] = []
    output_lines.append("General Overview")
    output_lines.append("=================")
    output_lines.append(general_overview)
    output_lines.append("")

    output_lines.append("Story Arc Formats")
    output_lines.append("=================")

    for index, story_format in enumerate(formats_in_use, start=1):
        bucket = format_buckets.get(story_format.identifier, [])
        count = len(bucket)
        share = (count / story_count * 100) if story_count else 0
        output_lines.append(f"Story Arc Format #{index}: {story_format.name}")
        overview_line = (
            f"Overview: {story_format.overview}"
            f" (observed in {count} of {story_count} stories, {share:.0f}% of the corpus)."
        )
        output_lines.append(overview_line)

        section_times: List[List[float]] = [[] for _ in story_format.sections]
        for profile in bucket:
            for idx, word_count in enumerate(profile.section_word_counts):
                minutes = word_count / 120.0 if word_count else 0.0
                section_times[idx].append(minutes)

        for section_index, section in enumerate(story_format.sections, start=1):
            time_range = _format_time_range(section_times[section_index - 1])
            output_lines.append(f"  {section_index}. {section.name} ({time_range})")
            output_lines.append(f"     - Purpose: {section.purpose}")
            output_lines.append(f"     - Tone: {section.tone}")
            output_lines.append(f"     - Writing Style: {section.style}")
        output_lines.append("")

    min_duration = _round_to_five(min(durations)) if durations else 0
    max_duration = _round_to_five(max(durations)) if durations else 0
    average_duration = avg_minutes

    most_common_identifier = max(format_buckets, key=lambda key: len(format_buckets[key]))
    most_common_format = next(
        fmt for fmt in STORY_ARC_FORMATS if fmt.identifier == most_common_identifier
    )
    most_common_count = len(format_buckets[most_common_identifier])

    variation_descriptions: List[str] = []
    for fmt in formats_in_use:
        if fmt.identifier == most_common_format.identifier:
            continue
        bucket = format_buckets.get(fmt.identifier, [])
        if bucket:
            normalized_overview = fmt.overview.lower().rstrip(".")
            variation_descriptions.append(
                f"{fmt.name} ({_format_story_count(len(bucket))}) leans on the idea that {normalized_overview}"
            )

    peak_statement: str
    if peak_positions:
        avg_peak = mean(peak_positions)
        peak_percent = int(round(avg_peak * 100))
        if avg_peak >= 0.75:
            peak_statement = (
                f"Tension climbs almost linearly toward the finale, with climaxes arriving around {peak_percent}%"
                " of each runtime."
            )
        elif avg_peak >= 0.6:
            peak_statement = (
                f"Most stories crest late, peaking near the {peak_percent}% mark before dropping into reflective codas."
            )
        elif avg_peak >= 0.45:
            peak_statement = (
                f"The corpus cycles through mid-story spikes, cresting near {peak_percent}% before settling into fallout sections."
            )
        else:
            peak_statement = (
                f"The narratives front-load their shocks, striking near {peak_percent}% and lingering in aftermath."
            )
    else:
        peak_statement = "The narratives maintain a consistent tension profile without a dominant peak."

    output_lines.append("Average Runtime & Narrative Rhythm")
    output_lines.append("==================================")
    output_lines.append(
        f"Most videos span ≈{min_duration}–{max_duration} minutes, with a mean near {average_duration} minutes."
    )
    dominant_overview = most_common_format.overview.rstrip(".")
    output_lines.append(
        f"{most_common_format.name} is the prevailing blueprint ({most_common_count} of {story_count} stories), "
        f"where {dominant_overview.lower()}."
    )
    if variation_descriptions:
        output_lines.append("Structural variations appear when " + "; ".join(variation_descriptions) + ".")
    output_lines.append(peak_statement)
    output_lines.append("")

    motif_results = _analyse_motifs(profiles)

    output_lines.append("Recurring Motifs")
    output_lines.append("================")
    if not motif_results:
        output_lines.append("No recurring motifs met the detection threshold across the analysed transcripts.")
    else:
        for motif in motif_results:
            output_lines.append(
                f"- **{motif['name']}** — {motif['description']} Appears in {_format_story_count(motif['story_count'])}"
                f" ({motif['total_hits']} textual hits)."
            )
            output_lines.append(f"  - Arc Placement: Typically surfaces during *{motif['dominant_stage']}*.")
            output_lines.append(f"  - Reinforces: {motif['tone']}")
    output_lines.append("")

    return "\n".join(output_lines)


def save_outline(outline_text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(outline_text, encoding="utf-8")
