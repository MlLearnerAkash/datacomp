"""
gemini_cap_gen.py — Refine image captions using Google Gemini.

Reads images + alt-text from a directory and existing PaliGemma captions
from a Parquet file, then uses Gemini to produce higher-quality captions.
Results are appended to the input Parquet (or saved to a new one).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("gemini_captions")

# Constants
VALID_IMAGE_EXTENSIONS: frozenset = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MiB per Gemini limits
DEFAULT_MODEL: str = "gemini-2.5-flash"#"gemini-{1/2/3}.5-flash"
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RATE_LIMIT_RPS: float = 2.0  # requests per second
TEMPERATURE: float = 0.4
TOP_P: float = 0.9
MAX_OUTPUT_TOKENS: int = 1024


# Prompt template
SYSTEM_INSTRUCTION: str = (
    "You are an expert image captioner. Your task is to describe images "
    "accurately and concisely using any available context (alt-text, prior "
    "captions). Never fabricate details. If uncertain, omit the detail."
)

CAPTION_PROMPT_TEMPLATE: str = (
    "The image below came from a web page and had the alt-text: {alt_text}.\n"
    "A previous model (PaliGemma) generated this caption: {pali_caption}.\n\n"
    "Please describe what is in the image using the alt-text and the PaliGemma "
    "caption as guides to GROUND your response. For example, if the alt-text "
    "contains a specific brand name, use that brand name in your output. "
    "Be descriptive but concise. DO NOT make things up. If you cannot tell "
    "something with certainty, simply do not say anything about it."
    "make the description not more than 30 words, STRICTLY."
    "if the background is relvant, describe that in short with the subject" #helps in scene description.
)

FALLBACK_PROMPT: str = (
    "Describe this image concisely and accurately. "
    "Do not fabricate any details."
)


# Data structures
@dataclass(frozen=True)
class ImageRecord:
    """A single image with its metadata."""

    filename: str
    full_path: Path
    alt_text: str
    pali_caption: str
    uid: Optional[str] = None  # unique identifier from the parquet if present


@dataclass
class ProcessingStats:
    """Accumulated processing statistics."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_image: int = 0
    skipped_no_parquet_match: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Total={self.total}  Succeeded={self.succeeded}  Failed={self.failed}  "
            f"Skipped(missing_img={self.skipped_missing_image} "
            f"invalid_img={self.skipped_invalid_image} "
            f"no_parquet_match={self.skipped_no_parquet_match})  "
            f"SuccessRate={self.success_rate:.1%}"
        )

# Rate limiter — simple token bucket
class RateLimiter:
    """Token-bucket rate limiter to respect API quotas."""

    def __init__(self, requests_per_second: float) -> None:
        self._period: float = 1.0 / requests_per_second
        self._last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._period:
            time.sleep(self._period - elapsed)
        self._last = time.monotonic()


# Image helpers
def _load_image_bytes(path: Path) -> bytes:
    """Load and re-encode image to bytes; raise on invalid/corrupt images."""
    try:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA", "P", "PA"):
            img = img.convert("RGB")
        elif img.mode == "A":
            img = img.convert("L")
        # Re-encode to a safe format
        from io import BytesIO

        buf = BytesIO()
        fmt = img.format or "JPEG"
        img.save(buf, format=fmt)
        raw = buf.getvalue()
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image {path.name} is {len(raw)} bytes (limit {MAX_IMAGE_BYTES})"
            )
        return raw
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image format: {path.name}")
    except Exception as exc:
        raise ValueError(f"Failed to load image {path.name}: {exc}") from exc


# Directory traversal — recursive
def _supported_images(dir_path: Path) -> Iterator[Path]:
    """Recursively yield image file paths in *dir_path*.

    Uses manual recursion (stack-based) for clarity and explicit depth
    limiting to avoid infinite loops from symlink cycles.
    """
    MAX_DEPTH = 20
    stack: List[Tuple[Path, int]] = [(dir_path, 0)]

    while stack:
        current, depth = stack.pop()
        if depth > MAX_DEPTH:
            logger.warning("Max recursion depth reached at %s", current)
            continue

        try:
            entries = sorted(current.iterdir())
        except PermissionError:
            logger.warning("Permission denied: %s", current)
            continue
        except OSError as exc:
            logger.warning("OS error reading %s: %s", current, exc)
            continue

        for entry in entries:
            if entry.is_symlink():
                # Follow symlink once (resolve to real path, check depth)
                resolved = entry.resolve()
                if resolved.is_dir():
                    stack.append((resolved, depth + 1))
                elif resolved.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    yield resolved
            elif entry.is_dir():
                stack.append((entry, depth + 1))
            elif entry.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                yield entry


# Prompt construction
def build_prompt(record: ImageRecord) -> Tuple[str, List]:
    """Return (text_prompt, contents_list) for the Gemini API.

    The contents list contains the image bytes and the text prompt.
    """
    if record.alt_text or record.pali_caption:
        text = CAPTION_PROMPT_TEMPLATE.format(
            alt_text=record.alt_text or "(none)",
            pali_caption=record.pali_caption or "(none)",
        )
    else:
        text = FALLBACK_PROMPT

    return text


# Gemini client
class GeminiCaptioner:
    """Wraps the Gemini client with retry, rate-limiting, and batching."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. "
                "Create a .env file with GEMINI_API_KEY=your_key"
            )

        self._client: genai.Client = genai.Client(
            api_key=api_key, vertexai=True
        )
        self._model_name: str = model_name
        self._rate_limiter = RateLimiter(rate_limit_rps)
        self._generation_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            system_instruction=SYSTEM_INSTRUCTION,
            seed= 42
        )

        logger.info(
            "GeminiCaptioner initialised — model=%s  rps=%.1f",
            model_name,
            rate_limit_rps,
        )

    @retry(
        retry=retry_if_exception_type(
            (
                requests.ConnectionError,
                requests.Timeout,
                OSError,
            )
        ),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_api(self, image_bytes: bytes, prompt: str) -> str:
        """Single API call with retry wrapper."""
        self._rate_limiter.wait()

        # Build contents: inline image then text
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ]

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._generation_config,
        )

        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = (
                feedback.block_reason if feedback else "no candidates returned"
            )
            msg = f"Gemini returned no candidates: {reason}"
            logger.warning(msg)
            raise ValueError(msg)

        return response.text.strip()

    def caption_one(self, record: ImageRecord) -> str:
        """Caption a single image record. Raises on failure after retries."""
        image_bytes = _load_image_bytes(record.full_path)
        prompt = build_prompt(record)
        return self._call_api(image_bytes, prompt)

    def caption_batch(
        self, records: List[ImageRecord], stats: ProcessingStats
    ) -> Dict[str, str]:
        """Caption a batch sequentially, collecting per-record results.

        Returns a dict mapping filename -> gemini_caption.
        Failed entries are logged in *stats*.
        """
        results: Dict[str, str] = {}
        for record in records:
            try:
                caption = self.caption_one(record)
                results[record.filename] = caption
                stats.succeeded += 1
                logger.debug("✓ %s → %s", record.filename, caption[:80])
            except Exception as exc:
                stats.failed += 1
                err_msg = f"{record.filename}: {exc}"
                stats.errors.append(err_msg)
                logger.error("✗ %s", err_msg)
        return results


# Parquet helpers
def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame; raise gracefully on issues."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        raise ValueError(f"Failed to read parquet {path}: {exc}") from exc


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet. Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Saved %d rows to %s", len(df), path)


# Core pipeline
def match_records(
    image_dir: Path,
    parquet_df: pd.DataFrame,
    stats: ProcessingStats,
) -> Iterator[ImageRecord]:
    """Match image files on disk against rows in the Parquet DataFrame.

    Uses recursive traversal of *image_dir*. Each image is matched to
    a Parquet row by filename (without extension).
    """
    # Build fast lookup: base_name → row
    parquet_lookup: Dict[str, pd.Series] = {}
    for _, row in parquet_df.iterrows():
        name = str(row.get("filename", ""))
        if not name:
            continue
        base = Path(name).stem
        parquet_lookup[base] = row

    for img_path in _supported_images(image_dir):
        stats.total += 1
        base_name = img_path.stem

        # 1. Check parquet match
        row = parquet_lookup.get(base_name)
        if row is None:
            stats.skipped_no_parquet_match += 1
            logger.debug("No parquet match for %s", img_path.name)
            continue

        # 2. Read alt-text
        alt_text_path = img_path.with_suffix(".txt")
        alt_text = ""
        if alt_text_path.exists():
            try:
                alt_text = alt_text_path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                alt_text = alt_text_path.read_text(
                    encoding="latin-1"
                ).strip()

        # 3. Extract PaliGemma caption
        pali_caption = str(row.get("caption", row.get("pali_caption", "")))

        yield ImageRecord(
            filename=img_path.name,
            full_path=img_path,
            alt_text=alt_text,
            pali_caption=pali_caption,
            uid=str(row.get("uid", "")),
        )


def run_pipeline(
    image_dir: Path,
    input_parquet: Path,
    output_parquet: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS,
    resume: bool = False,
) -> ProcessingStats:
    """Execute the full caption-refinement pipeline.

    Args:
        image_dir: Directory containing images (traversed recursively).
        input_parquet: Parquet with columns [filename, caption, ...].
        output_parquet: Output Parquet path.
        model_name: Gemini model identifier.
        batch_size: How many images to accumulate before writing partial results.
        rate_limit_rps: Rate limit for API calls.
        resume: If True, skip images already present in *output_parquet*.
    """
    stats = ProcessingStats()

    # Load input data
    df = load_parquet(input_parquet)

    # Resume support: skip already-processed images
    already_done: frozenset = frozenset()
    if resume and output_parquet.exists():
        try:
            existing = pd.read_parquet(output_parquet, engine="pyarrow")
            already_done = frozenset(
                existing["filename"].dropna().astype(str).tolist()
            )
            logger.info(
                "Resume mode: %d images already processed", len(already_done)
            )
        except Exception:
            logger.warning("Could not read existing output for resume; starting fresh.")

    # Initialise Gemini captioner
    captioner = GeminiCaptioner(
        model_name=model_name, rate_limit_rps=rate_limit_rps
    )

    # Accumulators
    results: Dict[str, str] = {}  # filename → gemini_caption
    batch: List[ImageRecord] = []

    def _flush() -> None:
        """Write accumulated results to output parquet."""
        if not results:
            return
        result_df = pd.DataFrame(
            [
                {"filename": fn, "gemini_caption": cap}
                for fn, cap in results.items()
            ]
        )
        merged = df.merge(result_df, on="filename", how="left")
        save_parquet(merged, output_parquet)
        results.clear()

    # Main processing loop
    for record in match_records(image_dir, df, stats):
        # Skip already-done in resume mode
        if record.filename in already_done:
            stats.skipped_missing_image += 1
            continue

        # Edge case: missing image file
        if not record.full_path.exists():
            stats.skipped_missing_image += 1
            logger.warning("Image file missing: %s", record.full_path)
            continue

        # Edge case: invalid / corrupt image
        try:
            Image.open(record.full_path).verify()
        except Exception:
            stats.skipped_invalid_image += 1
            logger.warning("Corrupt/invalid image: %s", record.full_path)
            continue

        batch.append(record)

        if len(batch) >= batch_size:
            new_captions = captioner.caption_batch(batch, stats)
            results.update(new_captions)
            _flush()
            batch.clear()

    # Process remaining items
    if batch:
        new_captions = captioner.caption_batch(batch, stats)
        results.update(new_captions)
        _flush()

    logger.info("Pipeline complete. %s", stats.summary())

    # Write error log
    if stats.errors:
        error_path = output_parquet.with_suffix(".errors.json")
        error_path.write_text(
            json.dumps(stats.errors, indent=2), encoding="utf-8"
        )
        logger.warning("Errors written to %s", error_path)

    return stats


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine image captions using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python gemini_cap_gen.py --image_dir ./images \\\n"
            "      --input_parquet paligemma_captions.parquet \\\n"
            "      --output_parquet refined_captions.parquet\n\n"
            "Requires a .env file with: GEMINI_API_KEY=your_key"
        ),
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Root directory containing images (recursively searched).",
    )
    parser.add_argument(
        "--input_parquet",
        type=Path,
        required=True,
        help="Parquet file with PaliGemma captions (columns: filename, caption).",
    )
    parser.add_argument(
        "--output_parquet",
        type=Path,
        required=True,
        help="Path to save the refined captions parquet.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Images per write batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_RPS,
        help=f"Max API calls per second (default: {DEFAULT_RATE_LIMIT_RPS}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in output_parquet.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only discover images, do not call Gemini.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.image_dir.is_dir():
        logger.error("image_dir is not a directory: %s", args.image_dir)
        sys.exit(1)
    if not args.input_parquet.exists():
        logger.error("input_parquet not found: %s", args.input_parquet)
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run — discovering images (no API calls)...")
        df = load_parquet(args.input_parquet)
        stats = ProcessingStats()
        count = 0
        for rec in match_records(args.image_dir, df, stats):
            count += 1
            print(f"  {rec.filename}")
        logger.info(
            "Discovered %d images. %s", count, stats.summary()
        )
        return

    run_pipeline(
        image_dir=args.image_dir,
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        model_name=args.model,
        batch_size=args.batch_size,
        rate_limit_rps=args.rate_limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()