#!/usr/bin/env python3
"""
make_subs_faster_whisper.py

Generate subtitles (SRT/VTT/TXT) from a video or audio file using faster-whisper.

Requirements:
  - Python 3.8+
  - pip install faster-whisper
  - ffmpeg installed and available in PATH (apt install ffmpeg)

Usage examples:
  python make_subs_faster_whisper.py "video.mp4"
  python make_subs_faster_whisper.py "video.mp4" --model small --language id --format srt --out-dir subs
  python make_subs_faster_whisper.py "video.mp4" --word-timestamps --format vtt

Notes:
  - The model is downloaded automatically on first run (cached in ~/.cache/faster_whisper).
  - For CPU-only machines, compute_type=int8 gives the best speed/quality trade-off.
  - If you have an NVIDIA GPU with CUDA installed, you can try: --device cuda --compute-type float16
"""

import argparse
import os
import math
from pathlib import Path
from datetime import timedelta

from faster_whisper import WhisperModel


def hhmmss_ms(seconds: float, for_srt: bool = True) -> str:
    """
    Convert seconds float to SRT/VTT timestamp format.
    SRT -> HH:MM:SS,mmm
    VTT -> HH:MM:SS.mmm
    """
    if seconds is None or math.isnan(seconds):
        seconds = 0.0
    td = timedelta(seconds=float(seconds))
    # total seconds may exceed 24h; format manually
    total_ms = int(td.total_seconds() * 1000)
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, ms = divmod(rem_ms, 1000)
    if for_srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_srt(segments, out_path: Path, include_words: bool = False):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = hhmmss_ms(seg.start, True)
            end = hhmmss_ms(seg.end, True)
            f.write(f"{i}\n{start} --> {end}\n")
            if include_words and getattr(seg, "words", None):
                line = "".join([w.word for w in seg.words])
            else:
                line = seg.text.strip()
            f.write(line.strip() + "\n\n")


def write_vtt(segments, out_path: Path, include_words: bool = False):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = hhmmss_ms(seg.start, False)
            end = hhmmss_ms(seg.end, False)
            f.write(f"{start} --> {end}\n")
            if include_words and getattr(seg, "words", None):
                line = "".join([w.word for w in seg.words])
            else:
                line = seg.text.strip()
            f.write(line.strip() + "\n\n")


def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from media using faster-whisper.")
    parser.add_argument("input", help="Path to input video/audio file")
    parser.add_argument("--model", default="small", help="Model size/name (tiny/base/small/medium/large-v3 etc.)")
    parser.add_argument("--language", default=None, help="Force language code, e.g., 'id' for Indonesian. Auto-detect if omitted.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Transcribe (same language) or translate to English")
    parser.add_argument("--format", default="srt", choices=["srt", "vtt", "txt"], help="Subtitle format")
    parser.add_argument("--out-dir", default="subs", help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute-type", default="int8", help="Compute type: int8/int8_float16/int16/float16/float32")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--best-of", type=int, default=5, help="Number of candidates when sampling")
    parser.add_argument("--vad", action="store_true", help="Enable VAD filtering (helps cut long silences/noise)")
    parser.add_argument("--min-silence-ms", type=int, default=500, help="VAD: minimum silence (ms) to split")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word-level timestamps (slower)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="CPU threads")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for the model")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / (in_path.stem + f".{args.format}")

    print(f"[INFO] Loading model '{args.model}' (device={args.device}, compute_type={args.compute_type})...")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        num_workers=args.num_workers,
        cpu_threads=args.threads,
    )

    print(f"[INFO] Transcribing: {in_path}")
    segments, info = model.transcribe(
        str(in_path),
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=args.vad,
        vad_parameters={"min_silence_duration_ms": args.min_silence_ms},
        word_timestamps=args.word_timestamps,
    )

    # Consume generator to a list so we can write multiple formats if needed.
    segments = list(segments)

    print(
        f"[INFO] Detected language: {info.language} (prob={info.language_probability:.2f}). "
        f"Duration: {info.duration:.1f}s, Segments: {len(segments)}"
    )

    if args.format == "srt":
        write_srt(segments, out_file, include_words=args.word_timestamps)
    elif args.format == "vtt":
        write_vtt(segments, out_file, include_words=args.word_timestamps)
    elif args.format == "txt":
        write_txt(segments, out_file)
    else:
        raise ValueError("Unsupported format")

    print(f"[OK] Wrote: {out_file.resolve()}")


if __name__ == "__main__":
    main()
