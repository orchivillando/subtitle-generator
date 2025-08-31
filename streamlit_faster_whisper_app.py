"""
Streamlit GUI for faster-whisper

Single-file Streamlit app that exposes the important parameters and runs
transcription/subtitle generation using faster-whisper.

Usage:
  1. Create and activate a virtualenv
     python3 -m venv .venv
     source .venv/bin/activate

  2. Install requirements
     pip install streamlit faster-whisper ffmpeg-python
     # ffmpeg binary should be installed via apt: sudo apt install ffmpeg

  3. Run the app
     streamlit run streamlit_faster_whisper_app.py

Notes:
 - The app accepts either a local file path (server-side) or an uploaded file.
 - For large files prefer running the app on the machine that has access to the files.
 - Default model: small (fast & decent on CPU). Use medium/large if you have RAM/time.
"""

import os
import tempfile
import shutil
import math
from pathlib import Path
from datetime import timedelta

import streamlit as st
from faster_whisper import WhisperModel


def hhmmss_ms(seconds: float, for_srt: bool = True) -> str:
    if seconds is None or math.isnan(seconds):
        seconds = 0.0
    td = timedelta(seconds=float(seconds))
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
                line = " ".join([w.word for w in seg.words])
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
                line = " ".join([w.word for w in seg.words])
            else:
                line = seg.text.strip()
            f.write(line.strip() + "\n\n")


def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


st.set_page_config(page_title="faster-whisper Subtitles GUI", layout="wide")
st.title("faster-whisper — Subtitle Generator (GUI)")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input")
    input_mode = st.radio("Input source:", ["Upload file (small)", "Local file path (server)"])

    input_file = None
    uploaded_file = None
    if input_mode == "Upload file (small)":
        uploaded_file = st.file_uploader("Upload video/audio file", type=["mp4", "mkv", "mov", "wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_file is not None:
            tmp_dir = tempfile.mkdtemp(prefix="fw_upload_")
            tmp_path = Path(tmp_dir) / uploaded_file.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            input_file = str(tmp_path)
            st.info(f"Uploaded and saved to: {input_file}")

    else:
        input_file = st.text_input("Absolute local path to file (on server):", value="")
        if input_file:
            if not Path(input_file).exists():
                st.warning("File not found on server. Make sure Streamlit server has access to the path.")

    st.markdown("---")

    st.header("Output & Options")
    out_dir = st.text_input("Output directory (server)", value=str(Path.cwd() / "subs"))
    fmt = st.selectbox("Subtitle format", ["srt", "vtt", "txt"], index=0)
    download_name = st.text_input("Output filename (without extension)", value="output_subs")

with col2:
    st.header("Model & Performance")
    model_name = st.selectbox("Model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    compute_type = st.selectbox("Compute type (CPU optim)", ["int8", "int8_float16", "int16", "float16", "float32"], index=0)
    threads = st.number_input("CPU threads", min_value=1, max_value=os.cpu_count() or 8, value=os.cpu_count() or 4)
    num_workers = st.number_input("Model num_workers", min_value=1, max_value=8, value=1)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.checkbox("Force language (leave empty to auto-detect)", key="force_lang_chk")
    language = st.text_input("Language code (e.g. id, en)", value="", key="lang_input")
    task = st.selectbox("Task", ["transcribe", "translate"], index=0)
    vad = st.checkbox("Enable VAD (split on silence)", value=True)
    min_silence_ms = st.number_input("Min silence (ms) for VAD", value=500, step=100)
    word_ts = st.checkbox("Word-level timestamps (slower)", value=False)

with col4:
    st.info("Advanced settings")
    beam_size = st.number_input("Beam size", min_value=1, max_value=20, value=5)
    best_of = st.number_input("Best of", min_value=1, max_value=20, value=5)

st.markdown("---")

run_button = st.button("Start transcription / generate subtitles")

if run_button:
    if not input_file:
        st.error("Please specify or upload an input file first.")
    else:
        out_path_dir = Path(out_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)
        out_file_path = out_path_dir / (download_name + "." + fmt)

        st.info(f"Loading model: {model_name} (device={device}, compute_type={compute_type})")

        try:
            with st.spinner("Loading model (may take a while first run)..."):
                model = WhisperModel(model_name, device=device, compute_type=compute_type, num_workers=int(num_workers), cpu_threads=int(threads))

            st.success("Model loaded")

            status_bar = st.progress(0)
            log = st.empty()

            # Run transcription
            log.text("Starting transcription...")
            segments_gen, info = model.transcribe(
                str(input_file),
                language=language or None,
                task=task,
                beam_size=int(beam_size),
                best_of=int(best_of),
                vad_filter=bool(vad),
                vad_parameters={"min_silence_duration_ms": int(min_silence_ms)},
                word_timestamps=bool(word_ts),
            )

            segments = list(segments_gen)
            log.text(f"Detected language: {info.language} (prob={info.language_probability:.2f}), duration {info.duration:.1f}s, segments: {len(segments)}")

            # write file
            if fmt == "srt":
                write_srt(segments, out_file_path, include_words=word_ts)
            elif fmt == "vtt":
                write_vtt(segments, out_file_path, include_words=word_ts)
            else:
                write_txt(segments, out_file_path)

            st.success(f"Wrote: {out_file_path}")

            # provide download
            try:
                with open(out_file_path, "rb") as f:
                    st.download_button(label="Download subtitle", data=f, file_name=out_file_path.name)
            except Exception as e:
                st.warning(f"Cannot provide download through browser: {e}. You can find file on server at {out_file_path}")

        except Exception as e:
            st.error(f"Error during processing: {e}")

# Cleanup any temporary uploaded files older than a day (best-effort)
try:
    tmp_base = Path(tempfile.gettempdir())
    for p in tmp_base.glob("fw_upload_*"):
        try:
            # remove if older than 24 hours
            if p.is_dir() and (p.stat().st_mtime < (os.times().system - 86400 if hasattr(os, 'times') else 0)):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
except Exception:
    pass