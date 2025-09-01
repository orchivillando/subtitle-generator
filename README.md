# Subtitle Generator

A Python project that generates subtitles from video and audio files using OpenAI's Whisper model via the faster-whisper library. The project provides both a command-line interface and a web-based GUI.

## Features

- 🎥 Support for multiple video formats (MP4, MKV, MOV)
- 🎵 Support for multiple audio formats (WAV, MP3, M4A)
- 📝 Multiple subtitle formats (SRT, VTT, TXT)
- 🚀 Fast processing with faster-whisper
- 🖥️ Command-line interface for batch processing
- 🌐 Web GUI using Streamlit for easy use
- 🎯 Multiple Whisper model sizes (tiny to large-v3)
- 🔧 CPU and GPU support
- 🎙️ Voice Activity Detection (VAD) filtering
- ⏱️ Word-level timestamps support
- 🌍 Auto language detection or forced language

## Installation

### Prerequisites

1. **Python 3.8+** is required
2. **FFmpeg** must be installed on your system:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS (with Homebrew)
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd subtitle-generator
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Generate subtitles using the command-line tool:

```bash
# Basic usage
python make_subs_faster_whisper.py "video.mp4"

# With custom options
python make_subs_faster_whisper.py "video.mp4" \
    --model small \
    --language id \
    --format srt \
    --out-dir subs

# Enable word timestamps and VTT format
python make_subs_faster_whisper.py "video.mp4" \
    --word-timestamps \
    --format vtt
```

#### Command Line Options

- `--model`: Model size (tiny/base/small/medium/large-v3) - default: small
- `--language`: Force language code (e.g., 'id', 'en') - default: auto-detect
- `--task`: transcribe or translate - default: transcribe
- `--format`: Output format (srt/vtt/txt) - default: srt
- `--out-dir`: Output directory - default: subs
- `--device`: Device (auto/cpu/cuda) - default: auto
- `--compute-type`: Compute type (int8/float16/etc.) - default: int8
- `--word-timestamps`: Enable word-level timestamps
- `--vad`: Enable Voice Activity Detection
- `--beam-size`: Beam size for decoding - default: 5

### Web GUI

Launch the Streamlit web interface:

```bash
streamlit run streamlit_faster_whisper_app.py
```

Then open your browser to `http://localhost:8501` to use the GUI.

The web interface provides:
- File upload for small files
- Local file path input for large files
- All command-line options in an easy-to-use interface
- Real-time progress tracking
- Download generated subtitles

## Project Structure

```
subtitle-generator/
├── make_subs_faster_whisper.py    # Command-line tool
├── streamlit_faster_whisper_app.py # Web GUI
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── subs/                         # Output directory for subtitles
└── .venv/                        # Virtual environment (created during setup)
```

## Model Information

The project uses faster-whisper, which provides several model sizes:

- **tiny**: Fastest, least accurate (~1GB VRAM)
- **base**: Good speed/accuracy balance (~1GB VRAM)
- **small**: Recommended for most use cases (~2GB VRAM)
- **medium**: Better accuracy (~5GB VRAM)
- **large-v3**: Best accuracy (~10GB VRAM)

Models are automatically downloaded and cached on first use.

## Performance Tips

- For CPU-only machines: Use `--compute-type int8` (default)
- For NVIDIA GPUs: Use `--device cuda --compute-type float16`
- Enable VAD (`--vad`) to handle long silences and noise
- Use smaller models (tiny/base) for faster processing
- Use larger models (medium/large-v3) for better accuracy

## Supported Languages

The model supports auto-detection and can process audio in many languages. You can force a specific language using language codes like:
- `en` (English)
- `id` (Indonesian)
- `es` (Spanish)
- `fr` (French)
- And many more...

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Make sure FFmpeg is installed and available in your PATH
2. **CUDA errors**: Ensure you have CUDA installed if using GPU acceleration
3. **Memory errors**: Try using a smaller model or reduce batch size
4. **Slow processing**: Enable VAD filtering or use a smaller model

### Getting Help

[Add contact information or issue reporting guidelines]
