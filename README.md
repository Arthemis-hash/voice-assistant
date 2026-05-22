# Voice Pipeline

100% local, offline voice AI pipeline for macOS Apple Silicon.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VOICE PIPELINE                                │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │   STT       │    │    LLM       │    │        TTS              │  │
│  │             │    │              │    │                         │  │
│  │ Apple       │───▶│  LM Studio   │───▶│      Piper              │  │
│  │ Speech      │    │  (HTTP API)  │    │  (Neural, Local)        │  │
│  │ Framework   │    │              │    │                         │  │
│  │             │    │              │    │                         │  │
│  │ On-device   │    │  Your choice │    │  ONNX model             │  │
│  │ Zero API    │    │  Local       │    │  100% offline           │  │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘  │
│                                                                      │
│  Audio In ──▶ Text ──▶ Response ──▶ Audio Out                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology | Type | Latency |
|-----------|------------|------|---------|
| **STT** | Apple Speech Framework (`SFSpeechRecognizer`) | On-device, native macOS | ~100ms |
| **LLM** | LM Studio (HTTP API) or any OpenAI-compatible API | Local or remote | Varies |
| **TTS** | Piper neural TTS (ONNX) | Local, offline | ~200ms |

**Total pipeline latency:** ~300ms + LLM inference time

## Technical Details

### STT: Apple Speech Framework

- Uses macOS native `SFSpeechRecognizer` via `speech_recognition` Python library
- **On-device processing** — audio never leaves the Mac
- **Zero API calls** — no network required
- **Privacy-first** — all transcription happens locally
- Supports 20+ languages
- Requires microphone permission on macOS

### LLM: LM Studio (HTTP Server)

- LM Studio serves local models via OpenAI-compatible HTTP API
- Default endpoint: `http://localhost:1234/v1`
- Supports streaming responses
- Any OpenAI-compatible API works (Ollama, vLLM, etc.)

#### Supported Models (via LM Studio)

Download any of these models in LM Studio:

| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| **Qwen3-4B** | ~2.5GB | Excellent | General purpose, fast |
| **Qwen3-8B** | ~5GB | Excellent | Best quality/speed ratio |
| **Llama-3.2-3B** | ~2GB | Good | Lightweight, fast |
| **Llama-3.1-8B** | ~5GB | Excellent | General purpose |
| **Mistral-7B** | ~4GB | Good | Balanced |
| **Phi-3-mini** | ~2GB | Good | Ultra-fast, low memory |
| **Gemma-2-9B** | ~6GB | Excellent | High quality |

**Recommended:** `Qwen3-4B` or `Qwen3-8B` for best performance on Apple Silicon.

#### LM Studio Setup

1. Download LM Studio: https://lmstudio.ai
2. Search and download a model (e.g., `Qwen3-4B`)
3. Start the HTTP server (default: `http://localhost:1234`)
4. Note the model name shown in LM Studio

### TTS: Piper Neural TTS

- ONNX-based neural TTS engine
- **100% offline** — no network calls
- **Fast** — real-time synthesis on Apple Silicon
- **High quality** — natural-sounding voices
- Supports voice cloning (with reference audio)

#### Available Voices

Download voices via: `python3 -m piper.download_voices <voice_name>`

| Voice | Language | Quality | Description |
|-------|----------|---------|-------------|
| `en_US-lessac-medium` | English (US) | Medium | Female, clear |
| `en_US-ryan-high` | English (US) | High | Male, deep |
| `fr_FR-upmc-medium` | French | Medium | Standard French |
| `fr_FR-tom-medium` | French | Medium | Alternative French |
| `de_DE-mls-medium` | German | Medium | Standard German |
| `es_ES-carlfm-medium` | Spanish | Medium | Castilian Spanish |
| `it_IT-riccardo-medium` | Italian | Medium | Standard Italian |
| `ja_JP-kamitsukichi-medium` | Japanese | Medium | Standard Japanese |

Full list: https://rhasspy.github.io/piper-samples/

## Quick Start

```bash
# 1. Activate virtual environment
cd voice-pipeline
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download a Piper voice
python3 -m piper.download_voices en_US-lessac-medium

# 4. Start LM Studio with a model (e.g., Qwen3-4B)
#    Server: http://localhost:1234

# 5. Launch TUI
python3 -m tui.app
```

## Interfaces

### 1. TUI (Terminal User Interface)

Launch the interactive terminal interface:

```bash
python3 -m tui.app
```

**Features:**
- **Dashboard** - System status, start/stop controls
- **Conversation** - Interactive chat with LLM
- **Settings** - Configure language, voice, LLM endpoint
- **Logs** - View system logs

**Keyboard shortcuts:**
- `1` - Dashboard
- `2` - Conversation
- `3` - Settings
- `4` - Logs
- `d` - Toggle dark mode
- `q` - Quit

### 2. HTTP API

Launch the REST API server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**API Documentation:** http://localhost:8000/docs

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/status` | System status |
| `GET` | `/info` | System information |
| `POST` | `/process_text` | Process text through pipeline |
| `POST` | `/process_audio` | Process audio file |
| `POST` | `/upload_audio` | Upload and process audio |
| `GET` | `/output/{filename}` | Download generated audio |
| `POST` | `/config` | Update configuration |
| `GET` | `/voices` | List available voices |
| `GET` | `/languages` | List supported languages |

**Example API usage:**

```bash
# Process text
curl -X POST http://localhost:8000/process_text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the capital of France?", "system_prompt": "Be concise."}'

# Check status
curl http://localhost:8000/status

# List voices
curl http://localhost:8000/voices
```

### 3. Python SDK

```python
from pipeline.voice_agent import VoicePipeline, PipelineConfig
from pipeline.lm_studio_provider import LMStudioProvider

# Connect to LM Studio
llm = LMStudioProvider(
    base_url="http://localhost:1234/v1",
    model="Qwen3-4B",
)

# Create pipeline
pipeline = VoicePipeline(
    llm_provider=llm,
    config=PipelineConfig(
        stt_language="fr-FR",
        tts_voice="fr_FR-upmc-medium",
    ),
)

# Process text
result = pipeline.process_text("Bonjour, comment allez-vous?")
print(result.llm_response)
```

## Project Structure

```
voice-pipeline/
├── .venv/                      # Python virtual environment
├── stt/
│   └── apple_stt.py            # Apple Speech STT wrapper
├── tts/
│   └── piper_tts.py            # Piper TTS wrapper
├── pipeline/
│   ├── voice_agent.py          # Pipeline orchestrator
│   ├── lm_studio_provider.py   # LM Studio LLM provider
│   └── database.py             # SQLite database layer
├── tui/
│   └── app.py                  # Terminal UI (Textual)
├── api/
│   └── server.py               # HTTP API (FastAPI)
├── examples/
│   └── basic_usage.py          # Usage example
├── requirements.txt
├── .gitignore
└── README.md
```

## Database

SQLite database for persistence (zero-config, file-based):

**Location:** `~/.voice_pipeline/voice_pipeline.db`

**Stored data:**
- Conversation history (user text, LLM response, audio path)
- Session tracking (turn count, duration)
- Configuration (persistent settings)
- Audio file metadata

**API endpoints:**
- `GET /history` - Get conversation history
- `GET /history/search?query=...` - Search conversations
- `GET /stats` - Pipeline statistics
- `DELETE /history` - Clear history
- `GET /sessions` - Session history

**TUI:** Press `5` to view conversation history.

## Requirements

- **macOS 12+** (Monterey or later)
- **Python 3.10+**
- **Apple Silicon** (M1/M2/M3) recommended
- **LM Studio** (or any OpenAI-compatible API server)
- **Microphone permission** for STT

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LM_STUDIO_MODEL` | `Qwen3-4B` | Model name to use |
| `PIPER_VOICE` | `en_US-lessac-medium` | Default TTS voice |
| `STT_LANGUAGE` | `en-US` | Default STT language |
| `OUTPUT_DIR` | `./output` | Output audio directory |
# voice-assistant
