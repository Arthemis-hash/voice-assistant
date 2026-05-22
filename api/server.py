"""
Voice Pipeline HTTP API Server.

FastAPI-based REST API for programmatic access to the voice pipeline.
Supports text processing, audio processing, and system management.

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.voice_agent import VoicePipeline, PipelineConfig
from pipeline.lm_studio_provider import LMStudioProvider
from pipeline.database import Database
from stt.apple_stt import AppleSTT, STTConfig
from tts.piper_tts import PiperTTS, SynthesisConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
pipeline: Optional[VoicePipeline] = None
is_running = False


# --- Request/Response Models ---

class TextRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt for LLM")
    output_path: Optional[str] = Field(None, description="Optional output audio path")


class AudioRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt for LLM")
    output_path: Optional[str] = Field(None, description="Optional output audio path")


class ConfigUpdate(BaseModel):
    stt_language: Optional[str] = Field(None, description="STT language code")
    tts_voice: Optional[str] = Field(None, description="TTS voice name")
    llm_url: Optional[str] = Field(None, description="LM Studio API URL")
    llm_model: Optional[str] = Field(None, description="LLM model name")
    output_dir: Optional[str] = Field(None, description="Output directory")


class PipelineResponse(BaseModel):
    input_text: str
    llm_response: str
    output_audio_path: str
    output_duration: float


class StatusResponse(BaseModel):
    is_running: bool
    stt_status: str
    llm_status: str
    tts_status: str
    config: dict


class SystemInfo(BaseModel):
    version: str = "1.0.0"
    stt_engine: str = "Apple Speech Framework"
    tts_engine: str = "Piper Neural TTS"
    llm_provider: str = "LM Studio (HTTP API)"
    supported_languages: list[str]
    available_voices: list[str]


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup pipeline on startup/shutdown."""
    global pipeline, is_running

    logger.info("Starting Voice Pipeline API server...")
    is_running = True

    try:
        llm = LMStudioProvider(
            base_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            model=os.getenv("LM_STUDIO_MODEL", "Qwen3-4B"),
        )

        pipeline = VoicePipeline(
            llm_provider=llm,
            config=PipelineConfig(
                stt_language=os.getenv("STT_LANGUAGE", "en-US"),
                tts_voice=os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
                output_dir=os.getenv("OUTPUT_DIR", "./output"),
            ),
        )
        logger.info("Voice Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None

    yield

    logger.info("Shutting down Voice Pipeline API server...")
    is_running = False


# --- App ---

app = FastAPI(
    title="Voice Pipeline API",
    description="100% local voice AI pipeline with STT, LLM, and TTS",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/", tags=["System"])
async def root():
    """API root - returns system info."""
    return {
        "name": "Voice Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get current system status."""
    stt_status = "online"  # Apple Speech is always available

    llm_status = "unknown"
    tts_status = "unknown"

    if pipeline:
        try:
            if pipeline.llm.is_available():
                llm_status = "online"
            else:
                llm_status = "offline"
        except Exception:
            llm_status = "offline"

        try:
            tts_status = "online"
        except Exception:
            tts_status = "offline"

    return StatusResponse(
        is_running=is_running,
        stt_status=stt_status,
        llm_status=llm_status,
        tts_status=tts_status,
        config={
            "stt_language": os.getenv("STT_LANGUAGE", "en-US"),
            "tts_voice": os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
            "llm_url": os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            "llm_model": os.getenv("LM_STUDIO_MODEL", "Qwen3-4B"),
            "output_dir": os.getenv("OUTPUT_DIR", "./output"),
        },
    )


@app.get("/info", response_model=SystemInfo, tags=["System"])
async def get_info():
    """Get system information."""
    from stt.apple_stt import AppleSTT
    from tts.piper_tts import PiperTTS

    return SystemInfo(
        supported_languages=list(AppleSTT.list_supported_languages().keys()),
        available_voices=PiperTTS.list_available_voices(),
    )


@app.post("/process_text", response_model=PipelineResponse, tags=["Pipeline"])
async def process_text(request: TextRequest):
    """
    Process text through the pipeline (skip STT).

    - Takes text input
    - Sends to LLM
    - Generates audio response via TTS
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.process_text(
            text=request.text,
            system_prompt=request.system_prompt,
            output_path=request.output_path,
        )

        return PipelineResponse(
            input_text=result.input_text,
            llm_response=result.llm_response,
            output_audio_path=result.output_audio_path,
            output_duration=result.output_duration,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/process_audio", response_model=PipelineResponse, tags=["Pipeline"])
async def process_audio(request: AudioRequest):
    """
    Process audio file through the full pipeline.

    - Transcribes audio (STT)
    - Sends to LLM
    - Generates audio response via TTS
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not Path(request.audio_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")

    try:
        result = pipeline.process_audio(
            audio_path=request.audio_path,
            system_prompt=request.system_prompt,
            output_path=request.output_path,
        )

        return PipelineResponse(
            input_text=result.input_text,
            llm_response=result.llm_response,
            output_audio_path=result.output_audio_path,
            output_duration=result.output_duration,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upload_audio", response_model=PipelineResponse, tags=["Pipeline"])
async def upload_audio(
    file: UploadFile = File(...),
    system_prompt: Optional[str] = None,
):
    """
    Upload and process an audio file.

    - Uploads audio file
    - Processes through full pipeline
    - Returns response with audio path
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Save uploaded file
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_path = output_dir / f"upload_{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = pipeline.process_audio(
            audio_path=str(temp_path),
            system_prompt=system_prompt,
        )

        return PipelineResponse(
            input_text=result.input_text,
            llm_response=result.llm_response,
            output_audio_path=result.output_audio_path,
            output_duration=result.output_duration,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/output/{filename}", tags=["Output"])
async def get_output_file(filename: str):
    """Retrieve a generated audio file."""
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@app.post("/config", tags=["System"])
async def update_config(config: ConfigUpdate):
    """Update pipeline configuration."""
    if config.stt_language:
        os.environ["STT_LANGUAGE"] = config.stt_language
    if config.tts_voice:
        os.environ["PIPER_VOICE"] = config.tts_voice
    if config.llm_url:
        os.environ["LM_STUDIO_URL"] = config.llm_url
    if config.llm_model:
        os.environ["LM_STUDIO_MODEL"] = config.llm_model
    if config.output_dir:
        os.environ["OUTPUT_DIR"] = config.output_dir

    return {"message": "Configuration updated", "config": get_status().config}


@app.get("/voices", tags=["System"])
async def list_voices():
    """List available TTS voices."""
    from tts.piper_tts import PiperTTS

    voices = PiperTTS.list_available_voices()
    return {"voices": voices, "count": len(voices)}


@app.get("/languages", tags=["System"])
async def list_languages():
    """List supported STT languages."""
    from stt.apple_stt import AppleSTT

    return AppleSTT.list_supported_languages()


@app.get("/history", tags=["Database"])
async def get_history(limit: int = 50):
    """Get conversation history."""
    db = Database()
    conversations = db.list_conversations(limit=limit)
    return {"conversations": conversations, "count": len(conversations)}


@app.get("/history/search", tags=["Database"])
async def search_history(query: str, limit: int = 20):
    """Search conversation history."""
    db = Database()
    conversations = db.search_conversations(query, limit=limit)
    return {"conversations": conversations, "count": len(conversations)}


@app.get("/stats", tags=["Database"])
async def get_stats():
    """Get pipeline statistics."""
    db = Database()
    return db.get_stats()


@app.delete("/history", tags=["Database"])
async def clear_history():
    """Clear all conversation history."""
    db = Database()
    count = db.clear_conversations()
    return {"message": f"Cleared {count} conversations"}


@app.get("/sessions", tags=["Database"])
async def get_sessions(limit: int = 50):
    """Get session history."""
    db = Database()
    sessions = db.list_sessions(limit=limit)
    return {"sessions": sessions, "count": len(sessions)}


# --- Run ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
