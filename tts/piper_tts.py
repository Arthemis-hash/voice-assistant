"""
Piper TTS wrapper for local neural text-to-speech.

Fast, offline, high-quality speech synthesis.
Supports streaming, voice selection, and audio configuration.
"""

import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default voice cache directory
DEFAULT_VOICE_DIR = Path.home() / ".cache" / "piper" / "voices"


@dataclass
class SynthesisConfig:
    """Configuration for Piper TTS synthesis."""

    volume: float = 1.0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    normalize_audio: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.volume <= 2.0:
            raise ValueError("volume must be between 0.0 and 2.0")
        if not 0.1 <= self.length_scale <= 5.0:
            raise ValueError("length_scale must be between 0.1 and 5.0")
        if not 0.0 <= self.noise_scale <= 1.0:
            raise ValueError("noise_scale must be between 0.0 and 1.0")
        if not 0.0 <= self.noise_w_scale <= 1.0:
            raise ValueError("noise_w_scale must be between 0.0 and 1.0")


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""

    audio_path: str
    duration_seconds: float
    sample_rate: int
    text_length: int
    file_size_bytes: int


class PiperTTS:
    """
    Piper TTS wrapper for local neural text-to-speech.

    Fast, offline, high-quality speech synthesis using ONNX models.
    No API calls, no network, 100% local.

    Usage:
        tts = PiperTTS(voice="en_US-lessac-medium")

        # Synthesize to file
        result = tts.synthesize("Hello world", output_path="output.wav")

        # Synthesize and play
        audio_bytes = tts.synthesize_bytes("Hello world")

        # Stream synthesis (low latency)
        for chunk in tts.synthesize_stream("Hello world"):
            play_audio(chunk)
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        voice_dir: Optional[Path] = None,
        config: Optional[SynthesisConfig] = None,
    ) -> None:
        self.voice = voice
        self.voice_dir = voice_dir or DEFAULT_VOICE_DIR
        self.config = config or SynthesisConfig()
        self.config.validate()
        self._voice = None
        self._model_path: Optional[str] = None
        self._config_path: Optional[str] = None

    def _ensure_voice(self) -> None:
        """Ensure voice model is downloaded and loaded."""
        if self._voice is not None:
            return

        from piper import PiperVoice

        model_path = self.voice_dir / f"{self.voice}.onnx"
        config_path = self.voice_dir / f"{self.voice}.onnx.json"

        if not model_path.exists():
            logger.info(f"Downloading voice: {self.voice}")
            self._download_voice(self.voice, self.voice_dir)

        self._model_path = str(model_path)
        self._config_path = str(config_path)
        self._voice = PiperVoice.load(self._model_path, config_path=self._config_path)
        logger.info(f"Voice loaded: {self.voice}")

    @staticmethod
    def _download_voice(voice_name: str, target_dir: Path) -> None:
        """Download a Piper voice model."""
        import subprocess
        import sys

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "piper.download_voices",
                    voice_name,
                    "--download-dir",
                    str(target_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Voice download timed out: {voice_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download voice: {e.stderr}") from e

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        config: Optional[SynthesisConfig] = None,
    ) -> TTSResult:
        """
        Synthesize speech and save to WAV file.

        Args:
            text: Text to synthesize
            output_path: Output WAV file path (auto-generated if None)
            config: Override synthesis config

        Returns:
            TTSResult with metadata

        Raises:
            RuntimeError: If synthesis fails
        """
        self._ensure_voice()
        syn_config = config or self.config
        syn_config.validate()

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="piper_")
            os.close(fd)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            with wave.open(str(output), "wb") as wav_file:
                self._voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=syn_config,
                )

            file_size = output.stat().st_size

            # Calculate duration from WAV file
            with wave.open(str(output), "rb") as wav_in:
                frames = wav_in.getnframes()
                rate = wav_in.getframerate()
                duration = frames / rate if rate > 0 else 0.0

            return TTSResult(
                audio_path=str(output),
                duration_seconds=duration,
                sample_rate=rate,
                text_length=len(text),
                file_size_bytes=file_size,
            )

        except Exception as e:
            raise RuntimeError(f"TTS synthesis failed: {e}") from e

    def synthesize_bytes(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
    ) -> bytes:
        """
        Synthesize speech and return raw WAV bytes.

        Args:
            text: Text to synthesize
            config: Override synthesis config

        Returns:
            WAV audio as bytes
        """
        self._ensure_voice()
        syn_config = config or self.config
        syn_config.validate()

        import io

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)

        return buffer.getvalue()

    def synthesize_stream(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
    ) -> Generator[bytes, None, None]:
        """
        Stream synthesis for low-latency playback.

        Yields raw audio chunks (PCM int16) as they're generated.
        Ideal for real-time voice agents.

        Args:
            text: Text to synthesize
            config: Override synthesis config

        Yields:
            Audio chunks as bytes (PCM int16)
        """
        self._ensure_voice()
        syn_config = config or self.config
        syn_config.validate()

        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            yield chunk.audio_int16_bytes

    def synthesize_to_numpy(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech and return as numpy array.

        Args:
            text: Text to synthesize
            config: Override synthesis config

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._ensure_voice()
        syn_config = config or self.config
        syn_config.validate()

        audio_chunks = []
        sample_rate = None

        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            audio_chunks.append(
                np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            )
            if sample_rate is None:
                sample_rate = chunk.sample_rate

        if not audio_chunks:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(audio_chunks)
        return audio, sample_rate or 22050

    @staticmethod
    def list_available_voices() -> list[str]:
        """List voices available in the voice cache directory."""
        voice_dir = DEFAULT_VOICE_DIR
        if not voice_dir.exists():
            return []

        voices = []
        for f in voice_dir.glob("*.onnx"):
            name = f.stem.replace(".onnx", "")
            voices.append(name)

        return sorted(voices)

    @staticmethod
    def get_voice_info(voice_name: str) -> dict:
        """Get information about a specific voice."""
        voice_map = {
            "en_US-lessac-medium": {
                "language": "English (US)",
                "quality": "Medium",
                "speaker": "Lessac",
                "description": "High-quality female voice",
            },
            "en_US-ryan-high": {
                "language": "English (US)",
                "quality": "High",
                "speaker": "Ryan",
                "description": "Deep male voice",
            },
            "fr_FR-upmc-medium": {
                "language": "French",
                "quality": "Medium",
                "speaker": "UPMC",
                "description": "French voice",
            },
            "de_DE-mls-medium": {
                "language": "German",
                "quality": "Medium",
                "speaker": "MLS",
                "description": "German voice",
            },
            "es_ES-carlfm-medium": {
                "language": "Spanish",
                "quality": "Medium",
                "speaker": "CarlFM",
                "description": "Spanish voice",
            },
        }
        return voice_map.get(
            voice_name,
            {"language": "Unknown", "quality": "Unknown", "speaker": "Unknown"},
        )
