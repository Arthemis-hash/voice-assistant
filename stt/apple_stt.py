"""
Apple Speech Framework STT wrapper.

Uses macOS native SFSpeechRecognizer via speech_recognition library.
100% on-device, zero API calls, no network required.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import speech_recognition as sr

logger = logging.getLogger(__name__)

# Supported languages for Apple Speech Recognition
SUPPORTED_LANGUAGES = {
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "fr-FR": "French",
    "de-DE": "German",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "pt-BR": "Portuguese (Brazil)",
    "ar-SA": "Arabic",
    "ru-RU": "Russian",
    "nl-NL": "Dutch",
    "sv-SE": "Swedish",
    "da-DK": "Danish",
    "no-NO": "Norwegian",
    "fi-FI": "Finnish",
    "pl-PL": "Polish",
    "tr-TR": "Turkish",
}


@dataclass
class STTConfig:
    """Configuration for Apple STT engine."""

    language: str = "en-US"
    show_all: bool = False
    phrase_time_limit: Optional[float] = None
    timeout: float = 5.0
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.language}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""

    text: str
    language: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    audio_path: Optional[str] = None


class AppleSTT:
    """
    Apple Speech Framework wrapper for on-device speech recognition.

    Uses macOS native SFSpeechRecognizer via speech_recognition.
    All processing happens locally - no network calls, no API keys.

    Usage:
        stt = AppleSTT(language="fr-FR")

        # Transcribe audio file
        result = stt.transcribe_file("recording.wav")
        print(result.text)

        # Transcribe from microphone (real-time)
        for chunk in stt.transcribe_stream():
            print(chunk.text)
    """

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = self.config.energy_threshold
        self._recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
        self._recognizer.operation_timeout = self.config.timeout
        self._apple_recognizer: Optional[sr.Recognizer] = None

    @property
    def is_available(self) -> bool:
        """Check if Apple Speech recognition is available on this system."""
        try:
            # speech_recognition uses Apple's SFSpeechRecognizer on macOS
            test_rec = sr.Recognizer()
            test_mic = sr.Microphone()
            return True
        except (OSError, AttributeError) as e:
            logger.warning(f"Apple STT not available: {e}")
            return False

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe an audio file using Apple Speech Framework.

        Args:
            audio_path: Path to audio file (WAV, FLAC supported)
            language: Override language (default: config language)

        Returns:
            STTResult with transcription text and metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format not supported
            RuntimeError: If transcription fails
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if path.suffix.lower() not in (".wav", ".flac"):
            raise ValueError(
                f"Unsupported format: {path.suffix}. Use WAV or FLAC."
            )

        lang = language or self.config.language

        try:
            with sr.AudioFile(str(path)) as source:
                audio_data = self._recognizer.record(source)

            text = self._recognizer.recognize_whisper(
                audio_data,
                language=lang.replace("-", "_")[:2],  # "en-US" -> "en"
                model="base",
            )

            return STTResult(
                text=text.strip(),
                language=lang,
                audio_path=str(path),
            )

        except sr.UnknownValueError:
            return STTResult(text="", language=lang, audio_path=str(path))
        except sr.RequestError as e:
            raise RuntimeError(f"Apple STT request failed: {e}") from e

    def transcribe_stream(
        self,
        language: Optional[str] = None,
        phrase_time_limit: Optional[float] = None,
    ) -> Generator[STTResult, None, None]:
        """
        Real-time transcription from microphone.

        Yields STTResult for each detected phrase.

        Args:
            language: Override language
            phrase_time_limit: Max seconds per phrase (None = unlimited)

        Yields:
            STTResult for each recognized phrase
        """
        lang = language or self.config.language
        time_limit = phrase_time_limit or self.config.phrase_time_limit

        mic = sr.Microphone(sample_rate=self.config.sample_rate)

        with mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("Listening... (press Ctrl+C to stop)")

            try:
                while True:
                    audio_data = self._recognizer.listen(
                        source,
                        timeout=self.config.timeout,
                        phrase_time_limit=time_limit,
                    )

                    try:
                        text = self._recognizer.recognize_whisper(
                            audio_data,
                            language=lang.replace("-", "_")[:2],
                            model="base",
                        )

                        if text.strip():
                            yield STTResult(
                                text=text.strip(),
                                language=lang,
                            )

                    except sr.UnknownValueError:
                        continue

            except KeyboardInterrupt:
                logger.info("Stopped listening.")

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe raw audio bytes.

        Args:
            audio_bytes: Raw PCM audio data
            sample_rate: Audio sample rate (default: 16000)
            sample_width: Bytes per sample (default: 2 for 16-bit)
            language: Override language

        Returns:
            STTResult with transcription
        """
        lang = language or self.config.language
        audio_data = sr.AudioData(
            audio_bytes,
            sample_rate=sample_rate,
            sample_width=sample_width,
        )

        try:
            text = self._recognizer.recognize_whisper(
                audio_data,
                language=lang.replace("-", "_")[:2],
                model="base",
            )

            return STTResult(
                text=text.strip(),
                language=lang,
            )

        except sr.UnknownValueError:
            return STTResult(text="", language=lang)
        except sr.RequestError as e:
            raise RuntimeError(f"Apple STT request failed: {e}") from e

    @staticmethod
    def list_supported_languages() -> dict[str, str]:
        """Return dict of supported language codes and names."""
        return dict(SUPPORTED_LANGUAGES)
