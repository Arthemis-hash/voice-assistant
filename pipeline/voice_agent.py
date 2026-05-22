"""
Voice pipeline orchestrator.

Connects STT (Apple Speech) → LLM (user's choice) → TTS (Piper).
Fully local, no external API calls. Auto-logs conversations to SQLite.
"""

import logging
import os
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable, Generator, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt.apple_stt import AppleSTT, STTConfig, STTResult
from tts.piper_tts import PiperTTS, SynthesisConfig, TTSResult
from pipeline.database import Database, Conversation, Session

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the voice pipeline."""

    stt_language: str = "en-US"
    stt_energy_threshold: int = 300
    stt_timeout: float = 5.0
    tts_voice: str = "en_US-lessac-medium"
    tts_voice_dir: Optional[str] = None
    tts_volume: float = 1.0
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_noise_w_scale: float = 0.8
    output_dir: str = "./output"
    log_level: str = "INFO"


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface to connect your preferred LLM.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response from the LLM."""
        ...

    @abstractmethod
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate a streaming text response from the LLM."""
        ...


class VoicePipeline:
    """
    Orchestrates the full voice pipeline: STT → LLM → TTS.

    All components run locally. No external API calls.

    Usage:
        pipeline = VoicePipeline(
            llm_provider=my_llm,
            config=PipelineConfig(
                stt_language="fr-FR",
                tts_voice="fr_FR-upmc-medium",
            ),
        )

        # Interactive conversation mode
        pipeline.conversation_loop()

        # Process a single audio file
        result = pipeline.process_audio("input.wav")
        print(f"Response: {result.response_text}")
        print(f"Audio saved: {result.output_audio_path}")
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[PipelineConfig] = None,
        db: Optional[Database] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.llm = llm_provider
        self.db = db or Database()
        self._session: Optional[Session] = None
        self._turn_count = 0

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Initialize STT
        self.stt = AppleSTT(
            config=STTConfig(
                language=self.config.stt_language,
                energy_threshold=self.config.stt_energy_threshold,
                timeout=self.config.stt_timeout,
            )
        )

        # Initialize TTS
        voice_dir = None
        if self.config.tts_voice_dir:
            voice_dir = Path(self.config.tts_voice_dir)

        self.tts = PiperTTS(
            voice=self.config.tts_voice,
            voice_dir=voice_dir,
            config=SynthesisConfig(
                volume=self.config.tts_volume,
                length_scale=self.config.tts_length_scale,
                noise_scale=self.config.tts_noise_scale,
                noise_w_scale=self.config.tts_noise_w_scale,
            ),
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Start session
        self._session = self.db.create_session()

        logger.info("Voice pipeline initialized")
        logger.info(f"  STT: Apple Speech ({self.config.stt_language})")
        logger.info(f"  TTS: Piper ({self.config.tts_voice})")
        logger.info(f"  Session: {self._session.id}")

    def process_audio(
        self,
        audio_path: str,
        system_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> "PipelineResult":
        """
        Process an audio file through the full pipeline.

        Args:
            audio_path: Path to input audio file
            system_prompt: Optional system prompt for the LLM
            output_path: Optional output audio path (auto-generated if None)

        Returns:
            PipelineResult with all metadata
        """
        logger.info(f"Processing audio: {audio_path}")

        # Step 1: STT
        logger.info("Step 1: Speech-to-Text")
        stt_result = self.stt.transcribe_file(audio_path)
        logger.info(f"  Transcribed: {stt_result.text[:100]}...")

        if not stt_result.text.strip():
            raise ValueError("No speech detected in audio")

        # Step 2: LLM
        logger.info("Step 2: LLM inference")
        prompt = stt_result.text
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {prompt}"

        llm_response = self.llm.generate(prompt)
        logger.info(f"  Response: {llm_response[:100]}...")

        # Step 3: TTS
        logger.info("Step 3: Text-to-Speech")
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"response_{Path(audio_path).stem}.wav",
            )

        tts_result = self.tts.synthesize(llm_response, output_path=output_path)
        logger.info(f"  Audio saved: {tts_result.audio_path}")

        # Log to database
        if self._session:
            conv = Conversation(
                id=str(Path(output_path).stem),
                session_id=self._session.id,
                user_text=stt_result.text,
                llm_response=llm_response,
                stt_language=self.config.stt_language,
                tts_voice=self.config.tts_voice,
                audio_path=output_path,
                audio_duration=tts_result.duration_seconds,
            )
            self.db.add_conversation(conv)
            self._turn_count += 1
            self.db.update_session_stats(
                self._session.id, self._turn_count, tts_result.duration_seconds
            )

        return PipelineResult(
            stt_result=stt_result,
            llm_response=llm_response,
            tts_result=tts_result,
        )

    def conversation_loop(
        self,
        system_prompt: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """
        Interactive conversation loop using microphone input.

        Listens for speech, processes through LLM, speaks response.
        Press Ctrl+C to exit.

        Args:
            system_prompt: Optional system prompt for the LLM
            max_turns: Maximum conversation turns (None = unlimited)
        """
        logger.info("Starting conversation loop (Ctrl+C to exit)")
        if system_prompt:
            logger.info(f"System: {system_prompt}")

        self._turn_count = 0

        try:
            for stt_result in self.stt.transcribe_stream():
                if not stt_result.text.strip():
                    continue

                self._turn_count += 1
                if max_turns and self._turn_count > max_turns:
                    logger.info(f"Max turns ({max_turns}) reached")
                    break

                logger.info(f"Turn {self._turn_count}: {stt_result.text}")

                # LLM response
                prompt = stt_result.text
                if system_prompt:
                    prompt = f"{system_prompt}\n\nUser: {prompt}"

                llm_response = self.llm.generate(prompt)
                logger.info(f"LLM: {llm_response[:100]}...")

                # TTS response
                output_path = os.path.join(
                    self.config.output_dir, f"turn_{self._turn_count}.wav"
                )
                tts_result = self.tts.synthesize(llm_response, output_path=output_path)
                logger.info(f"Response saved: {output_path}")

                # Log to database
                if self._session:
                    conv = Conversation(
                        id=f"turn_{self._turn_count}",
                        session_id=self._session.id,
                        user_text=stt_result.text,
                        llm_response=llm_response,
                        stt_language=self.config.stt_language,
                        tts_voice=self.config.tts_voice,
                        audio_path=output_path,
                        audio_duration=tts_result.duration_seconds,
                    )
                    self.db.add_conversation(conv)
                    self.db.update_session_stats(
                        self._session.id, self._turn_count, tts_result.duration_seconds
                    )

        except KeyboardInterrupt:
            logger.info("Conversation ended by user")
        finally:
            if self._session:
                self.db.end_session(self._session.id)
                logger.info(f"Session ended: {self._session.id}")

    def process_text(
        self,
        text: str,
        system_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> "PipelineResult":
        """
        Process text directly (skip STT).

        Args:
            text: Input text
            system_prompt: Optional system prompt for the LLM
            output_path: Optional output audio path

        Returns:
            PipelineResult with all metadata
        """
        logger.info(f"Processing text: {text[:100]}...")

        # Step 1: LLM
        prompt = text
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {prompt}"

        llm_response = self.llm.generate(prompt)

        # Step 2: TTS
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir, "response.wav"
            )

        tts_result = self.tts.synthesize(llm_response, output_path=output_path)

        # Log to database
        if self._session:
            conv = Conversation(
                id=str(Path(output_path).stem),
                session_id=self._session.id,
                user_text=text,
                llm_response=llm_response,
                stt_language=self.config.stt_language,
                tts_voice=self.config.tts_voice,
                audio_path=output_path,
                audio_duration=tts_result.duration_seconds,
            )
            self.db.add_conversation(conv)
            self._turn_count += 1
            self.db.update_session_stats(
                self._session.id, self._turn_count, tts_result.duration_seconds
            )

        return PipelineResult(
            stt_result=STTResult(text=text, language=self.config.stt_language),
            llm_response=llm_response,
            tts_result=tts_result,
        )


@dataclass
class PipelineResult:
    """Result from a full pipeline run."""

    stt_result: STTResult
    llm_response: str
    tts_result: TTSResult

    @property
    def input_text(self) -> str:
        return self.stt_result.text

    @property
    def output_audio_path(self) -> str:
        return self.tts_result.audio_path

    @property
    def output_duration(self) -> float:
        return self.tts_result.duration_seconds
