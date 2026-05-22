"""
Basic usage example for the voice pipeline.

Demonstrates how to use the pipeline with LM Studio as the LLM provider.

Prerequisites:
1. LM Studio running with a model loaded
2. Piper voice downloaded
3. Dependencies installed

Usage:
    python3 examples/basic_usage.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.voice_agent import VoicePipeline, PipelineConfig
from pipeline.lm_studio_provider import LMStudioProvider


def main() -> None:
    """Run the basic usage example."""

    print("Voice Pipeline - Basic Usage Example")
    print("=" * 50)

    # Step 1: Connect to LM Studio
    print("\n1. Connecting to LM Studio...")
    llm = LMStudioProvider(
        base_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
        model=os.getenv("LM_STUDIO_MODEL", "Qwen3-4B"),
    )

    if not llm.is_available():
        print("ERROR: Cannot connect to LM Studio.")
        print("Make sure LM Studio is running with a model loaded.")
        print(f"Expected endpoint: {llm.base_url}")
        sys.exit(1)

    print(f"  Connected to: {llm.base_url}")
    print(f"  Model: {llm.model}")

    # Step 2: Create pipeline
    print("\n2. Creating voice pipeline...")
    pipeline = VoicePipeline(
        llm_provider=llm,
        config=PipelineConfig(
            stt_language="en-US",
            tts_voice="en_US-lessac-medium",
            output_dir="./output",
        ),
    )

    print("  STT: Apple Speech (en-US)")
    print("  TTS: Piper (en_US-lessac-medium)")

    # Step 3: Process text (skip STT for this example)
    print("\n3. Processing text...")
    result = pipeline.process_text(
        text="What is the capital of France?",
        system_prompt="You are a helpful assistant. Keep responses concise.",
    )

    print(f"  Input: {result.input_text}")
    print(f"  Response: {result.llm_response}")
    print(f"  Audio saved: {result.output_audio_path}")
    print(f"  Duration: {result.output_duration:.2f}s")

    print("\nDone! Check the output directory for the generated audio.")


if __name__ == "__main__":
    main()
