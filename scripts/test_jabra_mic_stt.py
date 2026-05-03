"""
Jabra 마이크 + faster-whisper 단독 검증.

- agent_config.yaml의 stt 설정을 그대로 읽어 AudioPipeline을 시작
- 발화가 잡히면 wake-word / transcription 큐에 출력된 결과를 실시간으로 콘솔에 찍음
- Ctrl+C로 종료

사용법:
    cd ~/local_agent
    python3 -m scripts.test_jabra_mic_stt

말하면 됩니다. wake-word("오디스" 등)가 포함되면 [WAKE]로 표시되고,
아니면 [STT]로 표시됩니다. 아무것도 안 나오면 마이크/디바이스 경로 점검.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edge_node.audio_pipeline import AudioPipeline, AudioPipelineConfig


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config" / "agent_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    stt_cfg = cfg.get("stt", {})
    audio_cfg = cfg.get("audio", {})

    pipeline = AudioPipeline(AudioPipelineConfig(
        input_device=stt_cfg.get(
            "input_device", audio_cfg.get("input_device", "default")
        ),
        sample_rate=stt_cfg.get("sample_rate", 16000),
        frame_ms=stt_cfg.get("frame_ms", 30),
        vad_aggressiveness=stt_cfg.get("vad_aggressiveness", 2),
        silence_tail_ms=stt_cfg.get("silence_tail_ms", 700),
        min_utterance_ms=stt_cfg.get("min_utterance_ms", 300),
        max_utterance_ms=stt_cfg.get("max_utterance_ms", 12000),
        whisper_model=stt_cfg.get("whisper_model", "small"),
        whisper_compute_type=stt_cfg.get("whisper_compute_type", "int8"),
        whisper_device=stt_cfg.get("whisper_device", "cpu"),
        language=stt_cfg.get("language", "ko"),
        initial_prompt=stt_cfg.get(
            "initial_prompt",
            "오디스, 약, 처방전, 복용, 어르신, 사진, 찍어, 가져왔어",
        ),
    ))

    print("[mic-stt] Jabra 마이크 + faster-whisper 시작합니다 (Ctrl+C 로 종료)")
    await pipeline.start()

    async def wake_consumer() -> None:
        while True:
            r = await pipeline.wake_queue.get()
            print(f"[WAKE] keyword='{r.keyword}' confidence={r.confidence}")

    async def text_consumer() -> None:
        while True:
            r = await pipeline.transcription_queue.get()
            print(f"[STT]  text='{r.text}'")

    try:
        await asyncio.gather(wake_consumer(), text_consumer())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
