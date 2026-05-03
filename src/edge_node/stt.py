"""
STT [스트리밍 STT 모델: Whisper]

mermaid 노드: STT
mermaid 엣지:
  - User --> STT --> VAD                  (음성 -> 텍스트 변환 -> Wake-word 전달)
  - STT --> Instruction_Log               (STT 로그를 클라우드로 전송)
  - User --"약 가져왔어"--> STT            (사용자 촬영 트리거)
  - VAD --> State1 --> STT                (대화 대기 루프에서 STT 활성화)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    is_final: bool = True


class STT(ABC):
    """스트리밍 STT 추상 인터페이스 (Whisper 기반).

    실제 구현은 Whisper.cpp, faster-whisper, 또는 OpenAI Whisper API로 대체한다.
    """

    @abstractmethod
    async def start_stream(self) -> None:
        """오디오 스트리밍 STT를 시작한다."""
        ...

    @abstractmethod
    async def stop_stream(self) -> None:
        """오디오 스트리밍 STT를 중단한다."""
        ...

    @abstractmethod
    async def get_transcription(self) -> TranscriptionResult:
        """변환된 텍스트 결과를 대기하여 반환한다.

        mermaid: User --> STT
        """
        ...

    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        """단일 오디오 청크를 텍스트로 변환한다."""
        ...


class StubSTT(STT):
    """테스트/개발용 STT 스텁."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        self._running = False

    async def start_stream(self) -> None:
        self._running = True
        logger.info("StubSTT 스트리밍 시작")

    async def stop_stream(self) -> None:
        self._running = False
        logger.info("StubSTT 스트리밍 중단")

    async def get_transcription(self) -> TranscriptionResult:
        return await self._queue.get()

    async def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        return TranscriptionResult(text="", confidence=0.0, is_final=True)

    async def simulate_input(self, text: str, confidence: float = 0.95) -> None:
        await self._queue.put(TranscriptionResult(text=text, confidence=confidence))


class PipelineSTT(STT):
    """`AudioPipeline`이 발행한 transcription을 소비하는 STT 어댑터.

    실제 STT는 AudioPipeline 내부의 Whisper가 수행하고,
    이 클래스는 큐에서 결과를 꺼내 기존 인터페이스로 노출한다.
    """

    def __init__(self, pipeline: "object") -> None:  # AudioPipeline (forward ref)
        self._pipeline = pipeline
        self._running = False

    async def start_stream(self) -> None:
        self._running = True
        logger.info("PipelineSTT 스트리밍 시작")

    async def stop_stream(self) -> None:
        self._running = False
        logger.info("PipelineSTT 스트리밍 중단")

    async def get_transcription(self) -> TranscriptionResult:
        return await self._pipeline.transcription_queue.get()

    async def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        return TranscriptionResult(text="", confidence=0.0, is_final=True)
