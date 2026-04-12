"""
VAD [Wake-word 엔진]

mermaid 노드: VAD
mermaid 엣지:
  - User --> STT --> VAD                  (음성 입력에서 Wake-word 감지)
  - VAD --> Wait_UX --> TTS --> Speaker    (즉시 응답 트리거)
  - VAD --> State1 --> STT                (대화 대기 루프)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

WAKE_WORDS = ["오디스야", "오디스", "저기", "얘야", "오디", "어디스"]


@dataclass
class WakeWordResult:
    detected: bool
    keyword: str
    confidence: float


class VAD(ABC):
    """Wake-word 엔진 추상 인터페이스.

    실제 구현은 Silero VAD, Porcupine, 또는 커스텀 KWS 모델로 대체한다.
    """

    @abstractmethod
    async def start(self) -> None:
        """Wake-word 감지를 시작한다."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Wake-word 감지를 중단한다."""
        ...

    @abstractmethod
    async def wait_for_wakeword(self) -> WakeWordResult:
        """Wake-word가 감지될 때까지 대기한다.

        mermaid: User --> STT --> VAD
        """
        ...


class StubVAD(VAD):
    """테스트/개발용 VAD 스텁.

    외부에서 이벤트 큐를 통해 Wake-word를 시뮬레이션한다.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[WakeWordResult] = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("StubVAD 시작")

    async def stop(self) -> None:
        self._running = False
        logger.info("StubVAD 중단")

    async def wait_for_wakeword(self) -> WakeWordResult:
        return await self._queue.get()

    async def simulate_wakeword(self, keyword: str = "오디스야") -> None:
        await self._queue.put(WakeWordResult(detected=True, keyword=keyword, confidence=1.0))
