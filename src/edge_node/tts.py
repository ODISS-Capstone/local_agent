"""
TTS [대화형 TTS 모델: Qwen3-TTS]

mermaid 노드: TTS
mermaid 엣지:
  - Wait_UX --> TTS --> Speaker                                  (고정 멘트 출력)
  - Timer --> TTS --> Speaker                                    (카운트다운 음성)
  - Cloud_Server --> TTS --> Speaker                             (실시간 응답)
  - OCR_Engine --실패--> Wait_UX --> TTS -- "약봉투 다시..." --> Speaker
  - TTS -- "즉시 응답 (네 어르신 말씀해주세요!)" --> Speaker
  - TTS -- "에이전트: 어르신, XX약은 녹용이랑 드시면 안되요~" --> Speaker
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class TTSPriority(IntEnum):
    """TTS 출력 우선순위 (낮을수록 높은 우선순위)."""
    URGENT = 0       # 실패/재요청
    HIGH = 1         # 즉시 응답, 카운트다운
    NORMAL = 2       # 에이전트 응답
    LOW = 3          # 대기 멘트


@dataclass
class TTSRequest:
    text: str
    priority: TTSPriority = TTSPriority.NORMAL


class TTS(ABC):
    """대화형 TTS 추상 인터페이스 (Qwen3-TTS 기반).

    실제 구현은 Qwen3-TTS 로컬 모델 또는 API로 대체한다.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """텍스트를 음성 데이터(PCM/WAV bytes)로 변환한다."""
        ...

    @abstractmethod
    async def speak(self, text: str, priority: TTSPriority = TTSPriority.NORMAL) -> None:
        """텍스트를 음성으로 변환하고 Speaker로 출력한다.

        mermaid: TTS --> Speaker
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """현재 TTS 출력을 중단한다."""
        ...


class StubTTS(TTS):
    """테스트/개발용 TTS 스텁.

    실제 음성 합성 없이 텍스트 로깅만 수행한다.
    """

    def __init__(self) -> None:
        self._output_log: list[TTSRequest] = []

    async def synthesize(self, text: str) -> bytes:
        logger.info("StubTTS synthesize: %s", text)
        return b""

    async def speak(self, text: str, priority: TTSPriority = TTSPriority.NORMAL) -> None:
        request = TTSRequest(text=text, priority=priority)
        self._output_log.append(request)
        logger.info("StubTTS speak [%s]: %s", priority.name, text)

    async def stop(self) -> None:
        logger.info("StubTTS 중단")

    @property
    def output_log(self) -> list[TTSRequest]:
        return self._output_log
