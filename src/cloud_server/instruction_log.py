"""
Instruction_Log [STT 환자 지시 로그 수집]

mermaid 노드: Instruction_Log
mermaid 서브그래프: Prescription_Recognition (처방전 정밀 분석 메소드)
mermaid 엣지:
  - STT --> Instruction_Log
  - Instruction_Log --> DB

STT 텍스트를 타임스탬프와 함께 클라우드에 비동기 전송한다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class InstructionEntry:
    text: str
    timestamp: float = field(default_factory=time.time)
    source: str = "stt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class InstructionLogConfig:
    endpoint: str = "https://api.odiss.example.com"
    path: str = "/v1/perception/stt-log"
    timeout_sec: float = 10.0
    retry_count: int = 3
    retry_backoff_sec: float = 1.0


class InstructionLogClient(ABC):
    """Instruction_Log 클라우드 통신 추상 인터페이스."""

    @abstractmethod
    async def send_log(self, entry: InstructionEntry) -> bool:
        """STT --> Instruction_Log: STT 텍스트를 클라우드에 전송한다."""
        ...


class HttpInstructionLogClient(InstructionLogClient):
    """aiohttp 기반 Instruction_Log 클라이언트."""

    def __init__(self, config: InstructionLogConfig | None = None) -> None:
        self._config = config or InstructionLogConfig()
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def send_log(self, entry: InstructionEntry) -> bool:
        if self._session is None:
            await self.open()
        assert self._session is not None

        url = f"{self._config.endpoint}{self._config.path}"
        backoff = self._config.retry_backoff_sec

        for attempt in range(1, self._config.retry_count + 1):
            try:
                async with self._session.post(url, json=entry.to_dict()) as resp:
                    if resp.status == 200:
                        logger.info("Instruction_Log 전송 성공")
                        return True
                    logger.warning(
                        "Instruction_Log 응답 오류 (attempt %d/%d): HTTP %d",
                        attempt, self._config.retry_count, resp.status,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Instruction_Log 전송 실패 (attempt %d/%d): %s",
                    attempt, self._config.retry_count, exc,
                )

            if attempt < self._config.retry_count:
                await asyncio.sleep(backoff)
                backoff *= 2

        logger.error("Instruction_Log 전송 최종 실패")
        return False


class StubInstructionLogClient(InstructionLogClient):
    """테스트/개발용 Instruction_Log 스텁."""

    def __init__(self) -> None:
        self._entries: list[InstructionEntry] = []

    async def send_log(self, entry: InstructionEntry) -> bool:
        self._entries.append(entry)
        logger.info("StubInstructionLog: 로그 수신 '%s' (총 %d건)", entry.text, len(self._entries))
        return True

    @property
    def entries(self) -> list[InstructionEntry]:
        return self._entries
