"""
Speaker [홈 스피커 / 캠 내장 스피커]

mermaid 노드: Speaker
mermaid 엣지:
  - TTS --> Speaker
  - Speaker <--> User
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Speaker(ABC):
    """홈 스피커 오디오 출력 추상 인터페이스."""

    @abstractmethod
    async def play(self, audio_data: bytes) -> None:
        """TTS로부터 받은 오디오 데이터를 스피커로 출력한다."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """현재 재생 중인 오디오를 중단한다."""
        ...


class LocalSpeaker(Speaker):
    """로컬 오디오 디바이스를 통한 스피커 구현.

    Jetson 환경에서 ALSA/PulseAudio 디바이스로 출력한다.
    실제 오디오 출력은 aplay 또는 pyaudio를 사용할 수 있으며,
    여기서는 subprocess 기반으로 구현한다.

    Jabra SPEAK 510 USB 같은 USB 오디오를 쓰려면 device를
    `plughw:CARD=USB,DEV=0` 처럼 ALSA 식별자로 지정한다.
    """

    def __init__(
        self,
        device: str = "default",
        sample_rate: int = 22050,
        channels: int = 1,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._process: asyncio.subprocess.Process | None = None

    async def play(self, audio_data: bytes) -> None:
        if not audio_data:
            return
        self._process = await asyncio.create_subprocess_exec(
            "aplay",
            "-D", self._device,
            "-f", "S16_LE",
            "-r", str(self._sample_rate),
            "-c", str(self._channels),
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert self._process.stdin is not None
        self._process.stdin.write(audio_data)
        self._process.stdin.close()
        await self._process.wait()
        logger.debug(
            "오디오 재생 완료 (%d bytes, device=%s)",
            len(audio_data),
            self._device,
        )

    async def stop(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
            logger.debug("오디오 재생 중단")
