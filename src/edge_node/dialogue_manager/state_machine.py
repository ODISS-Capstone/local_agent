"""
Dialogue_Manager [대화 상태 제어]

mermaid 노드: State1(대화 대기), State3(촬영 대기)
mermaid 엣지:
  - VAD --> State1 --> STT            (대화 대기 루프)
  - Cloud_Server --> State3           (주기적 처방 레포트 요청)
  - State3 --"촬영 실행"--> Buffer     (촬영 시작)
  - User --"약 가져왔어"--> STT        (사용자 촬영 트리거)

상태 전이:
  [*] --> State1 (시스템 시작)
  State1 --> State3 ("약 가져왔어" 또는 Cloud 요청)
  State3 --> CaptureMode ("촬영 실행")
  CaptureMode --> State1 (OCR 성공)
  CaptureMode --> State3 (OCR 실패, 재촬영)
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """mermaid에 정의된 대화 상태."""
    STATE1_CONVERSATION_WAIT = auto()  # State1((대화 대기))
    STATE3_CAPTURE_WAIT = auto()       # State3((촬영 대기))
    CAPTURING = auto()                 # Buffer -> Timer -> OCR_Engine 진행 중
    PROCESSING = auto()                # OCR 처리 중
    CONFIRMING = auto()                # 신뢰도 미달 -> 사용자 확인 대기


class DialogueEvent(Enum):
    """상태 전이를 유발하는 이벤트."""
    WAKE_WORD_DETECTED = auto()        # VAD --> State1
    CAPTURE_TRIGGER = auto()           # "약 가져왔어" 또는 Cloud_Server 요청
    CLOUD_REPORT_REQUEST = auto()      # Cloud_Server --> State3
    CAPTURE_START = auto()             # State3 --> Buffer
    BESTSHOT_CAPTURED = auto()         # Timer --> OCR_Engine
    OCR_SUCCESS = auto()               # OCR_Engine --성공--> Drug_Parser
    OCR_FAIL = auto()                  # OCR_Engine --실패--> Wait_UX
    OCR_NEEDS_CONFIRM = auto()         # 신뢰도 미달
    USER_CONFIRMED = auto()            # 사용자 확인 완료
    USER_DENIED = auto()               # 사용자 거부 -> 재촬영


TransitionCallback = Callable[[DialogueState, DialogueState, DialogueEvent], Coroutine[Any, Any, None]]

_TRANSITIONS: dict[tuple[DialogueState, DialogueEvent], DialogueState] = {
    # State1(대화 대기) 전이
    (DialogueState.STATE1_CONVERSATION_WAIT, DialogueEvent.WAKE_WORD_DETECTED):
        DialogueState.STATE1_CONVERSATION_WAIT,
    (DialogueState.STATE1_CONVERSATION_WAIT, DialogueEvent.CAPTURE_TRIGGER):
        DialogueState.STATE3_CAPTURE_WAIT,
    (DialogueState.STATE1_CONVERSATION_WAIT, DialogueEvent.CLOUD_REPORT_REQUEST):
        DialogueState.STATE3_CAPTURE_WAIT,

    # State3(촬영 대기) 전이
    (DialogueState.STATE3_CAPTURE_WAIT, DialogueEvent.CAPTURE_START):
        DialogueState.CAPTURING,

    # Capturing 전이
    (DialogueState.CAPTURING, DialogueEvent.BESTSHOT_CAPTURED):
        DialogueState.PROCESSING,

    # Processing 전이
    (DialogueState.PROCESSING, DialogueEvent.OCR_SUCCESS):
        DialogueState.STATE1_CONVERSATION_WAIT,
    (DialogueState.PROCESSING, DialogueEvent.OCR_FAIL):
        DialogueState.STATE3_CAPTURE_WAIT,
    (DialogueState.PROCESSING, DialogueEvent.OCR_NEEDS_CONFIRM):
        DialogueState.CONFIRMING,

    # Confirming 전이
    (DialogueState.CONFIRMING, DialogueEvent.USER_CONFIRMED):
        DialogueState.STATE1_CONVERSATION_WAIT,
    (DialogueState.CONFIRMING, DialogueEvent.USER_DENIED):
        DialogueState.STATE3_CAPTURE_WAIT,
}


class StateMachine:
    """대화 상태 머신.

    mermaid의 State1(대화 대기)과 State3(촬영 대기)를 중심으로
    Capture/Processing/Confirming 중간 상태를 관리한다.
    """

    def __init__(self) -> None:
        self._state = DialogueState.STATE1_CONVERSATION_WAIT
        self._callbacks: list[TransitionCallback] = []
        self._event_queue: asyncio.Queue[DialogueEvent] = asyncio.Queue()
        self._running = False

    @property
    def state(self) -> DialogueState:
        return self._state

    def on_transition(self, callback: TransitionCallback) -> None:
        """상태 전이 발생 시 호출될 콜백을 등록한다."""
        self._callbacks.append(callback)

    async def send_event(self, event: DialogueEvent) -> None:
        """외부에서 이벤트를 주입한다."""
        await self._event_queue.put(event)

    async def run(self) -> None:
        """이벤트 루프를 시작한다."""
        self._running = True
        logger.info("DialogueManager 시작: 초기 상태=%s", self._state.name)
        while self._running:
            try:
                event = await self._event_queue.get()
                await self._handle_event(event)
            except asyncio.CancelledError:
                break

    def stop(self) -> None:
        self._running = False

    async def _handle_event(self, event: DialogueEvent) -> None:
        key = (self._state, event)
        next_state = _TRANSITIONS.get(key)

        if next_state is None:
            logger.warning(
                "무시된 이벤트: state=%s, event=%s (유효한 전이 없음)",
                self._state.name, event.name,
            )
            return

        prev_state = self._state
        self._state = next_state
        logger.info(
            "상태 전이: %s --%s--> %s",
            prev_state.name, event.name, next_state.name,
        )

        for cb in self._callbacks:
            try:
                await cb(prev_state, next_state, event)
            except Exception:
                logger.exception("전이 콜백 실행 실패")
