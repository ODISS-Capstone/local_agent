"""
로컬 에이전트 메인 엔트리포인트.

OCR_Agent.mermaid에 정의된 모든 노드를 조립하고,
엣지(데이터 흐름)를 asyncio 이벤트 루프 위에서 연결한다.

=== mermaid 엣지 매핑 ===
1-2. User --> STT --> VAD --> Wait_UX --> TTS --> Speaker   (호출 및 즉시 응답)
3.   STT --> Instruction_Log                               (STT 로그를 클라우드로)
4.   Cloud_Server --> State3                               (주기적 처방 레포트 요청)
5.   User --"약 가져왔어"--> STT                             (사용자 촬영 트리거)
6.   State3 --> Buffer --> Timer --> TTS --> Speaker        (촬영 실행 흐름)
7.   Timer --> OCR_Engine                                  (타이머 완료 후 OCR)
8.   VAD --> State1 --> STT                                (대화 대기 루프)
9.   Wait_UX --> TTS --> Speaker                           (고정 멘트)
10.  OCR_Engine --성공--> Drug_Parser --> DB                 (클라우드 전송)
11.  OCR_Engine --실패--> Wait_UX --> TTS --> Speaker        (재요청)
12.  Cam --> Capture_Mode                                  (RTSP 스트림)
13.  Cloud_Server --> TTS --> Speaker                       (클라우드 실시간 응답)
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from src.config_loader import load_config
from src.home_environment.cam import Cam, CamConfig
from src.home_environment.speaker import LocalSpeaker
from src.edge_node.vad import StubVAD, PipelineVAD
from src.edge_node.stt import StubSTT, PipelineSTT, TranscriptionResult
from src.edge_node.tts import StubTTS, GTTSEngine, TTSPriority
from src.edge_node.audio_pipeline import AudioPipeline, AudioPipelineConfig
from src.edge_node.wait_ux import WaitUX
from src.edge_node.capture_mode.buffer import Buffer, QualityConfig, QualityFailReason
from src.edge_node.capture_mode.timer import Timer
from src.edge_node.capture_mode.ocr_engine import (
    OCREngine,
    OCREngineConfig,
    OCRResult,
    ActionRequired,
)
from src.edge_node.dialogue_manager.state_machine import (
    StateMachine,
    DialogueState,
    DialogueEvent,
)
from src.cloud_server.drug_parser import StubDrugParserClient
from src.cloud_server.instruction_log import StubInstructionLogClient, InstructionEntry
from src.runtime import turboquant_runtime

logger = logging.getLogger(__name__)

CAPTURE_TRIGGER_KEYWORDS = ["약 가져왔어", "약 찍어", "약 보여줄게", "사진 찍어"]


class LocalAgent:
    """mermaid 아키텍처에 따른 로컬 에이전트 오케스트레이터.

    모든 mermaid 노드를 생성하고, 엣지를 asyncio 태스크/큐로 연결한다.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_config()

        turboquant_runtime.install(cfg.get("turboquant"))

        # --- Home_Environment 노드 ---
        rtsp_cfg = cfg.get("rtsp", {})
        self.cam = Cam(CamConfig(
            url=rtsp_cfg.get("url", "rtsp://192.168.0.100:554/stream"),
            reconnect_backoff_sec=rtsp_cfg.get("reconnect_backoff_sec", 2.0),
            max_reconnect_attempts=rtsp_cfg.get("max_reconnect_attempts", 10),
        ))
        audio_cfg = cfg.get("audio", {})
        self.speaker = LocalSpeaker(
            device=audio_cfg.get("output_device", "default"),
            sample_rate=audio_cfg.get("sample_rate", 22050),
            channels=audio_cfg.get("channels", 1),
        )

        # --- Edge_Node 노드 ---
        stt_cfg = cfg.get("stt", {})
        if stt_cfg.get("enabled", False):
            pipeline_cfg = AudioPipelineConfig(
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
            )
            self.audio_pipeline: AudioPipeline | None = AudioPipeline(pipeline_cfg)
            self.vad = PipelineVAD(self.audio_pipeline)
            self.stt = PipelineSTT(self.audio_pipeline)
            logger.info("STT 모드: 실제 마이크 + faster-whisper")
        else:
            self.audio_pipeline = None
            self.vad = StubVAD()
            self.stt = StubSTT()
            logger.info("STT 모드: Stub (시뮬레이션)")

        tts_cfg = cfg.get("tts", {})
        if tts_cfg.get("enabled", False) and tts_cfg.get("engine", "gtts") == "gtts":
            self.tts = GTTSEngine(
                speaker=self.speaker,
                lang=tts_cfg.get("lang", "ko"),
                tld=tts_cfg.get("tld", "co.kr"),
                slow=tts_cfg.get("slow", False),
                sample_rate=tts_cfg.get("sample_rate", 22050),
                channels=tts_cfg.get("channels", 1),
            )
            logger.info(
                "TTS 모드: gTTS (lang=%s tld=%s)",
                tts_cfg.get("lang", "ko"),
                tts_cfg.get("tld", "co.kr"),
            )
        else:
            self.tts = StubTTS()
            logger.info("TTS 모드: Stub (로그 출력만)")

        quality_cfg = cfg.get("quality", {})
        self.buffer = Buffer(
            cam=self.cam,
            quality_config=QualityConfig(
                blur_threshold=quality_cfg.get("blur_threshold", 100.0),
                brightness_min=quality_cfg.get("brightness_min", 50),
                brightness_max=quality_cfg.get("brightness_max", 230),
                glare_max_ratio=quality_cfg.get("glare_max_ratio", 0.15),
            ),
            buffer_size=rtsp_cfg.get("buffer_size", 30),
        )

        capture_cfg = cfg.get("capture", {})
        self.timer = Timer(
            buffer=self.buffer,
            timer_seconds=capture_cfg.get("timer_seconds", 3),
        )

        ocr_cfg = cfg.get("ocr", {})
        self.ocr_engine = OCREngine(OCREngineConfig(
            model_path=ocr_cfg.get("model_path", "models/glm-ocr"),
            provider=ocr_cfg.get("provider", "stub"),
            hf_device=ocr_cfg.get("hf_device", "auto"),
            hf_torch_dtype=ocr_cfg.get("hf_torch_dtype", "auto"),
            hf_prompt=ocr_cfg.get("hf_prompt", "Text Recognition:"),
            hf_max_new_tokens=ocr_cfg.get("hf_max_new_tokens", 4096),
            hf_max_image_side=ocr_cfg.get("hf_max_image_side", 1280),
            hf_extract_document=ocr_cfg.get("hf_extract_document", True),
            hf_repetition_penalty=ocr_cfg.get("hf_repetition_penalty", 1.15),
            hf_no_repeat_ngram_size=ocr_cfg.get("hf_no_repeat_ngram_size", 8),
            gemini_model=ocr_cfg.get("gemini_model", "gemini-3-flash-preview"),
            gemini_fallback_models=ocr_cfg.get(
                "gemini_fallback_models",
                ["gemini-2.5-flash", "gemini-flash-latest"],
            ),
            gemini_api_key=ocr_cfg.get("gemini_api_key", ""),
            gemini_api_key_env=ocr_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"),
            gemini_prompt=ocr_cfg.get("gemini_prompt", ""),
            glmocr_mode=ocr_cfg.get("glmocr_mode", "maas"),
            glmocr_api_key_env=ocr_cfg.get("glmocr_api_key_env", "ZHIPU_API_KEY"),
            glmocr_timeout_sec=ocr_cfg.get("glmocr_timeout_sec", 600),
            glmocr_save_dir=ocr_cfg.get("glmocr_save_dir", "runtime/ocr"),
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.85),
            fuzzy_match_max_distance=ocr_cfg.get("fuzzy_match_max_distance", 2),
            medical_dict_path=ocr_cfg.get("medical_dict_path", "config/medical_terms.json"),
        ))

        wait_ux_cfg = cfg.get("wait_ux", {})
        self.wait_ux = WaitUX(
            escalation_intervals=wait_ux_cfg.get("waiting_escalation_intervals_sec", [3, 7, 15]),
        )

        self.state_machine = StateMachine()

        # --- Cloud_Server 노드 ---
        self.drug_parser = StubDrugParserClient()
        self.instruction_log = StubInstructionLogClient()

        # --- 내부 큐 (엣지 연결용) ---
        self._tts_queue: asyncio.Queue[str] = asyncio.Queue()
        self._quality_fail_queue: asyncio.Queue[QualityFailReason] = asyncio.Queue()

        self._tasks: list[asyncio.Task[Any]] = []

    async def start(self) -> None:
        """모든 노드를 초기화하고 mermaid 엣지를 연결하여 에이전트를 시작한다."""
        logger.info("=== 로컬 에이전트 시작 ===")

        await self.ocr_engine.load()
        if self.audio_pipeline is not None:
            await self.audio_pipeline.start()
        await self.vad.start()
        await self.stt.start_stream()

        self._wire_edges()

        self._tasks = [
            asyncio.create_task(self.state_machine.run(), name="state_machine"),
            asyncio.create_task(self._wakeword_loop(), name="wakeword_loop"),
            asyncio.create_task(self._tts_consumer_loop(), name="tts_consumer"),
            asyncio.create_task(self._quality_feedback_loop(), name="quality_feedback"),
        ]

        logger.info("에이전트 실행 중: %d개 태스크", len(self._tasks))

    def _wire_edges(self) -> None:
        """mermaid 엣지에 따라 노드 간 큐/콜백을 연결한다."""

        # Wait_UX --> TTS (엣지 9, 11)
        self.wait_ux.set_tts_queue(self._tts_queue)

        # Timer --> TTS (엣지 6: 카운트다운)
        self.timer.set_tts_queue(self._tts_queue)

        # Buffer 품질 실패 --> Wait_UX (엣지 12 -> 품질 피드백)
        self.buffer.set_quality_fail_queue(self._quality_fail_queue)

        # StateMachine 전이 콜백 등록
        self.state_machine.on_transition(self._on_state_transition)

    async def _wakeword_loop(self) -> None:
        """엣지 1-2, 8: User --> STT --> VAD --> Wait_UX --> TTS --> Speaker

        VAD --> State1 --> STT (대화 대기 루프)
        """
        while True:
            try:
                result = await self.vad.wait_for_wakeword()
                if not result.detected:
                    continue

                logger.info("Wake-word 감지: '%s'", result.keyword)

                # VAD --> Wait_UX --> TTS --> Speaker (즉시 응답)
                await self.wait_ux.immediate_response()

                # VAD --> State1 (대화 대기 상태 진입)
                await self.state_machine.send_event(DialogueEvent.WAKE_WORD_DETECTED)

                # State1 --> STT (대화 대기 루프: 사용자 발화 수집)
                transcription = await self.stt.get_transcription()
                await self._handle_stt_result(transcription)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Wake-word 루프 오류")

    async def _handle_stt_result(self, result: TranscriptionResult) -> None:
        """STT 결과를 처리한다.

        엣지 3: STT --> Instruction_Log
        엣지 5: User --"약 가져왔어"--> STT --> 촬영 트리거
        """
        text = result.text.strip()
        if not text:
            return

        # 엣지 3: STT --> Instruction_Log
        entry = InstructionEntry(text=text, timestamp=result.timestamp)
        asyncio.create_task(self.instruction_log.send_log(entry))

        # 엣지 5: 촬영 트리거 키워드 검사
        if any(kw in text for kw in CAPTURE_TRIGGER_KEYWORDS):
            logger.info("촬영 트리거 감지: '%s'", text)
            await self.state_machine.send_event(DialogueEvent.CAPTURE_TRIGGER)

    async def _on_state_transition(
        self,
        prev: DialogueState,
        next_state: DialogueState,
        event: DialogueEvent,
    ) -> None:
        """StateMachine 상태 전이에 따른 액션을 실행한다."""

        # 엣지 6: State3 --"촬영 실행"--> Buffer --> Timer
        if next_state == DialogueState.STATE3_CAPTURE_WAIT:
            await self.state_machine.send_event(DialogueEvent.CAPTURE_START)

        elif next_state == DialogueState.CAPTURING:
            asyncio.create_task(self._run_capture_flow())

    async def _run_capture_flow(self) -> None:
        """촬영 실행 흐름 (엣지 6, 7, 10, 11).

        State3 --> Buffer --> Timer --> OCR_Engine
                              |
                              +--> TTS --> Speaker (카운트다운)
        """
        try:
            # Wait_UX 대기 멘트 시작
            await self.wait_ux.start_waiting()

            # 엣지 12: Cam --> Capture_Mode (Buffer 활성화)
            if not self.cam.is_opened:
                try:
                    await self.cam.open()
                except ConnectionError:
                    logger.error("Cam 연결 실패: 재촬영 상태로 전이")
                    await self.wait_ux.stop_waiting()
                    await self.wait_ux.retry_request()
                    await self.state_machine.send_event(DialogueEvent.OCR_FAIL)
                    return

            self.buffer.activate()

            # Buffer에 프레임 수집할 시간 확보
            capture_task = asyncio.create_task(self.buffer.capture_loop())
            await asyncio.sleep(0.5)

            # 엣지 6-7: Buffer --> Timer --> TTS + Timer --> OCR_Engine
            capture_result = await self.timer.run_countdown()

            # 버퍼 수집 중단
            self.buffer.deactivate()
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass

            await self.wait_ux.stop_waiting()

            if capture_result is None:
                # 실패: 엣지 11
                await self.wait_ux.retry_request()
                await self.state_machine.send_event(DialogueEvent.OCR_FAIL)
                return

            # 엣지 7: Timer --> OCR_Engine
            await self.state_machine.send_event(DialogueEvent.BESTSHOT_CAPTURED)
            ocr_result = await self.ocr_engine.process(capture_result.frame)

            await self._handle_ocr_result(ocr_result)

        except Exception:
            logger.exception("촬영 흐름 오류")
            await self.wait_ux.stop_waiting()
            await self.state_machine.send_event(DialogueEvent.OCR_FAIL)

    async def _handle_ocr_result(self, result: OCRResult) -> None:
        """OCR 결과에 따라 성공/실패/확인 요청을 분기한다.

        엣지 10: OCR_Engine --성공--> Drug_Parser --> DB
        엣지 11: OCR_Engine --실패--> Wait_UX --> TTS --> Speaker
        """
        if result.action_required == ActionRequired.PROCEED.value:
            # 엣지 10: Drug_Parser 전송
            logger.info("OCR 성공: Drug_Parser로 전송")
            payload = result.to_dict()
            asyncio.create_task(self.drug_parser.send_ocr_result(payload))
            await self.state_machine.send_event(DialogueEvent.OCR_SUCCESS)

        elif result.action_required == ActionRequired.NEEDS_CONFIRMATION.value:
            # 신뢰도 미달: 확인 요청
            logger.info("OCR 신뢰도 미달: 사용자 확인 요청")
            await self.wait_ux.confidence_confirm(result.text)
            await self.state_machine.send_event(DialogueEvent.OCR_NEEDS_CONFIRM)

        else:
            # 엣지 11: 실패 재요청
            logger.info("OCR 실패: 재촬영 요청")
            await self.wait_ux.retry_request()
            await self.state_machine.send_event(DialogueEvent.OCR_FAIL)

    async def _tts_consumer_loop(self) -> None:
        """TTS 큐에서 메시지를 소비하여 TTS --> Speaker로 전달한다.

        모든 Wait_UX / Timer --> TTS --> Speaker 엣지가 이 루프를 통과한다.
        """
        while True:
            try:
                text = await self._tts_queue.get()
                await self.tts.speak(text, TTSPriority.NORMAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("TTS consumer 오류")

    async def _quality_feedback_loop(self) -> None:
        """Buffer 품질 실패 --> Wait_UX --> TTS --> Speaker.

        실시간 프레임 품질 미달 시 구두 가이드를 트리거한다.
        """
        while True:
            try:
                reason = await self._quality_fail_queue.get()
                await self.wait_ux.quality_guide(reason)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("품질 피드백 루프 오류")

    async def stop(self) -> None:
        """모든 태스크를 정리하고 에이전트를 종료한다."""
        logger.info("=== 로컬 에이전트 종료 ===")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self.state_machine.stop()
        self.buffer.deactivate()
        await self.vad.stop()
        await self.stt.stop_stream()
        if self.audio_pipeline is not None:
            await self.audio_pipeline.stop()
        await self.tts.stop()
        await self.cam.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = LocalAgent()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))

    await agent.start()

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
