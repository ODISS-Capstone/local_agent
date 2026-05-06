# 지능형 복용지도 시스템 - 로컬 에이전트 (OCR 에이전트)

## 개요

어르신 대상 지능형 복용지도 시스템의 **Perception(인지) 단계**를 담당하는 로컬 에이전트이다.
`OCR_Agent.mermaid`에 정의된 3개 서브그래프(Home\_Environment, Edge\_Node, Cloud\_Server)와
그 안의 노드/엣지를 1:1로 Python 모듈에 매핑하여 구현한다.

---

## mermaid 구조 분석

`OCR_Agent.mermaid`에서 추출한 정확한 구조:

```
Home_Environment           Edge_Node                           Cloud_Server
├── User                   ├── VAD                             ├── Prescription_Recognition
├── Cam                    ├── STT                             │   ├── Drug_Parser
└── Speaker                ├── TTS                             │   └── Instruction_Log
                           ├── Capture_Mode                    └── DB
                           │   ├── Buffer
                           │   ├── Timer
                           │   └── OCR_Engine
                           ├── Dialogue_Manager
                           │   ├── State1 (대화 대기)
                           │   └── State3 (촬영 대기)
                           └── Wait_UX
```

---

## mermaid 엣지 (데이터 흐름) 정리

| 단계 | 흐름 | mermaid 원문 |
|------|------|-------------|
| 1-2 | User -> STT -> VAD -> Wait\_UX -> TTS -> Speaker | 호출 및 즉시 응답 |
| 3 | STT -> Instruction\_Log | STT 로그를 클라우드로 |
| 4 | Cloud\_Server -> State3 | 주기적 처방 레포트 요청 |
| 5 | User -> "약 가져왔어" -> STT | 사용자 촬영 트리거 |
| 6 | State3 -> Buffer -> Timer -> TTS -> Speaker | 촬영 실행 흐름 |
| 7 | Timer -> OCR\_Engine | 타이머 완료 후 OCR 실행 |
| 8 | VAD -> State1 -> STT | 대화 대기 상태 루프 |
| 9 | Wait\_UX -> TTS -> Speaker | 고정 멘트 출력 |
| 10 | OCR\_Engine --성공--> Drug\_Parser -> DB | OCR 성공 시 클라우드 전송 |
| 11 | OCR\_Engine --실패--> Wait\_UX -> TTS -> Speaker | OCR 실패 시 재요청 |
| 12 | Cam -> Capture\_Mode | RTSP 스트림 입력 |
| 13 | Cloud\_Server -> TTS -> Speaker | 클라우드 실시간 응답 |

---

## 프로젝트 구조 (mermaid 서브그래프 1:1 매핑)

```
local_agent/
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── agent_config.yaml
│   └── medical_terms.json
├── src/
│   ├── __init__.py
│   ├── main.py                            # 전체 에이전트 조립 및 asyncio 루프
│   │
│   ├── home_environment/                  # --- [Home_Environment] 서브그래프 ---
│   │   ├── __init__.py
│   │   ├── cam.py                         #   Cam [홈캠 RTSP 스트리밍]
│   │   └── speaker.py                     #   Speaker [홈 스피커 / 캠 내장 스피커]
│   │
│   ├── edge_node/                         # --- [Edge_Node] 서브그래프 ---
│   │   ├── __init__.py
│   │   ├── vad.py                         #   VAD [Wake-word 엔진] (인터페이스)
│   │   ├── stt.py                         #   STT [스트리밍 STT: Whisper] (인터페이스)
│   │   ├── tts.py                         #   TTS [대화형 TTS: Qwen3-TTS] (인터페이스)
│   │   ├── wait_ux.py                     #   Wait_UX [고정 멘트 엔진]
│   │   │
│   │   ├── capture_mode/                  #   [Capture_Mode] 서브그래프
│   │   │   ├── __init__.py
│   │   │   ├── buffer.py                  #     Buffer [RTSP 프레임 버퍼링]
│   │   │   ├── timer.py                   #     Timer [촬영 타이머: 하나 둘 셋 찰칵]
│   │   │   └── ocr_engine.py              #     OCR_Engine [GLM-OCR: 텍스트 추출]
│   │   │
│   │   └── dialogue_manager/              #   [Dialogue_Manager] 서브그래프
│   │       ├── __init__.py
│   │       └── state_machine.py           #     State1(대화 대기) / State3(촬영 대기)
│   │
│   └── cloud_server/                      # --- [Cloud_Server] 서브그래프 (인터페이스) ---
│       ├── __init__.py
│       ├── drug_parser.py                 #   Drug_Parser [약물 정보 파싱]
│       ├── instruction_log.py             #   Instruction_Log [STT 환자 지시 로그]
│       └── db.py                          #   DB [(환자 통합 MD파일 DB)]
│
├── tests/
└── OCR_Agent.mermaid
```

---

## 노드별 상세 설계

### Home\_Environment (하드웨어 추상화)

**`cam.py`** - `Cam` 노드

- mermaid 엣지: `Cam --"RTSP 통신 지원"--> Capture_Mode`, `Speaker <-->User <-->Cam`
- Jetson GStreamer 파이프라인으로 RTSP 수신: `rtspsrc -> nvv4l2decoder -> nvvidconv -> appsink`
- `async def read_frame() -> numpy.ndarray` : 단일 프레임 반환
- `Capture_Mode.Buffer`에 프레임을 공급하는 역할

**`speaker.py`** - `Speaker` 노드

- mermaid 엣지: `TTS --> Speaker`, `Speaker <-->User`
- 오디오 출력 디바이스 추상화
- `async def play(audio_data: bytes) -> None`

### Edge\_Node - Capture\_Mode (핵심 구현 대상)

**`buffer.py`** - `Buffer` 노드

- mermaid 엣지: `Cam --> Capture_Mode`, `State3 --"촬영 실행"--> Buffer`, `Buffer --> Timer`
- `Cam`으로부터 RTSP 프레임을 수신하여 링 버퍼에 저장
- 명세 반영: 실시간 프리뷰 프레임 품질 평가 (블러/조도/글레어 감지)
  - 블러: Laplacian variance, 조도: HSV V-channel 평균, 글레어: 고휘도 비율
- 품질 미달 시 `Wait_UX`에 피드백 신호 전달 (명세 1항/5항: 능동적 피드백)
- BestShot 로직: 버퍼 내 최고 품질 프레임을 `Timer`에 전달

**`timer.py`** - `Timer` 노드

- mermaid 엣지: `Buffer --> Timer`, `Timer --> TTS`, `Timer --> OCR_Engine`
- 촬영 카운트다운 ("하나, 둘, 셋, 찰칵!") - 각 카운트를 `TTS`에 전달
- 카운트 완료 시 `Buffer`에서 BestShot 프레임 확정
- 확정된 프레임을 `OCR_Engine`에 전달

**`ocr_engine.py`** - `OCR_Engine` 노드

- mermaid 엣지: `Timer --> OCR_Engine`, `OCR_Engine --성공--> Drug_Parser`, `OCR_Engine --실패--> Wait_UX`
- GLM-OCR 모델 래핑 (Jetson TensorRT FP16 최적화)
- 명세 반영 - 내부적으로 다음 기능 수행:
  - **ROI 분류** (명세 2항/6항): 처방전/알약 자동 판별
  - **처방전 모드**: 표 구조 Key-Value 추출 (약품명, 투여횟수, 투약일수)
  - **알약 모드**: 텍스트 + 시각적 특징 추출 (형태, 색상, 분할선, 각인)
  - **신뢰도 평가** (명세 3항/7항): confidence < 0.85이면 실패 판정
  - **의학용어 사전 매칭**: fuzzy matching으로 약품명 검증
- 성공 시: 결과를 `Drug_Parser`로 전송 (mermaid 엣지 10)
- 실패 시: `Wait_UX`에 재요청 신호 (mermaid 엣지 11)

```python
@dataclass
class OCRResult:
    perception_timestamp: str       # ISO 8601
    input_type: Literal["PRESCRIPTION", "PILL_IMAGE"]
    text: str
    text_confidence_score: float
    visual_features: dict | None    # 알약 모드: shape, color, line
    structured_data: dict | None    # 처방전 모드: drugs list
    action_required: Literal[
        "PROCEED_TO_IDENTIFICATION",
        "NEEDS_CONFIRMATION",
        "RETRY_CAPTURE"
    ]
```

### Edge\_Node - Dialogue\_Manager

**`state_machine.py`** - `State1`(대화 대기) / `State3`(촬영 대기) 노드

- mermaid 엣지:
  - `VAD --> State1 --> STT` (대화 대기 루프)
  - `Cloud_Server --"주기적 처방 레포트 요청"--> State3` (클라우드 트리거)
  - `State3 --"촬영 실행"--> Buffer` (촬영 시작)
- 상태 전이 (mermaid에 정의된 두 상태 기반):

```
[*] --> State1 (시스템 시작)
State1 --> State1       : VAD -> STT 대화 루프
State1 --> State3       : "약 가져왔어" 또는 Cloud 요청
State3 --> CaptureMode  : "촬영 실행"
CaptureMode --> State1  : OCR 성공 전송 완료
CaptureMode --> State3  : OCR 실패 재촬영
```

### Edge\_Node - Wait\_UX

**`wait_ux.py`** - `Wait_UX` 노드

- mermaid 엣지:
  - `VAD --> Wait_UX --> TTS` (즉시 응답: "네 어르신 말씀해주세요!")
  - `Wait_UX --> TTS` (고정 멘트: "오디스가 생각하고 있어요. 잠시만 기다려 주세요~")
  - `OCR_Engine --실패--> Wait_UX --> TTS` (재요청: "어르신 약봉투를 다시 들어주시겠어요?")
- 멘트 유형별 템플릿 관리:
  - `IMMEDIATE_RESPONSE`: 즉시 응답 멘트
  - `WAITING`: 처리 중 대기 멘트 (시간 경과에 따라 에스컬레이션)
  - `RETRY_REQUEST`: 재촬영/재확인 요청 멘트
  - `QUALITY_GUIDE`: 품질 미달 촬영 가이드 멘트 (명세 1항/5항)
  - `CONFIDENCE_CONFIRM`: 신뢰도 미달 확인 요청 (명세 3항/7항)
- 모든 출력은 `TTS`를 경유하여 `Speaker`로 전달 (mermaid 흐름 그대로)

### Edge\_Node - 인터페이스 (구현 범위 외, 추상 인터페이스만)

- **`vad.py`** - `VAD` 노드: Wake-word 감지 ABC
- **`stt.py`** - `STT` 노드: Whisper 스트리밍 ABC
- **`tts.py`** - `TTS` 노드: Qwen3-TTS ABC, `async def speak(text, priority)`

### Cloud\_Server (인터페이스/스텁)

**`drug_parser.py`** - `Drug_Parser` 노드

- mermaid 엣지: `OCR_Engine --성공--> Drug_Parser --> DB`
- 클라우드 API 호출 인터페이스: OCR 결과 JSON을 비동기 전송
- 명세 4항: 타임스탬프 동기화, 경량 JSON만 전송 (원본 이미지 미전송)

**`instruction_log.py`** - `Instruction_Log` 노드

- mermaid 엣지: `STT --> Instruction_Log --> DB`
- STT 텍스트를 타임스탬프와 함께 클라우드에 비동기 전송

**`db.py`** - `DB` 노드

- 환자 통합 MD파일 DB 추상 인터페이스

---

## 핵심 데이터 흐름 (mermaid 엣지 순서 그대로)

```
[촬영 흐름]
Cam --(RTSP)--> Buffer --(BestShot)--> Timer --(확정 프레임)--> OCR_Engine
                                        |                         |
                                        +--(카운트)--> TTS        +--성공--> Drug_Parser --> DB
                                                       |          +--실패--> Wait_UX --> TTS --> Speaker
                                                       v
                                                    Speaker

[호출 흐름]
User --(음성)--> STT --> VAD --> Wait_UX --> TTS --> Speaker
                  |
                  +--> Instruction_Log --> DB
```

---

## mermaid 노드 -> Python 모듈 매핑

| mermaid 서브그래프 | 노드 | 파일 |
|---|---|---|
| **Home\_Environment** | `Cam` | `src/home_environment/cam.py` |
| | `Speaker` | `src/home_environment/speaker.py` |
| **Edge\_Node** | `VAD` | `src/edge_node/vad.py` (ABC + Stub) |
| | `STT` | `src/edge_node/stt.py` (ABC + Stub) |
| | `TTS` | `src/edge_node/tts.py` (ABC + Stub) |
| | `Wait_UX` | `src/edge_node/wait_ux.py` |
| **Edge\_Node > Capture\_Mode** | `Buffer` | `src/edge_node/capture_mode/buffer.py` |
| | `Timer` | `src/edge_node/capture_mode/timer.py` |
| | `OCR_Engine` | `src/edge_node/capture_mode/ocr_engine.py` |
| **Edge\_Node > Dialogue\_Manager** | `State1` / `State3` | `src/edge_node/dialogue_manager/state_machine.py` |
| **Cloud\_Server** | `Drug_Parser` | `src/cloud_server/drug_parser.py` |
| | `Instruction_Log` | `src/cloud_server/instruction_log.py` |
| | `DB` | `src/cloud_server/db.py` |

---

## mermaid 엣지 연결 방식

`src/main.py`의 `LocalAgent` 클래스가 모든 노드를 조립하며, 각 엣지는 다음과 같이 구현된다:

- **asyncio.Queue**: `Wait_UX -> TTS`, `Timer -> TTS`, `Buffer -> Wait_UX(품질)`, `Timer -> OCR_Engine`
- **콜백 함수**: `StateMachine.on_transition` -> 촬영 흐름 트리거
- **비동기 태스크**: `_wakeword_loop` (엣지 1-2-8), `_tts_consumer_loop` (엣지 9-13), `_quality_feedback_loop` (명세 1/5항), `_run_capture_flow` (엣지 6-7-10-11)
- **직접 호출**: `OCR_Engine -> Drug_Parser` (엣지 10), `STT -> Instruction_Log` (엣지 3)

---

## 기술 스택

- **Python 3.10+**, asyncio 이벤트 루프
- **OpenCV** + GStreamer (Jetson HW RTSP 디코딩, 이미지 품질 분석)
- **GLM-OCR** (TensorRT FP16, Jetson GPU 가속)
- **aiohttp** (Cloud 비동기 통신)
- **pydantic** (데이터 모델 검증)
- **rapidfuzz** (의약품명 fuzzy matching)
- **PyYAML** (설정 관리)

---

## 실행

```bash
pip install -r requirements.txt
python -m src.main
```

## CI 품질 게이트

`local_agent/.github/workflows/ci.yml`의 `mock` job은 아래 하드 게이트를 수행한다.

- 모듈 컴파일: `python -m compileall src`
- 설정/상태머신 스모크: `StateMachine`, `WaitUX` 검증
- 클라우드 계약 게이트: `pytest -q tests/test_contract_payloads.py`
- TurboQuant 런타임 계약 게이트: `pytest -q tests/test_turboquant_runtime_contract.py`
- 일반 테스트가 존재하면 `pytest -q` 전체 실행

Jetson 배포 전 하드웨어 게이트는
`local_agent/.github/workflows/jetson-deploy.yml`의 `jetson-smoke` job에서 수행한다.
