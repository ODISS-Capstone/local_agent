"""
OCR_Engine [GLM-OCR: 텍스트 추출]

mermaid 노드: OCR_Engine
mermaid 엣지:
  - Timer --> OCR_Engine                      (확정 프레임 수신)
  - OCR_Engine --성공 시 OCR 전송--> Drug_Parser  (성공 -> 클라우드 전송)
  - OCR_Engine --실패 시 재요청--> Wait_UX         (실패 -> 재촬영 요청)

내부 기능:
  - ROI 분류: 처방전(PRESCRIPTION) / 알약(PILL_IMAGE) 자동 판별
  - 처방전 모드: 표 구조 Key-Value 추출 (약품명, 투여횟수, 투약일수)
  - 알약 모드: 텍스트 + 시각적 특징 추출 (형태, 색상, 분할선, 각인)
  - 신뢰도 평가: confidence < 0.85 -> 실패 판정
  - 의학용어 사전 fuzzy matching
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    PILL_IMAGE = "PILL_IMAGE"


class ActionRequired(str, Enum):
    PROCEED = "PROCEED_TO_IDENTIFICATION"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    RETRY = "RETRY_CAPTURE"


@dataclass
class VisualFeatures:
    shape: str
    color: str
    line: str


@dataclass
class DrugEntry:
    name: str
    dosage: str
    frequency: str
    days: int


@dataclass
class OCRResult:
    perception_timestamp: str
    input_type: str
    text: str
    text_confidence_score: float
    visual_features: dict[str, str] | None = None
    structured_data: dict[str, Any] | None = None
    action_required: str = ActionRequired.PROCEED.value

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "perception_timestamp": self.perception_timestamp,
            "input_type": self.input_type,
            "ocr_results": {
                "text": self.text,
                "text_confidence_score": self.text_confidence_score,
            },
            "action_required": self.action_required,
        }
        if self.visual_features is not None:
            result["ocr_results"]["visual_features"] = self.visual_features
        if self.structured_data is not None:
            result["ocr_results"]["structured_data"] = self.structured_data
        return result


@dataclass
class OCREngineConfig:
    model_path: str = "models/glm-ocr"
    confidence_threshold: float = 0.85
    fuzzy_match_max_distance: int = 2
    medical_dict_path: str = "config/medical_terms.json"


# ---------------------------------------------------------------------------
# Shape / Color 한글 매핑 상수
# ---------------------------------------------------------------------------
SHAPE_LABELS = {
    "circle": "원형",
    "ellipse": "타원형",
    "rectangle": "장방형",
    "other": "기타",
}

COLOR_RANGES_HSV = {
    "하얀색": ((0, 0, 200), (180, 30, 255)),
    "노란색": ((20, 100, 100), (35, 255, 255)),
    "주황색": ((10, 100, 100), (20, 255, 255)),
    "빨간색": ((0, 100, 100), (10, 255, 255)),
    "분홍색": ((150, 50, 100), (170, 255, 255)),
    "초록색": ((35, 100, 100), (85, 255, 255)),
    "파란색": ((85, 100, 100), (130, 255, 255)),
    "갈색": ((10, 50, 50), (20, 200, 150)),
    "투명": ((0, 0, 0), (180, 30, 50)),
}


class OCREngine:
    """GLM-OCR 기반 텍스트 추출 엔진.

    ROI 분류, 모드별 파싱, 신뢰도 검증, 의학사전 매칭을
    하나의 노드 안에서 수행한다.
    """

    def __init__(self, config: OCREngineConfig | None = None) -> None:
        self._config = config or OCREngineConfig()
        self._medical_terms: list[str] = []
        self._model_loaded = False

    async def load(self) -> None:
        """모델 및 의학 사전 로드."""
        self._load_medical_dict()
        await self._load_model()
        self._model_loaded = True

    def _load_medical_dict(self) -> None:
        dict_path = Path(self._config.medical_dict_path)
        if dict_path.exists():
            with open(dict_path, encoding="utf-8") as f:
                data = json.load(f)
            self._medical_terms = data.get("terms", [])
            logger.info("의학용어 사전 로드: %d개 항목", len(self._medical_terms))
        else:
            logger.warning("의학용어 사전 파일 없음: %s", dict_path)

    async def _load_model(self) -> None:
        """GLM-OCR 모델을 로드한다 (TensorRT FP16 최적화 대상)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        model_path = Path(self._config.model_path)
        if model_path.exists():
            logger.info("GLM-OCR 모델 로드: %s", model_path)
        else:
            logger.warning(
                "GLM-OCR 모델 경로 없음: %s (스텁 모드로 동작)", model_path
            )
        self._apply_turboquant_wrap()

    def _apply_turboquant_wrap(self) -> None:
        """Wrap the loaded HF model (if any) with TurboQuant compressed KV.

        Looks for a ``self._model`` / ``self._glm_model`` attribute that a
        concrete subclass may have populated; no-ops otherwise.  Keeps the
        stubbed path silent on CPU-only dev hosts.
        """
        try:
            from src.runtime.turboquant_runtime import wrap
        except Exception as exc:  # noqa: BLE001
            logger.debug("TurboQuant runtime unavailable: %s", exc)
            return

        for attr in ("_model", "_glm_model", "model"):
            model = getattr(self, attr, None)
            if model is None:
                continue
            setattr(self, attr, wrap(model))

    async def process(self, frame: np.ndarray) -> OCRResult:
        """Timer -> OCR_Engine: 프레임을 받아 OCR 처리를 수행한다.

        Returns:
            OCRResult:
              - 성공: action_required = PROCEED_TO_IDENTIFICATION
              - 신뢰도 미달: action_required = NEEDS_CONFIRMATION
              - 실패: action_required = RETRY_CAPTURE
        """
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        input_type = self._classify_roi(frame)

        if input_type == InputType.PRESCRIPTION:
            return await self._process_prescription(frame, timestamp)
        else:
            return await self._process_pill(frame, timestamp)

    # ------------------------------------------------------------------
    # ROI 분류
    # ------------------------------------------------------------------
    def _classify_roi(self, frame: np.ndarray) -> InputType:
        """처방전/알약 자동 판별.

        문서(처방전/약봉투)는 직선이 많고 텍스트 영역이 넓은 반면,
        알약은 원형/타원형 윤곽이 지배적이다.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=100, param2=40, minRadius=15, maxRadius=200,
        )
        circle_count = len(circles[0]) if circles is not None else 0

        if line_count > circle_count * 3:
            logger.debug("ROI 분류: PRESCRIPTION (lines=%d, circles=%d)", line_count, circle_count)
            return InputType.PRESCRIPTION
        else:
            logger.debug("ROI 분류: PILL_IMAGE (lines=%d, circles=%d)", line_count, circle_count)
            return InputType.PILL_IMAGE

    # ------------------------------------------------------------------
    # 처방전 모드
    # ------------------------------------------------------------------
    async def _process_prescription(self, frame: np.ndarray, timestamp: str) -> OCRResult:
        loop = asyncio.get_running_loop()
        raw_texts, confidences = await loop.run_in_executor(
            None, self._run_ocr_inference, frame
        )

        if not raw_texts:
            return OCRResult(
                perception_timestamp=timestamp,
                input_type=InputType.PRESCRIPTION.value,
                text="",
                text_confidence_score=0.0,
                action_required=ActionRequired.RETRY.value,
            )

        avg_confidence = sum(confidences) / len(confidences)
        full_text = " ".join(raw_texts)

        drugs = self._parse_prescription_table(raw_texts)
        action = self._determine_action(raw_texts, confidences)

        return OCRResult(
            perception_timestamp=timestamp,
            input_type=InputType.PRESCRIPTION.value,
            text=full_text,
            text_confidence_score=round(avg_confidence, 3),
            structured_data={"drugs": [vars(d) for d in drugs]} if drugs else None,
            action_required=action.value,
        )

    def _parse_prescription_table(self, texts: list[str]) -> list[DrugEntry]:
        """처방전 표 구조에서 약품명/용법/일수를 추출한다.

        실제 GLM-OCR 결과의 bbox 기반 표 재구성은 모델 출력 포맷에 의존하며,
        여기서는 텍스트 패턴 기반 추출 로직을 제공한다.
        """
        drugs: list[DrugEntry] = []
        for text in texts:
            matched = self._fuzzy_match_drug_name(text)
            if matched is not None:
                drugs.append(DrugEntry(
                    name=matched,
                    dosage="",
                    frequency="",
                    days=0,
                ))
        return drugs

    # ------------------------------------------------------------------
    # 알약 모드
    # ------------------------------------------------------------------
    async def _process_pill(self, frame: np.ndarray, timestamp: str) -> OCRResult:
        loop = asyncio.get_running_loop()

        raw_texts, confidences = await loop.run_in_executor(
            None, self._run_ocr_inference, frame
        )
        visual = await loop.run_in_executor(None, self._extract_visual_features, frame)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = " ".join(raw_texts) if raw_texts else ""

        action = self._determine_action(raw_texts, confidences) if raw_texts else ActionRequired.RETRY

        return OCRResult(
            perception_timestamp=timestamp,
            input_type=InputType.PILL_IMAGE.value,
            text=full_text,
            text_confidence_score=round(avg_confidence, 3),
            visual_features={
                "shape": visual.shape,
                "color": visual.color,
                "line": visual.line,
            },
            action_required=action.value,
        )

    def _extract_visual_features(self, frame: np.ndarray) -> VisualFeatures:
        """알약의 형태, 색상, 분할선을 분석한다."""
        shape = self._detect_shape(frame)
        color = self._detect_color(frame)
        has_line = self._detect_split_line(frame)
        return VisualFeatures(shape=shape, color=color, line="있음" if has_line else "없음")

    def _detect_shape(self, frame: np.ndarray) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return SHAPE_LABELS["other"]

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter == 0:
            return SHAPE_LABELS["other"]

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(largest)
        w, h = rect[1]
        if min(w, h) == 0:
            return SHAPE_LABELS["other"]
        aspect_ratio = max(w, h) / min(w, h)

        if circularity > 0.85:
            return SHAPE_LABELS["circle"]
        elif aspect_ratio > 1.5:
            return SHAPE_LABELS["rectangle"] if circularity < 0.6 else SHAPE_LABELS["ellipse"]
        else:
            return SHAPE_LABELS["ellipse"]

    def _detect_color(self, frame: np.ndarray) -> str:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_color = "기타"
        best_ratio = 0.0
        total = hsv.shape[0] * hsv.shape[1]
        for color_name, (lower, upper) in COLOR_RANGES_HSV.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = float(cv2.countNonZero(mask)) / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name
        return best_color

    def _detect_split_line(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=5)
        if lines is None:
            return False
        h, w = gray.shape
        center_y = h // 2
        tolerance = h // 6
        for line in lines:
            _, y1, _, y2 = line[0]
            if abs(y1 - center_y) < tolerance and abs(y2 - center_y) < tolerance:
                return True
        return False

    # ------------------------------------------------------------------
    # GLM-OCR 추론
    # ------------------------------------------------------------------
    def _run_ocr_inference(self, frame: np.ndarray) -> tuple[list[str], list[float]]:
        """GLM-OCR 모델 추론.

        실제 GLM-OCR 모델 연동 시 이 메서드를 교체한다.
        현재는 OpenCV 전처리 + 스텁 형태로, 모델 통합 지점을 명확히 한다.
        """
        if not self._model_loaded:
            logger.warning("OCR 모델 미로드 상태: 빈 결과 반환")
            return [], []

        # --- GLM-OCR 모델 추론 통합 지점 ---
        # model_output = self._glm_model.infer(frame)
        # texts = [r.text for r in model_output.regions]
        # confidences = [r.confidence for r in model_output.regions]
        # return texts, confidences

        logger.info("GLM-OCR 추론 실행 (스텁): frame shape=%s", frame.shape)
        return [], []

    # ------------------------------------------------------------------
    # 신뢰도 검증 및 의학사전 매칭
    # ------------------------------------------------------------------
    def _determine_action(
        self, texts: list[str], confidences: list[float]
    ) -> ActionRequired:
        if not texts or not confidences:
            return ActionRequired.RETRY

        avg_conf = sum(confidences) / len(confidences)
        if avg_conf < self._config.confidence_threshold:
            return ActionRequired.NEEDS_CONFIRMATION

        for text, conf in zip(texts, confidences):
            if conf < self._config.confidence_threshold:
                matched = self._fuzzy_match_drug_name(text)
                if matched is None:
                    return ActionRequired.NEEDS_CONFIRMATION

        return ActionRequired.PROCEED

    def _fuzzy_match_drug_name(self, text: str) -> str | None:
        """의학용어 사전과 fuzzy matching하여 가장 가까운 약품명을 반환한다."""
        if not self._medical_terms or not text.strip():
            return None

        result = process.extractOne(
            text.strip(),
            self._medical_terms,
            scorer=fuzz.ratio,
            score_cutoff=70,
        )
        if result is None:
            return None

        matched_term, score, _ = result
        distance = int((100 - score) * len(matched_term) / 100)
        if distance <= self._config.fuzzy_match_max_distance:
            return matched_term
        return None
