"""
카메라 프레임 또는 로컬 이미지로 GLM-OCR 단독 테스트.

사전 준비:
    cp .env.example .env
    # .env 안의 GEMINI_API_KEY 또는 RTSP_URL 설정

사용법:
    cd ~/local_agent
    python3 -m scripts.test_glm_ocr
    python3 -m scripts.test_glm_ocr --image /tmp/cam_onvif1.jpg

출력:
    - runtime/ocr/test_input.jpg 에 입력 이미지 저장
    - GLM-OCR 결과 JSON을 콘솔 출력
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import cv2
from src.config_loader import load_config
from src.edge_node.capture_mode.ocr_engine import OCREngine, OCREngineConfig


def read_image_from_camera(rtsp_url: str) -> tuple[bool, object]:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;15000000"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return False, None
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True, frame
        return False, None
    finally:
        cap.release()


async def run(args: argparse.Namespace) -> None:
    cfg = load_config()
    ocr_cfg = cfg.get("ocr", {})

    provider = ocr_cfg.get("provider")
    gemini_key = (
        ocr_cfg.get("gemini_api_key")
        or os.environ.get(ocr_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"))
        or ""
    ).strip()
    if provider == "gemini_ocr" and not gemini_key:
        raise SystemExit(
            "Gemini API 키가 없습니다. config/agent_config.yaml의 ocr.gemini_api_key "
            "또는 GEMINI_API_KEY 환경변수를 설정하세요."
        )
    if provider not in ("hf_glm_ocr", "gemini_ocr") and not os.environ.get(
        ocr_cfg.get("glmocr_api_key_env", "ZHIPU_API_KEY")
    ):
        raise SystemExit(
            "ZHIPU_API_KEY 환경변수가 없습니다. 먼저 `export ZHIPU_API_KEY='sk-...'` 를 실행하세요."
        )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image}")
    else:
        ok, frame = read_image_from_camera(cfg["rtsp"]["url"])
        if not ok:
            raise SystemExit(f"카메라 프레임을 읽을 수 없습니다: {cfg['rtsp']['url']}")

    save_dir = Path(ocr_cfg.get("glmocr_save_dir", "runtime/ocr"))
    save_dir.mkdir(parents=True, exist_ok=True)
    input_path = save_dir / "test_input.jpg"
    cv2.imwrite(str(input_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[test] input saved: {input_path} shape={frame.shape}")

    engine = OCREngine(OCREngineConfig(
        model_path=ocr_cfg.get("model_path", "models/glm-ocr"),
        provider=ocr_cfg.get("provider", "glmocr"),
        hf_device=ocr_cfg.get("hf_device", "auto"),
        hf_torch_dtype=ocr_cfg.get("hf_torch_dtype", "auto"),
        hf_prompt=ocr_cfg.get("hf_prompt", "Text Recognition:"),
        hf_max_new_tokens=ocr_cfg.get("hf_max_new_tokens", 4096),
        hf_max_image_side=ocr_cfg.get("hf_max_image_side", 1280),
        hf_extract_document=ocr_cfg.get("hf_extract_document", True),
        hf_repetition_penalty=ocr_cfg.get("hf_repetition_penalty", 1.15),
        hf_no_repeat_ngram_size=ocr_cfg.get("hf_no_repeat_ngram_size", 8),
        gemini_model=ocr_cfg.get("gemini_model", "gemini-3-flash"),
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
    await engine.load()
    result = await engine.process(frame)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="카메라 대신 OCR할 로컬 이미지 경로")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
