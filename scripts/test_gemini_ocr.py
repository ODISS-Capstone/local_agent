"""
카메라 프레임 또는 로컬 이미지로 Gemini OCR 단독 테스트.

사전 준비:
    cp .env.example .env
    # .env 안의 GEMINI_API_KEY 또는 RTSP_URL 설정

사용법:
    cd ~/local_agent
    python3 -m scripts.test_gemini_ocr
    python3 -m scripts.test_gemini_ocr --image /tmp/cam_onvif1.jpg

출력:
    - runtime/ocr/test_input.jpg 에 입력 이미지 저장
    - Gemini OCR 결과 JSON을 콘솔 출력
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
    provider = ocr_cfg.get("provider", "gemini_ocr")
    if provider != "gemini_ocr":
        raise SystemExit(
            f"Gemini 전용 테스트입니다. ocr.provider를 gemini_ocr로 설정하세요: {provider}"
        )

    gemini_key = (
        ocr_cfg.get("gemini_api_key")
        or os.environ.get(ocr_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"))
        or ""
    ).strip()
    if not gemini_key:
        raise SystemExit(
            "Gemini API 키가 없습니다. .env의 GEMINI_API_KEY 또는 "
            "config/agent_config.yaml의 ocr.gemini_api_key를 설정하세요."
        )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image}")
    else:
        ok, frame = read_image_from_camera(cfg["rtsp"]["url"])
        if not ok:
            raise SystemExit(f"카메라 프레임을 읽을 수 없습니다: {cfg['rtsp']['url']}")

    save_dir = Path(ocr_cfg.get("save_dir", "runtime/ocr"))
    save_dir.mkdir(parents=True, exist_ok=True)
    input_path = save_dir / "test_input.jpg"
    cv2.imwrite(str(input_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"[test] input saved: {input_path} shape={frame.shape}")

    engine = OCREngine(OCREngineConfig(
        provider="gemini_ocr",
        save_dir=str(save_dir),
        gemini_model=ocr_cfg.get("gemini_model", "gemini-3-flash-preview"),
        gemini_fallback_models=ocr_cfg.get(
            "gemini_fallback_models",
            ["gemini-2.5-flash", "gemini-flash-latest"],
        ),
        gemini_api_key=ocr_cfg.get("gemini_api_key", ""),
        gemini_api_key_env=ocr_cfg.get("gemini_api_key_env", "GEMINI_API_KEY"),
        gemini_prompt=ocr_cfg.get("gemini_prompt", ""),
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
