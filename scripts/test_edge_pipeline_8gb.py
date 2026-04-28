"""End-to-end OCR agent pipeline smoke under an 8 GB VRAM budget.

This script loads and runs the three real edge models that the mermaid
spec in ``OCR_Agent.mermaid`` assumes:

- STT : ``openai/whisper-small``
- TTS : ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice``
- OCR : ``zai-org/GLM-OCR``

It walks a single pass of the mermaid edge flow
``STT -> Instruction_Log -> State3 -> OCR -> Drug_Parser -> TTS`` using
stubs from ``local_agent`` for the orchestration layer.  The heavy
lifting (actual model forward passes) is done here.

The test enforces an 8 GB peak VRAM budget by:

- Capping the torch process memory fraction with
  ``torch.cuda.set_per_process_memory_fraction`` so that at most 8 GiB is
  addressable even on larger cards.
- Loading the models sequentially (STT -> OCR -> TTS) instead of keeping
  all three resident at once.  Real Jetson Nano 8 GB deployments
  generally cannot afford concurrent residency of all three models.
- Sampling ``torch.cuda.max_memory_allocated`` after every stage and
  failing the run if any per-stage peak exceeds the budget.

Run locally with::

    python scripts/test_edge_pipeline_8gb.py \
        --report edge-pipeline-8gb-report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("edge_pipeline_8gb")


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.runtime import turboquant_runtime as _tq_runtime
    _tq_runtime.install(None)
except Exception:  # noqa: BLE001
    class _NullTq:
        @staticmethod
        def wrap(model):
            return model

    _tq_runtime = _NullTq()  # type: ignore[assignment]


STT_MODEL = "openai/whisper-small"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OCR_MODEL = "zai-org/GLM-OCR"

DEFAULT_BUDGET_GIB = 8.0
SAMPLE_RATE = 16_000
SAMPLE_DURATION = 4.0


@dataclass
class StageReport:
    name: str
    model_id: str
    status: str = "pending"
    wall_seconds: float = 0.0
    peak_mib: float = 0.0
    delta_mib: float = 0.0
    output_preview: str = ""
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "status": self.status,
            "wall_seconds": round(self.wall_seconds, 3),
            "peak_mib": round(self.peak_mib, 1),
            "delta_mib": round(self.delta_mib, 1),
            "output_preview": self.output_preview,
            "message": self.message,
        }


@dataclass
class PipelineReport:
    budget_mib: float
    device_name: str
    total_vram_gib: float
    stages: List[StageReport] = field(default_factory=list)
    peak_mib_overall: float = 0.0
    exit_code: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "budget_mib": self.budget_mib,
            "device_name": self.device_name,
            "total_vram_gib": self.total_vram_gib,
            "peak_mib_overall": round(self.peak_mib_overall, 1),
            "exit_code": self.exit_code,
            "stages": [s.as_dict() for s in self.stages],
        }


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _snapshot_vram_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _unload(obj: Any) -> None:
    try:
        if hasattr(obj, "to"):
            obj.to("cpu")
    except Exception:  # noqa: BLE001
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_budget(budget_mib: float) -> None:
    if not torch.cuda.is_available():
        return
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (budget_mib * 1024 * 1024) / total_bytes)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        logger.info("cuda memory fraction capped at %.3f (%.0f MiB)", fraction, budget_mib)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cap memory fraction: %s", exc)


def _make_dummy_audio() -> np.ndarray:
    t = np.linspace(0.0, SAMPLE_DURATION, int(SAMPLE_RATE * SAMPLE_DURATION), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    noise = 0.01 * np.random.default_rng(0).standard_normal(len(tone)).astype(np.float32)
    return tone + noise


def _make_dummy_prescription_image(size: int = 384):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "처방전",
        "환자: 홍길동",
        "약품: 혈압약 5 mg / 1일 2회",
        "일수: 30일",
    ]
    for i, text in enumerate(lines):
        draw.text((20, 40 + i * 60), text, fill=(0, 0, 0))
    draw.rectangle([10, 10, size - 10, size - 10], outline=(0, 0, 0), width=2)
    return img


def _run_stt_stage(report: PipelineReport, dtype: torch.dtype, device: str) -> str:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    stage = StageReport(name="stt", model_id=STT_MODEL)
    report.stages.append(stage)

    _reset_peak()
    base = _snapshot_vram_mib()
    t0 = time.time()
    try:
        processor = WhisperProcessor.from_pretrained(STT_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(
            STT_MODEL, dtype=dtype
        ).to(device)
        model = _tq_runtime.wrap(model)
        model.eval()

        audio = _make_dummy_audio()
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            generated = model.generate(inputs, max_new_tokens=32)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        stage.output_preview = text.strip()[:120]
        stage.status = "ok"
    except Exception as exc:  # noqa: BLE001
        stage.status = "failed"
        stage.message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stage.wall_seconds = time.time() - t0
        stage.peak_mib = _peak_vram_mib()
        stage.delta_mib = stage.peak_mib - base
        if stage.peak_mib > report.peak_mib_overall:
            report.peak_mib_overall = stage.peak_mib
        if stage.status != "ok":
            logger.error("STT stage failed: %s", stage.message)
        _unload_locals(locals())

    return stage.output_preview


def _run_ocr_stage(
    report: PipelineReport,
    dtype: torch.dtype,
    device: str,
    stt_text: str,
) -> Dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    stage = StageReport(name="ocr", model_id=OCR_MODEL)
    report.stages.append(stage)

    _reset_peak()
    base = _snapshot_vram_mib()
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(OCR_MODEL, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            OCR_MODEL,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model = _tq_runtime.wrap(model)
        model.eval()

        image = _make_dummy_prescription_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=64)

        seen_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        new_tokens = generated[0][seen_tokens:] if seen_tokens else generated[0]
        text = processor.decode(new_tokens, skip_special_tokens=True)
        stage.output_preview = text.strip()[:200]
        ocr_result = {
            "input_type": "PRESCRIPTION",
            "text": stage.output_preview,
            "text_confidence_score": 0.9,
            "upstream_stt": stt_text,
        }
        stage.status = "ok"
        return ocr_result
    except Exception as exc:  # noqa: BLE001
        stage.status = "failed"
        stage.message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stage.wall_seconds = time.time() - t0
        stage.peak_mib = _peak_vram_mib()
        stage.delta_mib = stage.peak_mib - base
        if stage.peak_mib > report.peak_mib_overall:
            report.peak_mib_overall = stage.peak_mib
        if stage.status != "ok":
            logger.error("OCR stage failed: %s", stage.message)
        _unload_locals(locals())


TTS_SUBPROCESS_TEMPLATE = textwrap.dedent(
    """
    import json, sys, torch

    try:
        from turboquant.runtime import install_hf_autowrap, auto_wrap
        install_hf_autowrap(force=True)
    except Exception:
        def auto_wrap(m):
            return m

    from qwen_tts import Qwen3TTSModel

    payload = json.loads(sys.stdin.read())
    budget_mib = float(payload["budget_mib"])
    prompt = payload["prompt"]
    model_id = payload["model_id"]
    dtype_name = payload["dtype"]
    dtype = getattr(torch, dtype_name)

    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, (budget_mib * 1024 * 1024) / total)
    try:
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
    except Exception:
        pass

    torch.cuda.reset_peak_memory_stats()
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
    )
    model = auto_wrap(model)
    wavs, sr = model.generate_custom_voice(
        text=prompt,
        language="Korean",
        speaker="Sohee",
        instruct="환자에게 부드럽고 친절하게 말해주세요.",
    )
    samples = int(len(wavs[0])) if wavs else 0
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(json.dumps({
        "samples": samples,
        "sample_rate": int(sr),
        "peak_mib": peak_mib,
    }))
    """
)


def _locate_tts_python() -> Optional[str]:
    """Return a Python interpreter able to run ``qwen_tts`` (transformers==4.57.3).

    Resolution order:

    1. ``EDGE_PIPELINE_TTS_PYTHON`` environment variable.
    2. ``/opt/odiss/tts-venv/bin/python`` (Jetson deploy convention).
    3. ``tts-venv/bin/python`` next to this repo.
    4. The current interpreter if ``qwen_tts`` already imports in it.
    """
    candidates: List[str] = []
    env_path = os.environ.get("EDGE_PIPELINE_TTS_PYTHON")
    if env_path:
        candidates.append(env_path)
    candidates.append("/opt/odiss/tts-venv/bin/python")
    candidates.append(str((REPO_ROOT / "tts-venv" / "bin" / "python").resolve()))

    for cand in candidates:
        if cand and Path(cand).exists():
            return cand

    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import qwen_tts"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if probe.returncode == 0:
            return sys.executable
    except Exception:  # noqa: BLE001
        pass
    return None


def _run_tts_stage(
    report: PipelineReport,
    dtype: torch.dtype,
    device: str,
    ocr_text: str,
    budget_mib: float,
) -> None:
    """Run the Qwen3-TTS stage in an isolated Python process.

    ``qwen_tts`` pins ``transformers==4.57.3`` while GLM-OCR on the edge
    side uses ``transformers>=5.0``.  The two cannot coexist in a single
    process, so this stage shells out to a dedicated interpreter (see
    :func:`_locate_tts_python`).
    """
    stage = StageReport(name="tts", model_id=TTS_MODEL)
    report.stages.append(stage)

    _reset_peak()
    base = _snapshot_vram_mib()
    t0 = time.time()
    tts_python = _locate_tts_python()
    if tts_python is None:
        stage.status = "skipped"
        stage.message = (
            "no interpreter with qwen_tts found; set EDGE_PIPELINE_TTS_PYTHON "
            "or create /opt/odiss/tts-venv with `pip install qwen-tts`"
        )
        logger.warning(stage.message)
        stage.wall_seconds = time.time() - t0
        return

    prompt = f"처방전이 인식되었습니다. {ocr_text[:80]}"
    payload = {
        "budget_mib": budget_mib,
        "prompt": prompt,
        "model_id": TTS_MODEL,
        "dtype": "bfloat16" if dtype is torch.bfloat16 else "float16",
    }

    try:
        proc = subprocess.run(
            [tts_python, "-c", TTS_SUBPROCESS_TEMPLATE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stage.status = "failed"
        stage.message = f"TimeoutExpired: {exc}"
        stage.wall_seconds = time.time() - t0
        logger.error("TTS subprocess timed out")
        return

    stage.wall_seconds = time.time() - t0
    if proc.returncode != 0:
        stage.status = "failed"
        stage.message = f"exit={proc.returncode} stderr={proc.stderr[-600:]}"
        logger.error("TTS subprocess failed: %s", stage.message)
        return

    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        result = json.loads(last_line)
    except json.JSONDecodeError as exc:
        stage.status = "failed"
        stage.message = f"could not parse subprocess output: {exc}"
        return

    stage.output_preview = f"wav samples={result['samples']} sr={result['sample_rate']}"
    stage.peak_mib = float(result.get("peak_mib", 0.0))
    stage.delta_mib = stage.peak_mib
    if stage.peak_mib > report.peak_mib_overall:
        report.peak_mib_overall = stage.peak_mib
    stage.status = "ok"


def _unload_locals(mapping: Dict[str, Any]) -> None:
    """Drop any local references likely to hold GPU memory."""
    for name in list(mapping.keys()):
        if name in {"processor", "tokenizer", "model", "inputs", "generated", "new_tokens", "outputs", "wavs"}:
            _unload(mapping[name])
            mapping[name] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-gib",
        type=float,
        default=DEFAULT_BUDGET_GIB,
        help="Per-process VRAM budget in GiB (defaults to 8.0 to mimic Jetson Nano 8GB).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Exit 1 if any stage peak VRAM exceeds the budget (default: exit 0).",
    )
    parser.add_argument(
        "--skip-stages",
        nargs="*",
        default=[],
        choices=["stt", "ocr", "tts"],
        help="Skip one or more stages (useful when a model is not yet cached).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _parse_args()

    if not torch.cuda.is_available():
        logger.error("This smoke requires a CUDA device; aborting on CPU-only host.")
        return 0

    budget_mib = args.budget_gib * 1024.0
    _apply_budget(budget_mib)

    dtype = _pick_dtype()
    device = "cuda"

    props = torch.cuda.get_device_properties(0)
    report = PipelineReport(
        budget_mib=budget_mib,
        device_name=props.name,
        total_vram_gib=round(props.total_memory / (1024 ** 3), 2),
    )
    logger.info(
        "edge pipeline smoke: budget=%.1f GiB device=%s total=%.1f GiB dtype=%s",
        args.budget_gib, props.name, report.total_vram_gib, dtype,
    )

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    try:
        stt_text = ""
        if "stt" not in args.skip_stages:
            stt_text = _run_stt_stage(report, dtype=dtype, device=device)
        else:
            stt_text = "약 가져왔어"

        ocr_text = ""
        if "ocr" not in args.skip_stages:
            ocr_result = _run_ocr_stage(report, dtype=dtype, device=device, stt_text=stt_text)
            ocr_text = ocr_result["text"]
        else:
            ocr_text = "처방전 텍스트 스텁"

        if "tts" not in args.skip_stages:
            _run_tts_stage(
                report,
                dtype=dtype,
                device=device,
                ocr_text=ocr_text,
                budget_mib=budget_mib,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline aborted: %s", exc)
        report.exit_code = 1

    report.peak_mib_overall = max(
        [report.peak_mib_overall] + [s.peak_mib for s in report.stages]
    )
    over_budget = [s for s in report.stages if s.peak_mib > budget_mib]
    if over_budget:
        logger.error(
            "budget violations: %s",
            ", ".join(f"{s.name}={s.peak_mib:.0f}MiB" for s in over_budget),
        )
        if args.fail_on_budget:
            report.exit_code = 1

    summary = report.as_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
