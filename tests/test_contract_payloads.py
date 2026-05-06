"""Contract gates for local_agent cloud payloads."""
from __future__ import annotations

from src.cloud_server.drug_parser import DrugParserConfig, _to_server_ocr_payload
from src.cloud_server.instruction_log import InstructionEntry, InstructionLogConfig


def test_drug_parser_config_matches_ai_server_contract_paths() -> None:
    cfg = DrugParserConfig()
    assert cfg.path == "/api/ocr/analyze"
    assert cfg.endpoint.startswith("http://")


def test_ocr_payload_translation_to_server_contract() -> None:
    agent_payload = {
        "perception_timestamp": "2026-05-06T00:00:00+00:00",
        "speaker_id": "speaker-123",
        "ocr_results": {
            "text": "처방전 OCR 텍스트",
            "text_confidence_score": 0.92,
            "structured_data": {
                "drugs": [
                    {
                        "name": "혈압약A",
                        "dosage": "5 mg",
                        "frequency": "1일 2회",
                        "timing": "식후",
                    }
                ]
            },
        },
    }

    body = _to_server_ocr_payload(agent_payload)
    assert body["raw_text"] == "처방전 OCR 텍스트"
    assert body["confidence"] == 0.92
    assert body["speaker_id"] == "speaker-123"
    assert len(body["medications"]) == 1
    assert body["medications"][0] == {
        "name": "혈압약A",
        "strength": "5 mg",
        "dosage": "5 mg",
        "frequency": "1일 2회",
        "timing": "식후",
    }


def test_instruction_log_contract_defaults() -> None:
    cfg = InstructionLogConfig()
    assert cfg.path == "/api/stt/log"

    entry = InstructionEntry(text="약 가져왔어", timestamp=1715000000.0, source="stt")
    payload = entry.to_dict()
    assert payload["text"] == "약 가져왔어"
    assert payload["timestamp"] == 1715000000.0
    assert payload["source"] == "stt"
