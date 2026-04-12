"""
DB [(환자 통합 MD파일 DB)]

mermaid 노드: DB
mermaid 엣지:
  - Drug_Parser --> DB
  - Instruction_Log --> DB

환자 통합 MD파일 DB 추상 인터페이스.
실제 DB는 클라우드 ODISS 에이전트 측에서 관리하며,
이 모듈은 로컬 에이전트에서 클라우드 DB와 상호작용하기 위한 인터페이스를 제공한다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PatientRecord:
    patient_id: str
    prescriptions: list[dict[str, Any]]
    instruction_logs: list[dict[str, Any]]


class PatientDB(ABC):
    """환자 통합 DB 추상 인터페이스."""

    @abstractmethod
    async def get_patient(self, patient_id: str) -> PatientRecord | None:
        """환자 정보를 조회한다."""
        ...

    @abstractmethod
    async def get_active_prescriptions(self, patient_id: str) -> list[dict[str, Any]]:
        """현재 활성 처방 목록을 조회한다.

        Cloud_Server --"주기적 처방 레포트 요청"--> State3 과 연계.
        """
        ...


class StubPatientDB(PatientDB):
    """테스트/개발용 DB 스텁."""

    def __init__(self) -> None:
        self._records: dict[str, PatientRecord] = {}

    async def get_patient(self, patient_id: str) -> PatientRecord | None:
        return self._records.get(patient_id)

    async def get_active_prescriptions(self, patient_id: str) -> list[dict[str, Any]]:
        record = self._records.get(patient_id)
        if record is None:
            return []
        return record.prescriptions

    def add_record(self, record: PatientRecord) -> None:
        self._records[record.patient_id] = record
        logger.info("StubDB: 환자 레코드 추가 '%s'", record.patient_id)
