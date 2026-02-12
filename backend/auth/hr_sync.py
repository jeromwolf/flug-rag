"""
인사 DB 연동 모듈 (스텁)

SFR-014: 인사 DB 동기화
- 일일 동기화 스케줄러
- 신규/퇴직/부서이동 자동 반영
- 초기 구현: CSV 임포트 지원
"""

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HRRecord:
    """인사 정보 레코드"""
    employee_id: str
    username: str
    full_name: str
    email: str
    department: str
    role: str  # admin, manager, user, viewer
    is_active: bool = True


class BaseHRProvider(ABC):
    """인사 DB 프로바이더 인터페이스"""

    @abstractmethod
    async def fetch_all(self) -> list[HRRecord]:
        """전체 인사 정보 조회"""

    @abstractmethod
    async def fetch_changes(self, since: datetime) -> list[HRRecord]:
        """변경된 인사 정보 조회"""


class CSVHRProvider(BaseHRProvider):
    """CSV 파일 기반 인사 정보 프로바이더 (초기 구현용)"""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    async def fetch_all(self) -> list[HRRecord]:
        """CSV 파일에서 전체 인사 정보 로드"""
        if not self.csv_path.exists():
            logger.warning(f"HR CSV file not found: {self.csv_path}")
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(HRRecord(
                    employee_id=row.get("employee_id", ""),
                    username=row.get("username", ""),
                    full_name=row.get("full_name", row.get("name", "")),
                    email=row.get("email", ""),
                    department=row.get("department", ""),
                    role=row.get("role", "user"),
                    is_active=row.get("is_active", "true").lower() == "true",
                ))
        logger.info(f"Loaded {len(records)} HR records from CSV")
        return records

    async def fetch_changes(self, since: datetime) -> list[HRRecord]:
        """CSV는 변경 추적 불가 - 전체 반환"""
        return await self.fetch_all()


class DatabaseHRProvider(BaseHRProvider):
    """데이터베이스 기반 인사 정보 프로바이더 (향후 구현)"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        logger.info("DatabaseHRProvider initialized (stub)")

    async def fetch_all(self) -> list[HRRecord]:
        raise NotImplementedError("Database HR provider not yet implemented")

    async def fetch_changes(self, since: datetime) -> list[HRRecord]:
        raise NotImplementedError("Database HR provider not yet implemented")


@dataclass
class SyncResult:
    """동기화 결과"""
    new_users: int = 0
    deactivated_users: int = 0
    department_changes: int = 0
    role_changes: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


async def sync_hr_data(provider: BaseHRProvider) -> SyncResult:
    """인사 데이터 동기화 실행

    Args:
        provider: HR 데이터 프로바이더

    Returns:
        SyncResult with counts of changes
    """
    from auth.models import Role, User
    from auth.user_store import get_user_store

    result = SyncResult()
    role_map = {r.value: r for r in Role}

    try:
        records = await provider.fetch_all()
    except Exception as e:
        result.errors.append(f"HR data fetch failed: {e}")
        return result

    user_store = await get_user_store()
    existing_users = {u.id: u for u in await user_store.list_users()}

    for record in records:
        try:
            existing = existing_users.get(record.employee_id)

            if existing is None:
                # 신규 사용자
                new_role = role_map.get(record.role, Role.USER)
                new_user = User(
                    id=record.employee_id,
                    username=record.username,
                    email=record.email,
                    full_name=record.full_name,
                    department=record.department,
                    role=new_role,
                    is_active=record.is_active,
                )
                await user_store.create_user(new_user)
                result.new_users += 1
            else:
                # 기존 사용자 업데이트
                if not record.is_active and existing.is_active:
                    await user_store.set_active(record.employee_id, False)
                    result.deactivated_users += 1

                if existing.department != record.department:
                    await user_store.update_department(record.employee_id, record.department)
                    result.department_changes += 1

                new_role = role_map.get(record.role)
                if new_role and existing.role != new_role:
                    await user_store.update_role(record.employee_id, new_role)
                    result.role_changes += 1

        except Exception as e:
            result.errors.append(f"Error processing {record.username}: {e}")

    logger.info(
        f"HR sync completed: {result.new_users} new, "
        f"{result.deactivated_users} deactivated, "
        f"{result.department_changes} dept changes"
    )
    return result
