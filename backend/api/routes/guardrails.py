"""
Guardrails 관리 API
SFR-003: 필터 규칙 CRUD + 필터링 이력 조회
"""
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth.dependencies import require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/guardrails", tags=["guardrails"])


class RuleCreateRequest(BaseModel):
    name: str = Field(..., description="규칙 이름")
    rule_type: Literal["input_keyword", "input_pattern", "output_keyword", "output_pattern"] = Field(
        ..., description="규칙 유형"
    )
    pattern: str = Field(..., description="키워드 또는 정규식 패턴")
    action: Literal["block", "warn", "mask"] = Field("block", description="동작")
    message: str = Field("", description="사용자 안내 메시지")
    priority: int = Field(0, description="우선순위 (높을수록 먼저)")


class RuleUpdateRequest(BaseModel):
    name: str | None = None
    rule_type: Literal["input_keyword", "input_pattern", "output_keyword", "output_pattern"] | None = None
    pattern: str | None = None
    action: Literal["block", "warn", "mask"] | None = None
    message: str | None = None
    priority: int | None = None
    is_active: bool | None = None


class GuardrailTestRequest(BaseModel):
    test_input: str = Field(..., description="테스트 입력 텍스트")
    rule_id: str | None = Field(None, description="테스트할 규칙 ID (선택)")


@router.get("/admin/guardrails")
async def list_guardrail_rules(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """Guardrail 규칙 목록 (모든 규칙, 비활성 포함)."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()
    rules = await manager.list_all_rules()
    return {"rules": [r.__dict__ for r in rules]}


@router.post("/admin/guardrails")
async def create_guardrail_rule(
    request: RuleCreateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """Guardrail 규칙 추가."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()

    # Validate regex pattern
    if request.rule_type.endswith("_pattern"):
        import re
        try:
            re.compile(request.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 정규식: {e}")

    rule = await manager.add_rule(
        name=request.name,
        rule_type=request.rule_type,
        pattern=request.pattern,
        action=request.action,
        message=request.message,
        priority=request.priority,
        created_by=current_user.username,
    )
    logger.info("Guardrail rule created by %s: %s", current_user.username, rule.name)
    return {"status": "created", "rule": rule.__dict__}


@router.put("/admin/guardrails/{rule_id}")
async def update_guardrail_rule(
    rule_id: str,
    request: RuleUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """Guardrail 규칙 수정."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    # Validate regex if pattern is being updated
    if "pattern" in updates:
        # Determine rule_type: from update payload or existing rule
        rule_type = updates.get("rule_type")
        if not rule_type:
            existing_rules = await manager.list_all_rules()
            existing = next((r for r in existing_rules if r.id == rule_id), None)
            rule_type = existing.rule_type if existing else ""

        if rule_type and rule_type.endswith("_pattern"):
            import re
            try:
                re.compile(updates["pattern"])
            except re.error as e:
                raise HTTPException(status_code=400, detail=f"유효하지 않은 정규식: {e}")

    try:
        rule = await manager.update_rule(rule_id, **updates)
        logger.info("Guardrail rule updated by %s: %s", current_user.username, rule.name)
        return {"status": "updated", "rule": rule.__dict__}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/admin/guardrails/{rule_id}")
async def delete_guardrail_rule(
    rule_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """Guardrail 규칙 삭제."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()
    try:
        await manager.delete_rule(rule_id)
        logger.info("Guardrail rule deleted by %s: %s", current_user.username, rule_id)
        return {"status": "deleted", "rule_id": rule_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/admin/guardrails/logs")
async def get_guardrail_logs(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
    limit: Annotated[int, Query(le=200)] = 50,
    direction: Literal["input", "output"] | None = None,
):
    """Guardrail 필터링 이력 조회."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()
    logs = await manager.get_logs(limit=limit, direction=direction)
    return {"logs": [l.__dict__ for l in logs]}


@router.post("/admin/guardrails/test")
async def test_guardrail_input(
    request: GuardrailTestRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """입력 텍스트에 대해 Guardrail 테스트."""
    from rag.guardrails import get_guardrails_manager

    manager = await get_guardrails_manager()
    result = await manager.check_input(request.test_input, user_id=current_user.id)
    return {
        "passed": result.passed,
        "action": result.action,
        "triggered_rules": result.triggered_rules,
        "message": result.message,
    }
