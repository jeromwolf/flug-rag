"""Tests for golden data management."""
import asyncio
import tempfile
from pathlib import Path

import pytest

from rag.golden_data import GoldenDataEntry, GoldenDataManager


@pytest.fixture
async def manager():
    """Create a temporary GoldenDataManager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_golden.db"
        mgr = GoldenDataManager(db_path=db_path)
        await mgr._ensure_initialized()
        yield mgr


@pytest.mark.asyncio
async def test_add_golden_data(manager):
    """Test adding golden data entry."""
    entry = await manager.add(
        question="한국가스공사의 주요 업무는?",
        answer="천연가스의 도입, 저장, 공급 및 관련 사업",
        category="factual",
        evaluation_tag="accurate",
        created_by="test_admin",
    )

    assert entry.id
    assert entry.question == "한국가스공사의 주요 업무는?"
    assert entry.answer == "천연가스의 도입, 저장, 공급 및 관련 사업"
    assert entry.category == "factual"
    assert entry.evaluation_tag == "accurate"
    assert entry.created_by == "test_admin"
    assert entry.is_active is True


@pytest.mark.asyncio
async def test_list_golden_data(manager):
    """Test listing golden data entries."""
    # Add multiple entries
    await manager.add(
        question="질문1",
        answer="답변1",
        category="factual",
        created_by="user1",
    )
    await manager.add(
        question="질문2",
        answer="답변2",
        category="procedure",
        created_by="user2",
    )

    # List all entries
    entries = await manager.list_entries(limit=10)
    assert len(entries) == 2

    # Filter by category
    factual_entries = await manager.list_entries(category="factual")
    assert len(factual_entries) == 1
    assert factual_entries[0].category == "factual"


@pytest.mark.asyncio
async def test_update_golden_data(manager):
    """Test updating golden data entry."""
    # Add entry
    entry = await manager.add(
        question="원래 질문",
        answer="원래 답변",
        category="factual",
        created_by="admin",
    )

    # Update entry
    updated = await manager.update(
        entry.id,
        answer="수정된 답변",
        category="procedure",
    )

    assert updated.id == entry.id
    assert updated.question == "원래 질문"
    assert updated.answer == "수정된 답변"
    assert updated.category == "procedure"


@pytest.mark.asyncio
async def test_update_nonexistent_entry(manager):
    """Test updating non-existent entry raises error."""
    with pytest.raises(ValueError, match="Golden data not found"):
        await manager.update("nonexistent-id", answer="새 답변")


@pytest.mark.asyncio
async def test_deactivate_entry(manager):
    """Test deactivating entry."""
    # Add entry
    entry = await manager.add(
        question="질문",
        answer="답변",
        created_by="admin",
    )

    # Deactivate
    await manager.update(entry.id, is_active=False)

    # Should not appear in list
    entries = await manager.list_entries()
    assert len(entries) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
