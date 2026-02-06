# End-to-End Integration Test Suite

## Overview

`test_e2e.py` provides comprehensive integration testing for the entire flux-rag system, covering the full flow from document ingestion through RAG query to safety guards and agent routing.

## Test Structure

### 1. TestDocumentIngestion
Tests the document ingestion pipeline.

**Tests:**
- `test_ingest_single_document`: Verifies single document ingestion with Korean filename
- `test_ingest_all_sample_documents`: Batch ingests all 5 sample documents
- `test_ingest_nonexistent_file`: Error handling for missing files
- `test_ingest_empty_file`: Graceful failure for empty documents

**Key Assertions:**
- Status is "completed" for successful ingests
- Chunk count > 0
- Error messages for failures

### 2. TestRAGQuery
Tests RAG chain query execution with pre-ingested documents.

**Tests:**
- `test_rag_basic_query`: Basic Korean query about gas facility inspection
- `test_rag_sources_returned`: Validates source structure (chunk_id, content, score, metadata)
- `test_direct_mode_no_sources`: Verifies direct mode bypasses retrieval
- `test_rag_confidence_range`: Ensures confidence in [0.0, 1.0]

**Fixture:** `rag_chain_with_docs` ingests all sample documents into temp vectorstore

### 3. TestQualityControl
Tests confidence scoring and safety guards without requiring LLM.

**Tests:**
- `test_high_confidence`: Scores [0.9, 0.85, 0.88] → "high" level
- `test_medium_confidence`: Scores [0.6, 0.55, 0.5] → "medium" level
- `test_low_confidence_warning`: Scores [0.2, 0.1] → safety warning triggered
- `test_empty_scores`: Empty list → confidence = 0.0
- `test_single_high_score`: Single score maintains value
- `test_consistency_penalty`: Variance in scores reduces confidence

### 4. TestQueryRouting
Tests query routing using fallback keyword-based routing (no LLM required).

**Tests:**
- `test_document_search_routing`: "규정", "점검" keywords → DOCUMENT_SEARCH
- `test_chitchat_routing`: "안녕하세요!" → CHITCHAT
- `test_complex_task_routing`: "비교", "분석" keywords → COMPLEX_TASK
- `test_general_query_default`: No signals → GENERAL_QUERY
- `test_routing_confidence`: Validates all results have 0 <= confidence <= 1

**Note:** Tests use `_fallback_route()` method directly to avoid LLM dependency

### 5. TestConversationMemory
Tests SQLite-backed conversation storage.

**Tests:**
- `test_session_lifecycle`: Full CRUD operations
- `test_message_ordering`: Chronological message order
- `test_multiple_sessions`: Session isolation
- `test_session_metadata`: Custom session metadata
- `test_message_metadata`: Custom message metadata (confidence, sources)

**Fixture:** Uses temp database path to avoid polluting real data

### 6. TestMCPTools
Tests MCP tool registry and execution.

**Tests:**
- `test_list_tools`: Verifies calculator in tool list
- `test_calculator_tool`: Execute "2+3" → result = 5
- `test_calculator_complex_expression`: "(10 + 5) * 2" → 30
- `test_unknown_tool`: Non-existent tool → error status
- `test_tool_schemas`: Validates schema format

### 7. TestStreamingFlow
Tests streaming query responses.

**Tests:**
- `test_stream_events_order`: Validates event sequence: start → chunk(s) → end
- `test_stream_with_sources`: RAG mode emits source events

**Event Structure:**
- Start: `{"event": "start", "data": {"message_id": ...}}`
- Chunk: `{"event": "chunk", "data": {"content": token}}`
- End: `{"event": "end", "data": {"confidence_score": ...}}`

### 8. TestSafetyGuards
Tests safety warning generation for low-confidence responses.

**Tests:**
- `test_low_confidence_produces_warning`: Korean safety message with "확실하지 않은"
- `test_high_confidence_no_warning`: High confidence → no warning
- `test_safety_message_content`: Validates warning includes "⚠️", guidance
- `test_metadata_includes_warning_flag`: `has_safety_warning` flag in metadata

### 9. TestIntegrationFlow
Full end-to-end integration tests.

**Tests:**
- `test_full_rag_pipeline`: Complete flow: ingest → query → validate response
- `test_memory_with_routing`: Conversation memory + query routing together

## Running the Tests

### Run all e2e tests:
```bash
cd backend
pytest tests/test_e2e.py -v
```

### Run specific test class:
```bash
pytest tests/test_e2e.py::TestQualityControl -v
```

### Run specific test:
```bash
pytest tests/test_e2e.py::TestMCPTools::test_calculator_tool -v
```

### Run with coverage:
```bash
pytest tests/test_e2e.py --cov=. --cov-report=html
```

## Dependencies

### Required for all tests:
- pytest
- pytest-asyncio
- aiosqlite

### Required for LLM-dependent tests:
- Working embedder (vLLM or configured provider)
- Working LLM (for routing, RAG generation)

### Optional:
Tests gracefully skip when dependencies are unavailable using `pytest.skip()`

## Sample Documents

Tests use documents from `backend/data/sample/`:
- 가스안전관리_규정_2024.txt
- 설비점검_매뉴얼_v3.txt
- 비상대응_절차서.txt
- 월간_안전점검_보고서_202401.txt
- 교육훈련_계획서_2024.txt

## Test Isolation

- Uses `tmp_path` fixtures for temporary storage
- Each test class has independent fixtures
- Database tests use separate temp DB files
- Vectorstore tests use temp directories

## Coverage

The test suite covers:
- ✅ Document ingestion (success, failure, edge cases)
- ✅ RAG query execution (RAG mode, direct mode)
- ✅ Source retrieval and formatting
- ✅ Confidence scoring and quality control
- ✅ Safety guards and warnings
- ✅ Query routing (fallback keyword-based)
- ✅ Conversation memory (CRUD, ordering, metadata)
- ✅ MCP tool registry and execution
- ✅ Streaming response flow
- ✅ Full integration scenarios

## Notes

1. **LLM Dependencies**: Tests requiring LLM/embeddings use `pytest.skip()` when unavailable
2. **Korean Language**: Tests include Korean queries to match production use cases
3. **Temp Storage**: All tests use temporary storage to avoid side effects
4. **Async**: All I/O tests use `@pytest.mark.asyncio`
5. **Graceful Degradation**: Tests that can't run due to missing dependencies skip rather than fail
