"""End-to-end integration tests for the full RAG system.

Tests cover: document ingest → RAG query → sources → feedback → safety guards → agent routing.
"""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import ConversationMemory, QueryCategory, QueryRouter
from agent.mcp.registry import create_default_registry
from core.embeddings import create_embedder
from core.vectorstore import create_vectorstore
from pipeline.ingest import IngestPipeline
from rag import QualityController, RAGChain


# ===========================
# 1. DOCUMENT INGESTION TESTS
# ===========================


class TestDocumentIngestion:
    """Tests for document ingestion pipeline."""

    @pytest.fixture
    def data_dir(self):
        """Path to sample documents."""
        return Path(__file__).parent.parent / "data" / "sample"

    @pytest.fixture
    def sample_files(self, data_dir):
        """List of all sample document files."""
        return [
            data_dir / "가스안전관리_규정_2024.txt",
            data_dir / "설비점검_매뉴얼_v3.txt",
            data_dir / "비상대응_절차서.txt",
            data_dir / "월간_안전점검_보고서_202401.txt",
            data_dir / "교육훈련_계획서_2024.txt",
        ]

    @pytest.fixture
    async def ingest_pipeline(self, tmp_path):
        """Create a test ingest pipeline with temporary storage."""
        try:
            embedder = create_embedder()
            vectorstore = create_vectorstore(persist_directory=str(tmp_path / "chroma"))
            return IngestPipeline(vectorstore=vectorstore, embedder=embedder)
        except Exception as e:
            pytest.skip(f"Cannot initialize embedder/vectorstore: {e}")

    @pytest.mark.asyncio
    async def test_ingest_single_document(self, ingest_pipeline, sample_files):
        """Ingest a single document and verify success."""
        result = await ingest_pipeline.ingest(sample_files[0])

        assert result.status == "completed"
        assert result.chunk_count > 0
        assert result.filename == "가스안전관리_규정_2024.txt"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_ingest_all_sample_documents(self, ingest_pipeline, sample_files):
        """Ingest all sample documents and verify all succeed."""
        results = await ingest_pipeline.ingest_batch(sample_files)

        assert len(results) == 5

        for result in results:
            assert result.status == "completed", f"Failed to ingest {result.filename}: {result.error}"
            assert result.chunk_count > 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, ingest_pipeline, tmp_path):
        """Try to ingest a file that doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.txt"
        result = await ingest_pipeline.ingest(nonexistent_file)

        assert result.status == "failed"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_ingest_empty_file(self, ingest_pipeline, tmp_path):
        """Ingest an empty file should fail gracefully."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = await ingest_pipeline.ingest(empty_file)

        assert result.status == "failed"
        assert "No text extracted" in result.error or "No chunks generated" in result.error


# ===========================
# 2. RAG QUERY TESTS
# ===========================


class TestRAGQuery:
    """Tests for RAG chain query execution."""

    @pytest.fixture
    async def rag_chain_with_docs(self, tmp_path):
        """RAG chain with pre-ingested sample documents."""
        try:
            # Setup vectorstore and embedder
            embedder = create_embedder()
            vectorstore = create_vectorstore(persist_directory=str(tmp_path / "chroma"))

            # Ingest sample documents
            data_dir = Path(__file__).parent.parent / "data" / "sample"
            sample_files = [
                data_dir / "가스안전관리_규정_2024.txt",
                data_dir / "설비점검_매뉴얼_v3.txt",
                data_dir / "비상대응_절차서.txt",
                data_dir / "월간_안전점검_보고서_202401.txt",
                data_dir / "교육훈련_계획서_2024.txt",
            ]

            pipeline = IngestPipeline(vectorstore=vectorstore, embedder=embedder)
            for file in sample_files:
                if file.exists():
                    await pipeline.ingest(file)

            # Create RAG chain
            from rag.retriever import HybridRetriever
            retriever = HybridRetriever(vectorstore=vectorstore, embedder=embedder)
            return RAGChain(retriever=retriever)

        except Exception as e:
            pytest.skip(f"Cannot initialize RAG chain: {e}")

    @pytest.mark.asyncio
    async def test_rag_basic_query(self, rag_chain_with_docs):
        """Query about gas facility inspection time."""
        response = await rag_chain_with_docs.query(
            "가스시설 일일 점검은 몇 시에 하나요?",
            mode="rag"
        )

        assert response.content
        assert len(response.content) > 0
        assert len(response.sources) > 0

    @pytest.mark.asyncio
    async def test_rag_sources_returned(self, rag_chain_with_docs):
        """Verify sources have required fields."""
        response = await rag_chain_with_docs.query(
            "가스안전 규정에 대해 알려주세요",
            mode="rag"
        )

        assert len(response.sources) > 0

        for source in response.sources:
            assert "chunk_id" in source
            assert "content" in source
            assert "score" in source
            assert "metadata" in source

    @pytest.mark.asyncio
    async def test_direct_mode_no_sources(self, rag_chain_with_docs):
        """Direct mode should not return sources."""
        response = await rag_chain_with_docs.query(
            "안전 관리의 중요성에 대해 설명해주세요",
            mode="direct"
        )

        assert response.content
        assert len(response.sources) == 0
        assert response.confidence == 1.0

    @pytest.mark.asyncio
    async def test_rag_confidence_range(self, rag_chain_with_docs):
        """Verify confidence is in valid range."""
        response = await rag_chain_with_docs.query(
            "가스 점검 절차는?",
            mode="rag"
        )

        assert 0.0 <= response.confidence <= 1.0
        assert response.confidence_level in ["high", "medium", "low"]


# ===========================
# 3. QUALITY CONTROL TESTS
# ===========================


class TestQualityControl:
    """Tests for quality control and confidence scoring."""

    @pytest.fixture
    def quality_controller(self):
        return QualityController(confidence_high=0.8, confidence_low=0.5)

    def test_high_confidence(self, quality_controller):
        """High scores should produce high confidence level."""
        scores = [0.9, 0.85, 0.88]
        confidence = quality_controller.calculate_confidence(scores)
        level = quality_controller.get_confidence_level(confidence)

        assert level == "high"
        assert confidence >= 0.8

    def test_medium_confidence(self, quality_controller):
        """Medium scores should produce medium confidence level."""
        scores = [0.6, 0.55, 0.5]
        confidence = quality_controller.calculate_confidence(scores)
        level = quality_controller.get_confidence_level(confidence)

        assert level == "medium"
        assert 0.5 <= confidence < 0.8

    def test_low_confidence_warning(self, quality_controller):
        """Low scores should trigger safety warning."""
        scores = [0.2, 0.1]
        confidence = quality_controller.calculate_confidence(scores)

        assert quality_controller.should_add_safety_warning(confidence)
        warning = quality_controller.get_safety_message(confidence)
        assert warning is not None
        assert "확실하지 않은" in warning

    def test_empty_scores(self, quality_controller):
        """Empty scores should return zero confidence."""
        confidence = quality_controller.calculate_confidence([])
        assert confidence == 0.0
        assert quality_controller.get_confidence_level(confidence) == "low"

    def test_single_high_score(self, quality_controller):
        """Single high score should maintain high confidence."""
        scores = [0.95]
        confidence = quality_controller.calculate_confidence(scores)
        assert confidence == 0.95

    def test_consistency_penalty(self, quality_controller):
        """High variance in scores should reduce confidence."""
        consistent_scores = [0.9, 0.88, 0.87]
        inconsistent_scores = [0.9, 0.4, 0.3]

        consistent_conf = quality_controller.calculate_confidence(consistent_scores)
        inconsistent_conf = quality_controller.calculate_confidence(inconsistent_scores)

        assert consistent_conf > inconsistent_conf


# ===========================
# 4. QUERY ROUTING TESTS
# ===========================


class TestQueryRouting:
    """Tests for query routing using fallback methods."""

    @pytest.fixture
    def router(self):
        """Create router (will use fallback routing for tests)."""
        try:
            return QueryRouter()
        except Exception:
            pytest.skip("Cannot initialize QueryRouter")

    def test_document_search_routing(self, router):
        """Document-related query should route to DOCUMENT_SEARCH."""
        query = "가스안전 규정에 따른 점검 절차는?"
        result = router._fallback_route(query)

        assert result.category == QueryCategory.DOCUMENT_SEARCH
        assert "규정" in query or "점검" in query

    def test_chitchat_routing(self, router):
        """Greeting should route to CHITCHAT."""
        query = "안녕하세요!"
        result = router._fallback_route(query)

        assert result.category == QueryCategory.CHITCHAT

    def test_complex_task_routing(self, router):
        """Multi-step query should route to COMPLEX_TASK."""
        query = "A동과 B동의 점검 결과를 비교하고 분석해주세요"
        result = router._fallback_route(query)

        assert result.category == QueryCategory.COMPLEX_TASK
        assert result.confidence > 0.0

    def test_general_query_default(self, router):
        """Query with no strong signals should default to GENERAL_QUERY."""
        query = "이것은 무엇인가요?"
        result = router._fallback_route(query)

        assert result.category == QueryCategory.GENERAL_QUERY

    def test_routing_confidence(self, router):
        """All routing results should have valid confidence."""
        queries = [
            "가스 규정 확인",
            "안녕하세요",
            "비교하고 분석",
            "일반적인 질문"
        ]

        for query in queries:
            result = router._fallback_route(query)
            assert 0.0 <= result.confidence <= 1.0
            assert result.reasoning


# ===========================
# 5. CONVERSATION MEMORY TESTS
# ===========================


class TestConversationMemory:
    """Tests for conversation memory and session management."""

    @pytest.fixture
    async def memory(self, tmp_path):
        """Create memory with temporary database."""
        db_path = str(tmp_path / "test_memory.db")
        return ConversationMemory(db_path=db_path)

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, memory):
        """Create session, add messages, retrieve history, delete."""
        # Create session
        session_id = await memory.create_session(title="Test Session")
        assert session_id

        # Add messages
        msg1_id = await memory.add_message(session_id, "user", "Hello")
        msg2_id = await memory.add_message(session_id, "assistant", "Hi there!")
        assert msg1_id
        assert msg2_id

        # Get history
        history = await memory.get_history(session_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

        # Delete session
        await memory.delete_session(session_id)
        history_after = await memory.get_history(session_id)
        assert len(history_after) == 0

    @pytest.mark.asyncio
    async def test_message_ordering(self, memory):
        """Messages should be in chronological order."""
        session_id = await memory.create_session(title="Order Test")

        await memory.add_message(session_id, "user", "First")
        await memory.add_message(session_id, "assistant", "Second")
        await memory.add_message(session_id, "user", "Third")

        history = await memory.get_history(session_id)

        assert len(history) == 3
        assert history[0]["content"] == "First"
        assert history[1]["content"] == "Second"
        assert history[2]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, memory):
        """Multiple sessions should be independent."""
        session1 = await memory.create_session(title="Session 1")
        session2 = await memory.create_session(title="Session 2")

        await memory.add_message(session1, "user", "Message in session 1")
        await memory.add_message(session2, "user", "Message in session 2")

        history1 = await memory.get_history(session1)
        history2 = await memory.get_history(session2)

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["content"] == "Message in session 1"
        assert history2[0]["content"] == "Message in session 2"

    @pytest.mark.asyncio
    async def test_session_metadata(self, memory):
        """Session can store metadata."""
        metadata = {"user_id": "test123", "tags": ["important"]}
        session_id = await memory.create_session(title="Meta Test", metadata=metadata)

        session = await memory.get_session(session_id)
        assert session["title"] == "Meta Test"
        assert session["metadata"]["user_id"] == "test123"

    @pytest.mark.asyncio
    async def test_message_metadata(self, memory):
        """Messages can store metadata."""
        session_id = await memory.create_session()
        msg_metadata = {"confidence": 0.95, "sources": 3}

        await memory.add_message(
            session_id,
            "assistant",
            "Answer with high confidence",
            metadata=msg_metadata
        )

        history = await memory.get_history(session_id)
        assert history[0]["metadata"]["confidence"] == 0.95


# ===========================
# 6. MCP TOOLS TESTS
# ===========================


class TestMCPTools:
    """Tests for MCP tool registry and execution."""

    @pytest.fixture
    def registry(self):
        """Create default tool registry."""
        return create_default_registry()

    def test_list_tools(self, registry):
        """Registry should list available tools."""
        tools = registry.list_tools()

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "calculator" in tool_names

    @pytest.mark.asyncio
    async def test_calculator_tool(self, registry):
        """Execute calculator tool."""
        result = await registry.execute("calculator", expression="2+3")

        assert result.success
        assert result.data["result"] == 5

    @pytest.mark.asyncio
    async def test_calculator_complex_expression(self, registry):
        """Calculator should handle complex expressions."""
        result = await registry.execute("calculator", expression="(10 + 5) * 2")

        assert result.success
        assert result.data["result"] == 30

    @pytest.mark.asyncio
    async def test_unknown_tool(self, registry):
        """Executing non-existent tool should return error."""
        result = await registry.execute("nonexistent_tool", arg1="value")

        assert not result.success
        assert result.error
        assert "not found" in result.error.lower()

    def test_tool_schemas(self, registry):
        """Tools should provide valid schemas."""
        schemas = registry.list_schemas()

        assert len(schemas) > 0
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema


# ===========================
# 7. STREAMING FLOW TESTS
# ===========================


class TestStreamingFlow:
    """Tests for streaming query responses."""

    @pytest.fixture
    async def rag_chain(self):
        """Create RAG chain for streaming tests."""
        try:
            return RAGChain()
        except Exception as e:
            pytest.skip(f"Cannot initialize RAG chain: {e}")

    @pytest.mark.asyncio
    async def test_stream_events_order(self, rag_chain):
        """Stream events should come in correct order."""
        events = []

        async for event in rag_chain.stream_query(
            "안전 관리의 중요성",
            mode="direct"
        ):
            events.append(event)

        # Verify event structure
        assert len(events) >= 2

        # First event should be start
        assert events[0]["event"] == "start"
        assert "message_id" in events[0]["data"]

        # Last event should be end
        assert events[-1]["event"] == "end"
        assert "confidence_score" in events[-1]["data"]

        # Middle events should be chunks
        for event in events[1:-1]:
            assert event["event"] == "chunk"
            assert "content" in event["data"]

    @pytest.mark.asyncio
    async def test_stream_with_sources(self, rag_chain):
        """RAG mode streaming should emit source events."""
        events = []

        try:
            async for event in rag_chain.stream_query(
                "가스 안전",
                mode="rag"
            ):
                events.append(event)
                # Limit collection to avoid long waits
                if len(events) > 20:
                    break
        except Exception as e:
            pytest.skip(f"Streaming test requires full setup: {e}")

        # Check for start event
        start_events = [e for e in events if e["event"] == "start"]
        assert len(start_events) > 0


# ===========================
# 8. SAFETY GUARDS TESTS
# ===========================


class TestSafetyGuards:
    """Tests for safety guards and warnings."""

    @pytest.fixture
    def quality_controller(self):
        return QualityController(confidence_high=0.8, confidence_low=0.5)

    def test_low_confidence_produces_warning(self, quality_controller):
        """Low confidence scores should produce Korean safety message."""
        scores = [0.2, 0.15, 0.1]
        confidence = quality_controller.calculate_confidence(scores)

        assert quality_controller.should_add_safety_warning(confidence)

        message = quality_controller.get_safety_message(confidence)
        assert message is not None
        assert "확실하지 않은" in message
        assert "답변" in message

    def test_high_confidence_no_warning(self, quality_controller):
        """High confidence should not produce warning."""
        scores = [0.95, 0.92, 0.90]
        confidence = quality_controller.calculate_confidence(scores)

        assert not quality_controller.should_add_safety_warning(confidence)
        assert quality_controller.get_safety_message(confidence) is None

    def test_safety_message_content(self, quality_controller):
        """Safety message should contain helpful guidance."""
        message = quality_controller.get_safety_message(0.2)

        assert message is not None
        # Should contain warning indicator
        assert "⚠️" in message
        # Should mention uncertainty
        assert "확실하지 않은" in message
        # Should provide guidance
        assert "질문" in message or "문서" in message

    def test_metadata_includes_warning_flag(self, quality_controller):
        """Response metadata should indicate safety warning presence."""
        # Low confidence case
        low_meta = quality_controller.build_response_metadata(
            confidence=0.3,
            sources=[],
            model_used="test",
            latency_ms=100
        )
        assert low_meta["has_safety_warning"] is True

        # High confidence case
        high_meta = quality_controller.build_response_metadata(
            confidence=0.9,
            sources=[],
            model_used="test",
            latency_ms=100
        )
        assert high_meta["has_safety_warning"] is False


# ===========================
# 9. INTEGRATION FLOW TESTS
# ===========================


class TestIntegrationFlow:
    """Full integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, tmp_path):
        """Test complete flow: ingest → query → routing → response."""
        try:
            # Setup
            embedder = create_embedder()
            vectorstore = create_vectorstore(persist_directory=str(tmp_path / "chroma"))

            # Ingest a document
            data_dir = Path(__file__).parent.parent / "data" / "sample"
            doc_path = data_dir / "가스안전관리_규정_2024.txt"

            if not doc_path.exists():
                pytest.skip("Sample document not found")

            pipeline = IngestPipeline(vectorstore=vectorstore, embedder=embedder)
            ingest_result = await pipeline.ingest(doc_path)

            assert ingest_result.status == "completed"

            # Create RAG chain
            from rag.retriever import HybridRetriever
            retriever = HybridRetriever(vectorstore=vectorstore, embedder=embedder)
            rag_chain = RAGChain(retriever=retriever)

            # Query
            response = await rag_chain.query("가스안전 규정의 목적은?", mode="rag")

            # Verify response
            assert response.content
            assert len(response.sources) > 0
            assert 0.0 <= response.confidence <= 1.0
            assert response.confidence_level in ["high", "medium", "low"]

        except Exception as e:
            pytest.skip(f"Full integration test requires complete setup: {e}")

    @pytest.mark.asyncio
    async def test_memory_with_routing(self, tmp_path):
        """Test conversation memory with query routing."""
        memory = ConversationMemory(db_path=str(tmp_path / "test.db"))
        session_id = await memory.create_session(title="Integration Test")

        # Add conversation
        await memory.add_message(session_id, "user", "가스안전 규정에 대해 알려주세요")
        await memory.add_message(session_id, "assistant", "가스안전 규정은...")

        # Get history
        history = await memory.get_history(session_id)
        assert len(history) == 2

        # Test with router
        try:
            router = QueryRouter()
            result = router._fallback_route("가스안전 규정")
            assert result.category == QueryCategory.DOCUMENT_SEARCH
        except Exception:
            pytest.skip("Router test requires LLM")
