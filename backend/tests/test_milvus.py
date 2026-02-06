"""Tests for Milvus vector store implementation."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

pymilvus = pytest.importorskip("pymilvus", reason="pymilvus not installed")

from core.vectorstore import create_vectorstore
from core.vectorstore.base import SearchResult
from core.vectorstore.milvus import MilvusStore


@pytest.fixture
def mock_milvus():
    """Mock pymilvus components."""
    with patch("core.vectorstore.milvus.connections") as mock_conn, \
         patch("core.vectorstore.milvus.utility") as mock_util, \
         patch("core.vectorstore.milvus.Collection") as mock_coll:

        # Mock utility.has_collection to return False (create new collection)
        mock_util.has_collection.return_value = False

        # Mock collection instance
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_coll.return_value = mock_collection

        yield {
            "connections": mock_conn,
            "utility": mock_util,
            "Collection": mock_coll,
            "collection_instance": mock_collection,
        }


@pytest.mark.asyncio
async def test_milvus_store_creation(mock_milvus):
    """Test MilvusStore initialization and collection creation."""
    store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="test_collection",
        dimension=1024,
    )

    # Verify connection was established
    mock_milvus["connections"].connect.assert_called_once_with(
        alias="default",
        host="localhost",
        port=19530,
    )

    # Verify collection was created
    assert mock_milvus["Collection"].called
    assert store.collection_name == "test_collection"
    assert store.dimension == 1024


@pytest.mark.asyncio
async def test_milvus_add_documents(mock_milvus):
    """Test adding documents to Milvus."""
    store = MilvusStore(collection_name="test_collection")

    ids = ["doc1", "doc2"]
    texts = ["Hello world", "Test document"]
    embeddings = [[0.1] * 1024, [0.2] * 1024]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    await store.add(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Verify insert was called
    assert mock_milvus["collection_instance"].insert.called

    # Verify flush was called
    assert mock_milvus["collection_instance"].flush.called


@pytest.mark.asyncio
async def test_milvus_search(mock_milvus):
    """Test searching in Milvus."""
    store = MilvusStore(collection_name="test_collection")

    # Mock search results
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda key: {
        "id": "doc1",
        "content": "Test content",
        "metadata_json": json.dumps({"source": "test"}),
    }.get(key)
    mock_hit.score = 0.95

    mock_milvus["collection_instance"].search.return_value = [[mock_hit]]

    query_embedding = [0.1] * 1024
    results = await store.search(query_embedding, top_k=5)

    # Verify search was called
    assert mock_milvus["collection_instance"].search.called

    # Verify results
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].id == "doc1"
    assert results[0].content == "Test content"
    assert results[0].score == 0.95
    assert results[0].metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_milvus_delete(mock_milvus):
    """Test deleting documents from Milvus."""
    store = MilvusStore(collection_name="test_collection")

    ids = ["doc1", "doc2"]
    await store.delete(ids)

    # Verify delete was called
    assert mock_milvus["collection_instance"].delete.called
    assert mock_milvus["collection_instance"].flush.called


@pytest.mark.asyncio
async def test_milvus_get(mock_milvus):
    """Test getting documents by IDs."""
    store = MilvusStore(collection_name="test_collection")

    # Mock query results
    mock_milvus["collection_instance"].query.return_value = [
        {
            "id": "doc1",
            "content": "Test content",
            "metadata_json": json.dumps({"source": "test"}),
        }
    ]

    ids = ["doc1"]
    results = await store.get(ids)

    # Verify query was called
    assert mock_milvus["collection_instance"].query.called

    # Verify results
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["content"] == "Test content"
    assert results[0]["metadata"] == {"source": "test"}


@pytest.mark.asyncio
async def test_milvus_count(mock_milvus):
    """Test getting document count."""
    store = MilvusStore(collection_name="test_collection")

    mock_milvus["collection_instance"].num_entities = 42
    count = await store.count()

    assert count == 42


@pytest.mark.asyncio
async def test_milvus_clear(mock_milvus):
    """Test clearing collection."""
    store = MilvusStore(collection_name="test_collection")

    await store.clear()

    # Verify collection was dropped
    assert mock_milvus["utility"].drop_collection.called


@pytest.mark.asyncio
async def test_metadata_serialization(mock_milvus):
    """Test metadata JSON serialization/deserialization."""
    store = MilvusStore(collection_name="test_collection")

    # Test with complex metadata
    metadata = {
        "source": "test.pdf",
        "page": 42,
        "tags": ["important", "review"],
        "nested": {"key": "value"},
    }

    ids = ["doc1"]
    texts = ["Test"]
    embeddings = [[0.1] * 1024]
    metadatas = [metadata]

    await store.add(ids, texts, embeddings, metadatas)

    # Verify insert was called and metadata was serialized
    insert_call_args = mock_milvus["collection_instance"].insert.call_args
    entities = insert_call_args[0][0]
    metadata_json = entities[3][0]  # Fourth field is metadata_json

    # Verify it's valid JSON and matches original
    assert json.loads(metadata_json) == metadata


def test_filter_expression_building(mock_milvus):
    """Test Milvus filter expression building."""
    store = MilvusStore(collection_name="test_collection")

    # Test single string filter
    expr = store._build_filter_expr({"source": "test.pdf"})
    assert expr == 'source == "test.pdf"'

    # Test numeric filter
    expr = store._build_filter_expr({"page": 42})
    assert expr == "page == 42"

    # Test list filter
    expr = store._build_filter_expr({"tag": ["important", "review"]})
    assert "in" in expr.lower()

    # Test multiple filters
    expr = store._build_filter_expr({"source": "test.pdf", "page": 42})
    assert "source" in expr and "page" in expr and "and" in expr


@patch("core.vectorstore.milvus.connections")
@patch("core.vectorstore.milvus.utility")
@patch("core.vectorstore.milvus.Collection")
def test_create_vectorstore_factory_milvus(mock_coll, mock_util, mock_conn):
    """Test factory function with store_type='milvus'."""
    mock_util.has_collection.return_value = False
    mock_collection = MagicMock()
    mock_coll.return_value = mock_collection

    store = create_vectorstore(
        store_type="milvus",
        collection_name="test_milvus",
        host="localhost",
        port=19530,
    )

    assert isinstance(store, MilvusStore)
    assert store.collection_name == "test_milvus"
    assert store.host == "localhost"
    assert store.port == 19530


def test_get_collection_info(mock_milvus):
    """Test getting collection info."""
    store = MilvusStore(
        collection_name="test_collection",
        dimension=1024,
        index_type="IVF_FLAT",
        metric_type="COSINE",
    )

    mock_milvus["collection_instance"].num_entities = 100

    info = store.get_collection_info()

    assert info["name"] == "test_collection"
    assert info["dimension"] == 1024
    assert info["index_type"] == "IVF_FLAT"
    assert info["metric_type"] == "COSINE"
    assert info["count"] == 100
