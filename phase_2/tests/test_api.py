"""Tests for the Phase 2 Agent API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing."""
    return {
        "chat_id": "test-session-001",
        "messages": [
            {"type": "text", "content": "Hello, I want to find a laptop."}
        ],
    }


@pytest.fixture
def sample_image_request():
    """Sample request with an image message."""
    return {
        "chat_id": "test-session-002",
        "messages": [
            {"type": "text", "content": "Find a product similar to this image."},
            {"type": "image", "content": "data:image/jpeg;base64,/9j/4AAQ..."},
        ],
    }


@pytest.fixture
def multi_turn_requests():
    """Multi-turn conversation requests for testing memory."""
    return [
        {
            "chat_id": "test-multi-001",
            "messages": [
                {"type": "text", "content": "I'm looking for a desk for daily work."}
            ],
        },
        {
            "chat_id": "test-multi-001",
            "messages": [
                {"type": "text", "content": "My budget is around 5 million tomans."}
            ],
        },
        {
            "chat_id": "test-multi-001",
            "messages": [
                {"type": "text", "content": "I prefer wooden material."}
            ],
        },
    ]


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_valid_text_message(self):
        from app.models.schemas import ChatRequest

        request = ChatRequest(
            chat_id="abc-123",
            messages=[{"type": "text", "content": "Hello"}],
        )
        assert request.chat_id == "abc-123"
        assert len(request.messages) == 1
        assert request.messages[0].type == "text"

    def test_valid_image_message(self):
        from app.models.schemas import ChatRequest

        request = ChatRequest(
            chat_id="abc-123",
            messages=[{"type": "image", "content": "base64data..."}],
        )
        assert request.messages[0].type == "image"

    def test_response_model(self):
        from app.models.schemas import ChatResponse

        response = ChatResponse(
            message="Here is a product.",
            base_random_keys=["key1", "key2"],
            member_random_keys=None,
        )
        assert response.message == "Here is a product."
        assert len(response.base_random_keys) == 2
        assert response.member_random_keys is None

    def test_response_null_fields(self):
        from app.models.schemas import ChatResponse

        response = ChatResponse(
            message=None,
            base_random_keys=None,
            member_random_keys=None,
        )
        assert response.message is None


class TestMemory:
    """Test short-term memory manager."""

    def test_create_session(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=10)
        session = memory.get_or_create_session("test-1")
        assert session.chat_id == "test-1"
        assert len(session.messages) == 0

    def test_add_message(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=10)
        memory.add_message("test-1", "user", "Hello")
        memory.add_message("test-1", "assistant", "Hi there!")

        history = memory.get_history("test-1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_sliding_window(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=5)
        for i in range(10):
            memory.add_message("test-1", "user", f"Message {i}")

        history = memory.get_history("test-1")
        assert len(history) <= 5
        # Should have the most recent messages
        assert "Message 9" in history[-1]["content"]

    def test_session_isolation(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=10)
        memory.add_message("session-a", "user", "Hello A")
        memory.add_message("session-b", "user", "Hello B")

        history_a = memory.get_history("session-a")
        history_b = memory.get_history("session-b")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["content"] == "Hello A"
        assert history_b[0]["content"] == "Hello B"

    def test_clear_session(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=10)
        memory.add_message("test-1", "user", "Hello")
        memory.clear_session("test-1")

        sessions = memory.get_active_sessions()
        assert "test-1" not in sessions

    def test_context_update(self):
        from app.agent.memory import ShortTermMemory

        memory = ShortTermMemory(max_messages=10)
        memory.update_context("test-1", "budget", "5M")

        context = memory.get_context("test-1")
        assert context["budget"] == "5M"
