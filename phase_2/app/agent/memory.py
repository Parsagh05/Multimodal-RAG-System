"""Short-Term Memory Manager for the Agent.

Implements a sliding window memory that maintains conversation context
within a session. Each chat_id gets its own memory store.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry (message) in the conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationState:
    """State for a single conversation session."""

    chat_id: str
    messages: list[MemoryEntry] = field(default_factory=list)
    summary: Optional[str] = None
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)  # Extra session context


class ShortTermMemory:
    """Manages short-term memory across multiple chat sessions.

    Uses a sliding window approach: keeps the last N messages in full,
    and optionally summarizes older messages.
    """

    def __init__(self, max_messages: int = 20, summary_threshold: int = 15):
        """Initialize memory manager.

        Args:
            max_messages: Maximum messages to keep in sliding window.
            summary_threshold: Summarize messages beyond this count.
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create_session(self, chat_id: str) -> ConversationState:
        """Get existing session or create a new one.

        Args:
            chat_id: Unique conversation identifier.

        Returns:
            The conversation state for this chat_id.
        """
        if chat_id not in self._sessions:
            self._sessions[chat_id] = ConversationState(chat_id=chat_id)
            logger.info(f"Created new session: {chat_id}")
        return self._sessions[chat_id]

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a message to the conversation history.

        Args:
            chat_id: Session identifier.
            role: Message role (user/assistant/system/tool).
            content: Message content.
            metadata: Optional metadata dict.
        """
        session = self.get_or_create_session(chat_id)
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.messages.append(entry)
        session.last_active = time.time()
        session.turn_count += 1

        # Apply sliding window
        if len(session.messages) > self.max_messages:
            self._trim_messages(session)

        logger.debug(
            f"Session {chat_id}: added {role} message "
            f"(total: {len(session.messages)})"
        )

    def get_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> list[dict[str, str]]:
        """Get conversation history formatted for LLM consumption.

        Args:
            chat_id: Session identifier.
            max_messages: Override for max messages to return.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        session = self.get_or_create_session(chat_id)
        limit = max_messages or self.max_messages
        messages = session.messages[-limit:]

        history = []
        # Include summary if available
        if session.summary:
            history.append({
                "role": "system",
                "content": f"[Conversation Summary]: {session.summary}",
            })

        for entry in messages:
            history.append({"role": entry.role, "content": entry.content})

        return history

    def get_context(self, chat_id: str) -> dict:
        """Get session context (stored preferences, facts, etc.).

        Args:
            chat_id: Session identifier.

        Returns:
            Context dictionary.
        """
        session = self.get_or_create_session(chat_id)
        return session.context

    def update_context(self, chat_id: str, key: str, value: Any) -> None:
        """Update session context with a key-value pair.

        Args:
            chat_id: Session identifier.
            key: Context key.
            value: Context value.
        """
        session = self.get_or_create_session(chat_id)
        session.context[key] = value
        logger.debug(f"Session {chat_id}: context updated [{key}]")

    def clear_session(self, chat_id: str) -> None:
        """Clear a specific session's memory."""
        if chat_id in self._sessions:
            del self._sessions[chat_id]
            logger.info(f"Cleared session: {chat_id}")

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())

    def _trim_messages(self, session: ConversationState) -> None:
        """Trim messages using sliding window strategy.

        Keeps only the most recent messages within the window size.
        Older messages are discarded (summary generation is handled
        by the agent if needed).
        """
        overflow = len(session.messages) - self.max_messages
        if overflow > 0:
            # Keep only the last max_messages
            session.messages = session.messages[-self.max_messages:]
            logger.debug(
                f"Session {session.chat_id}: trimmed {overflow} old messages"
            )

    def set_summary(self, chat_id: str, summary: str) -> None:
        """Set a conversation summary for the session.

        Args:
            chat_id: Session identifier.
            summary: Summary text of earlier conversation.
        """
        session = self.get_or_create_session(chat_id)
        session.summary = summary
        logger.debug(f"Session {chat_id}: summary updated")

    def needs_summary(self, chat_id: str) -> bool:
        """Check if the session has enough messages to warrant summarization.

        Args:
            chat_id: Session identifier.

        Returns:
            True if summarization is recommended.
        """
        session = self.get_or_create_session(chat_id)
        return (
            len(session.messages) >= self.summary_threshold
            and session.summary is None
        )
