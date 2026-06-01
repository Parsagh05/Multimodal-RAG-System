"""Pydantic models for the /chat API contract."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the chat request."""

    type: str = Field(
        ...,
        description="Message type: 'text' or 'image'.",
        pattern="^(text|image)$",
    )
    content: str = Field(
        ...,
        description="Text content or base64-encoded image string.",
    )


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    chat_id: str = Field(
        ...,
        description="Unique conversation session identifier.",
    )
    messages: list[Message] = Field(
        ...,
        description="List of user messages (text and/or image).",
        min_length=1,
    )


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    message: Optional[str] = Field(
        None,
        description="Agent's text response to display to the user.",
    )
    base_random_keys: Optional[list[str]] = Field(
        None,
        description="List of Base Product random_keys suggested (max 10).",
        max_length=10,
    )
    member_random_keys: Optional[list[str]] = Field(
        None,
        description="List of Member Product random_keys suggested (max 10).",
        max_length=10,
    )
