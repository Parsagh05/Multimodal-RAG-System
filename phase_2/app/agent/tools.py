"""Agent Tools: Defines callable tools for the LangGraph agent.

Each tool wraps a Phase 1 RAG capability and provides structured
input/output for the agent's decision-making process.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from ..rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

# Global retriever instance (initialized once)
_retriever: Optional[RAGRetriever] = None


def get_retriever(config_path: str = "config.yaml") -> RAGRetriever:
    """Get or create the global RAG retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(config_path=config_path)
        _retriever.initialize()
    return _retriever


def set_retriever(retriever: RAGRetriever) -> None:
    """Set the global retriever (useful for testing/injection)."""
    global _retriever
    _retriever = retriever


@tool
def search_products_by_text(query: str, top_k: int = 10) -> str:
    """Search for products using a text query.

    Use this tool when the user asks about products, features, or needs
    product recommendations. Returns relevant products from the catalog
    with their details and similarity scores.

    Args:
        query: Natural language search query describing what to look for.
        top_k: Number of results to return (max 10).

    Returns:
        Formatted string with product search results including names,
        attributes, random_keys, and relevance scores.
    """
    retriever = get_retriever()
    top_k = min(top_k, 10)

    results = retriever.search_by_text(query, top_k=top_k)

    if not results:
        return "No products found matching the query."

    output_parts = [f"Found {len(results)} relevant products:\n"]
    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})
        name = metadata.get("name1", metadata.get("name", "Unknown"))
        random_key = metadata.get("random_key", r.get("id", "N/A"))
        score = r.get("score", 0.0)

        output_parts.append(
            f"{i}. **{name}**\n"
            f"   - random_key: {random_key}\n"
            f"   - score: {score:.4f}\n"
            f"   - content: {r.get('content', '')[:200]}\n"
        )

    return "\n".join(output_parts)


@tool
def search_products_by_image(image_base64: str, top_k: int = 10) -> str:
    """Search for products using an image.

    Use this tool when the user provides an image and wants to find
    similar or related products in the catalog.

    Args:
        image_base64: Base64-encoded image string.
        top_k: Number of results to return (max 10).

    Returns:
        Formatted string with visually similar product results.
    """
    retriever = get_retriever()
    top_k = min(top_k, 10)

    results = retriever.search_by_image(image_base64, top_k=top_k)

    if not results:
        return "No visually similar products found."

    output_parts = [f"Found {len(results)} visually similar products:\n"]
    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})
        name = metadata.get("name1", metadata.get("name", "Unknown"))
        random_key = metadata.get("random_key", r.get("id", "N/A"))
        score = r.get("score", 0.0)

        output_parts.append(
            f"{i}. **{name}**\n"
            f"   - random_key: {random_key}\n"
            f"   - score: {score:.4f}\n"
        )

    return "\n".join(output_parts)


@tool
def get_product_details(random_key: str) -> str:
    """Get detailed information about a specific product.

    Use this tool when you need to look up specific attributes or details
    of a product identified by its random_key.

    Args:
        random_key: The product's unique random_key identifier.

    Returns:
        Formatted string with all available product attributes.
    """
    retriever = get_retriever()
    details = retriever.get_product_details(random_key)

    if not details:
        return f"No product found with random_key: {random_key}"

    output_parts = [f"Product Details (random_key: {random_key}):\n"]
    for key, value in details.items():
        if value is not None and str(value).strip() and key != "embedding":
            output_parts.append(f"  - {key}: {value}")

    return "\n".join(output_parts)


@tool
def search_product_by_name(name: str) -> str:
    """Search for a product by its exact or partial name.

    Use this tool when the user mentions a specific product by name
    and you need to find its details or random_key.

    Args:
        name: Product name or partial name to search for.

    Returns:
        Formatted list of matching products with their keys.
    """
    retriever = get_retriever()
    results = retriever.get_product_by_name(name)

    if not results:
        return f"No products found matching name: '{name}'"

    output_parts = [f"Found {len(results)} products matching '{name}':\n"]
    for i, product in enumerate(results, 1):
        p_name = product.get("name1", product.get("name", "Unknown"))
        random_key = product.get("random_key", "N/A")
        output_parts.append(
            f"{i}. {p_name}\n   - random_key: {random_key}\n"
        )

    return "\n".join(output_parts)


# List of all available tools for the agent
AGENT_TOOLS = [
    search_products_by_text,
    search_products_by_image,
    get_product_details,
    search_product_by_name,
]
