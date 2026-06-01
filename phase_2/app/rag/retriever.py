"""RAG Retriever: Wraps Phase 1 RAG system as a callable tool.

This module provides the RAGRetriever class that interfaces with the
Phase 1 ChromaDB vector store to perform text and image-based retrieval.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Phase 1 RAG system wrapped as a callable tool for the Agent.

    Supports:
    - Text-based semantic search
    - Image-based search (CLIP embeddings)
    - Hybrid multimodal search
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG retriever with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.rag_config = self.config["rag"]
        self.top_k = self.rag_config["top_k"]
        self._text_model = None
        self._image_model = None
        self._image_processor = None
        self._collection = None
        self._metadata_df = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy initialization of models and database connections."""
        if self._initialized:
            return

        logger.info("Initializing RAG Retriever...")
        self._load_text_model()
        self._load_image_model()
        self._load_chroma_collection()
        self._load_metadata()
        self._initialized = True
        logger.info("RAG Retriever initialized successfully.")

    def _load_text_model(self) -> None:
        """Load the sentence-transformer text embedding model."""
        from sentence_transformers import SentenceTransformer

        model_id = self.rag_config["text_model_id"]
        logger.info(f"Loading text model: {model_id}")
        self._text_model = SentenceTransformer(model_id)

    def _load_image_model(self) -> None:
        """Load the CLIP image embedding model."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_id = self.rag_config["image_model_id"]
        logger.info(f"Loading image model: {model_id}")
        self._image_processor = CLIPProcessor.from_pretrained(model_id)
        self._image_model = CLIPModel.from_pretrained(model_id)
        self._image_model.eval()
        if torch.cuda.is_available():
            self._image_model = self._image_model.cuda()

    def _load_chroma_collection(self) -> None:
        """Connect to the ChromaDB collection from Phase 1."""
        import chromadb

        persist_dir = self.rag_config["chroma_persist_dir"]
        collection_name = self.rag_config["collection_name"]
        logger.info(f"Connecting to ChromaDB at: {persist_dir}")

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection '{collection_name}' loaded with "
            f"{self._collection.count()} documents."
        )

    def _load_metadata(self) -> None:
        """Load the product metadata DataFrame."""
        metadata_path = self.rag_config["metadata_path"]
        try:
            self._metadata_df = pd.read_parquet(metadata_path)
            logger.info(f"Loaded metadata: {len(self._metadata_df)} products.")
        except FileNotFoundError:
            logger.warning(
                f"Metadata file not found at {metadata_path}. "
                "Product detail lookups will be limited."
            )
            self._metadata_df = pd.DataFrame()

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding vector."""
        embedding = self._text_model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(embedding).flatten()

    def _encode_image(self, image_base64: str) -> np.ndarray:
        """Encode base64 image to CLIP embedding vector."""
        import torch
        from PIL import Image

        # Remove data URI prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = self._image_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._image_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()

    def search_by_text(
        self, query: str, top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Search products by text query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default from config).

        Returns:
            List of result dicts with keys: id, score, metadata, content.
        """
        self.initialize()
        k = top_k or self.top_k

        query_embedding = self._encode_text(query)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def search_by_image(
        self, image_base64: str, top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Search products by image.

        Args:
            image_base64: Base64-encoded image string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with keys: id, score, metadata, content.
        """
        self.initialize()
        k = top_k or self.top_k

        query_embedding = self._encode_image(image_base64)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def search_hybrid(
        self,
        text_query: Optional[str] = None,
        image_base64: Optional[str] = None,
        top_k: Optional[int] = None,
        text_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining text and image embeddings.

        Args:
            text_query: Text search query.
            image_base64: Base64-encoded image.
            top_k: Number of results.
            text_weight: Weight for text vs image (0-1).

        Returns:
            List of result dicts.
        """
        self.initialize()
        k = top_k or self.top_k

        if text_query and image_base64:
            text_emb = self._encode_text(text_query)
            image_emb = self._encode_image(image_base64)
            # Normalize dimensions for concatenation
            combined = np.concatenate(
                [text_emb * text_weight, image_emb * (1 - text_weight)]
            )
            combined = combined / np.linalg.norm(combined)
        elif text_query:
            combined = self._encode_text(text_query)
        elif image_base64:
            combined = self._encode_image(image_base64)
        else:
            return []

        results = self._collection.query(
            query_embeddings=[combined.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def get_product_details(self, random_key: str) -> Optional[dict[str, Any]]:
        """Get full product details by random_key.

        Args:
            random_key: The product's unique random_key identifier.

        Returns:
            Product details dict or None if not found.
        """
        if self._metadata_df is None or self._metadata_df.empty:
            return None

        matches = self._metadata_df[
            self._metadata_df["random_key"] == random_key
        ]
        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def get_product_by_name(self, name: str) -> list[dict[str, Any]]:
        """Search for products by exact or partial name match.

        Args:
            name: Product name to search for.

        Returns:
            List of matching product dicts.
        """
        if self._metadata_df is None or self._metadata_df.empty:
            return []

        # Case-insensitive partial match
        mask = self._metadata_df["name1"].str.contains(
            name, case=False, na=False
        )
        matches = self._metadata_df[mask]
        return matches.head(10).to_dict("records")

    def _format_results(self, raw_results: dict) -> list[dict[str, Any]]:
        """Format ChromaDB query results into a standardized list."""
        formatted = []
        if not raw_results or not raw_results.get("ids"):
            return formatted

        ids = raw_results["ids"][0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            result = {
                "id": doc_id,
                "content": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": 1 - distances[i] if i < len(distances) else 0.0,
            }
            # Extract random_key from metadata if available
            if result["metadata"] and "random_key" in result["metadata"]:
                result["random_key"] = result["metadata"]["random_key"]
            formatted.append(result)

        return formatted
