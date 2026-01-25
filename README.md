# Multimodal RAG System - Phase 1

This repository contains the first phase of a Multimodal Retrieval-Augmented Generation (RAG) system developed for the Torob dataset. The system compares text-only, image-only, and hybrid (multimodal) retrieval strategies.

## Project Structure

- **eda.ipynb**: Exploratory data analysis of the Torob dataset.
- **preprocessing.ipynb**: Data cleaning and preprocessing pipeline.
- **model-text.ipynb**: Implementation and evaluation of text-based retrieval.
- **model-image.ipynb**: Implementation and evaluation of image-based retrieval (CLIP).
- **model-multimodal.ipynb**: Hybrid retrieval system combining text and image embeddings.
- **config.yaml**: Main configuration file for models, paths, and pipeline parameters.

## Performance Overview

| Metric | Text-Only | Image-Only | Hybrid (Multimodal) |
| --- | --- | --- | --- |
| **Precision@10** | 0.6452 | 0.2729 | **0.6688** |
| **Recall@10** | **0.2304** | 0.0623 | 0.2020 |
| **MRR** | **1.0000** | 0.4855 | 0.8626 |
| **NDCG** | **0.9307** | 0.5014 | 0.8466 |

## Key Features

- **Multilingual Support**: Uses `paraphrase-multilingual-mpnet-base-v2` for text embeddings.
- **Visual Search**: Leverages `openai/clip-vit-base-patch32` for image processing.
- **Hybrid Fusion**: Employs concatenation strategy for multi-modal embedding generation.
- **Efficient Retrieval**: Implementation supports HNSW indexing for performance.

## Status: Phase 1
Phase 1 focuses on establishing a robust retrieval baseline and comparing unimodal vs. multimodal architectures.
