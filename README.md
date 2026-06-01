# MLOps - Machine Learning Systems Design

**Sharif University of Technology**  
**Dr. Fatemeh Seyed-Salehi**

A comprehensive Multimodal RAG (Retrieval-Augmented Generation) system with an intelligent Agent layer, built for the Torob e-commerce dataset.

---

## Project Overview

This project implements a production-ready ML system in two phases:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Multimodal RAG System (Text + Image + Hybrid retrieval) | ✅ Complete |
| **Phase 2** | Agentic AI with Tool-use, Memory, and Conversational Interface | ✅ Complete |

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Phase 2: Agentic AI                         │
│                                                                    │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  /chat API  │───▶│  LangGraph   │───▶│  Short-Term Memory   │  │
│  │  (FastAPI)  │    │  Agent Core  │    │  (Sliding Window)    │  │
│  └─────────────┘    └──────┬───────┘    └──────────────────────┘  │
│                            │                                       │
│                            ▼ Tool Calls                            │
├────────────────────────────────────────────────────────────────────┤
│                        Phase 1: RAG System                         │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  Text Search │   │ Image Search │   │  Hybrid/Multimodal   │  │
│  │  (SBERT)     │   │  (CLIP)      │   │  (Concatenation)     │  │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘  │
│         │                   │                      │               │
│         └───────────────────┼──────────────────────┘               │
│                             ▼                                      │
│              ┌──────────────────────────────┐                      │
│              │    ChromaDB Vector Store      │                      │
│              │    + Product Metadata         │                      │
│              └──────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
mlops/
├── README.md                 # This file
├── phase_1/                  # Phase 1: Multimodal RAG System
│   ├── README.md
│   ├── config.yaml
│   ├── preprocessing.ipynb
│   ├── generate-embeddings.ipynb
│   ├── model-text.ipynb
│   ├── model-image.ipynb
│   ├── model-multimodal.ipynb
│   └── eda.ipynb
├── phase_2/                  # Phase 2: Agentic AI
│   ├── README.md
│   ├── config.yaml
│   ├── requirements.txt
│   ├── cli.py
│   ├── app/
│   │   ├── main.py           # FastAPI /chat endpoint
│   │   ├── agent/
│   │   │   ├── core.py       # LangGraph agent
│   │   │   ├── memory.py     # Short-term memory
│   │   │   └── tools.py      # RAG tool wrappers
│   │   ├── rag/
│   │   │   └── retriever.py  # Phase 1 RAG integration
│   │   └── models/
│   │       └── schemas.py    # API schemas
│   └── tests/
│       └── test_api.py
└── .gitignore
```

## Quick Start

### Phase 1 (RAG Foundation)

```bash
cd phase_1
# Follow the Phase 1 README for data preparation and embedding generation
```

### Phase 2 (Agent System)

```bash
cd phase_2
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Start the API server
python cli.py serve

# Or use the interactive CLI
python cli.py chat
```

### Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "demo-1",
    "messages": [{"type": "text", "content": "I want an iPhone"}]
  }'
```

## Key Features

- **Multimodal RAG**: Text, image, and hybrid retrieval with ChromaDB
- **Intelligent Agent**: LangGraph-based decision-making with tool-use
- **Conversation Memory**: Sliding window short-term memory per session
- **Evidence-based Responses**: Grounded in retrieved product data
- **Multi-scenario Support**: Product search, comparison, seller hunt, image search
- **Production API**: FastAPI with proper validation and error handling

## Technologies

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| LLM Framework | LangChain |
| Vector DB | ChromaDB |
| Text Embeddings | Sentence-Transformers (paraphrase-multilingual-mpnet) |
| Image Embeddings | CLIP (ViT-B/32) |
| API Server | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Data Validation | Pydantic v2 |

## License

This project is developed as part of the Machine Learning Systems Design course at Sharif University of Technology.
