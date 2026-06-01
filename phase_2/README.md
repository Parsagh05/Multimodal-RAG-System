# Phase 2: Agentic AI on Phase 1 RAG

An intelligent Agent system that uses the Phase 1 RAG system as a tool to answer product-related queries in a conversational Q&A format.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│              (CLI / FastAPI /chat endpoint / Streamlit)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Core (LangGraph)                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Decision     │  │  Response         │  │  State Machine   │  │
│  │  Engine       │  │  Generation       │  │  (Graph Flow)    │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
└───────┬────────────────────┬─────────────────────┬──────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  RAG Tools    │  │  Short-Term      │  │  LLM Provider        │
│  (Phase 1)    │  │  Memory          │  │  (OpenAI/Ollama)     │
│               │  │                  │  │                      │
│ • Text Search │  │ • Sliding Window │  │ • GPT-4o-mini        │
│ • Image Search│  │ • Summaries      │  │ • Custom models      │
│ • Product Info│  │ • Context Store  │  │                      │
└───────┬───────┘  └──────────────────┘  └──────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ChromaDB (Phase 1 Vector Store)      │
│  + Product Metadata (Parquet)         │
└───────────────────────────────────────┘
```

## Features

- **Agent with Tool-use**: LangGraph-based agent that decides when to query the RAG system
- **Short-Term Memory**: Sliding window conversation context per session
- **Multi-modal Support**: Handles both text and image queries
- **Evidence-based Responses**: All answers are grounded in retrieved product data
- **Response Guardrails**: Honest about limitations, asks clarifying questions when needed
- **Multiple Interfaces**: FastAPI endpoint, CLI, ready for Streamlit

## Project Structure

```
phase_2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & /chat endpoint
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py          # LangGraph agent orchestrator
│   │   ├── memory.py        # Short-term memory manager
│   │   └── tools.py         # RAG tools (text/image search, product details)
│   ├── rag/
│   │   ├── __init__.py
│   │   └── retriever.py     # Phase 1 RAG wrapper
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Unit & integration tests
├── cli.py                   # Interactive CLI interface
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd phase_2
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# OR for Ollama (local)
# Update config.yaml: llm.base_url = "http://localhost:11434/v1"
```

### 3. Configure

Edit `config.yaml` to match your setup:
- Set the correct paths to Phase 1 outputs (ChromaDB, metadata)
- Choose your LLM provider (OpenAI or Ollama)
- Adjust memory and agent parameters

### 4. Run the API Server

```bash
# From the phase_2 directory
python cli.py serve

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Use the CLI

```bash
python cli.py chat
```

### 6. Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test-session-1",
    "messages": [{"type": "text", "content": "I want an iPhone 17 Pro Max"}]
  }'
```

## API Contract

### POST `/chat`

**Request:**
```json
{
  "chat_id": "unique-session-id",
  "messages": [
    {"type": "text", "content": "user message"},
    {"type": "image", "content": "base64-encoded-image"}
  ]
}
```

**Response:**
```json
{
  "message": "Agent's text response",
  "base_random_keys": ["product-key-1", "product-key-2"],
  "member_random_keys": ["seller-key-1"]
}
```

### Supported Scenarios

| # | Scenario | Description |
|---|----------|-------------|
| 1 | Specific Product Feature | Ask about an attribute of a named product |
| 2 | Seller Information | Ask about sellers/prices of a specific product |
| 3 | Seller Hunt (Multi-turn) | Guided conversation to find a seller |
| 4 | Product Comparison | Compare 2+ products for a use case |
| 5 | Image-based Search (Bonus) | Find products matching an image |

## Agent Decision Policy

The agent makes explainable decisions:

- **Use RAG Tool**: When the question involves products, features, prices, or sellers from the catalog
- **Read Memory**: When the question references previous conversation context
- **Ask Clarification**: When the query is too vague to provide useful results
- **Decline**: When the question is completely unrelated to product search

## Memory Design

### Short-Term Memory
- **Method**: Sliding window (configurable, default: last 20 messages)
- **Scope**: Per `chat_id` session
- **Features**: Automatic trimming, optional summarization

### Session Context
- Stores extracted user preferences (budget, brand, etc.)
- Available across the conversation turns

## Configuration Reference

```yaml
llm:
  provider: "openai"        # openai or ollama
  model: "gpt-4o-mini"     # Model name
  temperature: 0.3          # Response creativity
  max_tokens: 1024          # Max response length

rag:
  chroma_persist_dir: "..."  # Path to Phase 1 ChromaDB
  collection_name: "products"
  top_k: 10                  # Number of results per search

memory:
  max_messages: 20           # Sliding window size
  summary_threshold: 15      # When to summarize

agent:
  rag_confidence_threshold: 0.5
  max_clarification_rounds: 3
```

## Running Tests

```bash
cd phase_2
pytest tests/ -v
```

## Technologies Used

- **LangGraph**: Agent orchestration and state machine
- **LangChain**: Tool abstractions and LLM wrappers
- **FastAPI**: REST API server
- **ChromaDB**: Vector database (Phase 1 integration)
- **Sentence-Transformers**: Text embeddings
- **CLIP**: Image embeddings
- **Rich + Typer**: CLI interface
- **Pydantic**: Data validation

## Monitoring (Bonus)

The system is prepared for OpenTelemetry + Phoenix integration:
- Trace conversations and tool calls
- Track metrics: tokens/message, response times
- Log tool invocations and memory operations

To enable, install optional dependencies and configure in `config.yaml`.
