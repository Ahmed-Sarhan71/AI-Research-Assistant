# AI Research Assistant

Local RAG (Retrieval-Augmented Generation) app that lets you upload PDFs and ask questions about them. Runs entirely offline using Ollama.

## Architecture

- **FastAPI** backend — ingests PDFs, searches vectors, calls Ollama
- **Streamlit** frontend — chat interface with source filtering
- **Qdrant** — vector database for document chunks
- **Ollama** — embeddings (`nomic-embed-text`) + LLM (`llama3.1:8b`)

## Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai) running with models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1:8b
  ```
- [Qdrant](https://qdrant.tech) running:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

## Setup

```bash
pip install -e .
```

## Run

```bash
# Terminal 1 — API server
uvicorn main:app --reload --port 8000

# Terminal 2 — UI
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Upload a PDF (`pdf_path`, `source_id`) |
| POST | `/query` | Ask a question (`question`, `top_k`, `source`) |
| GET | `/sources` | List uploaded PDFs |

Docs available at `http://localhost:8000/docs`.

## Project Structure

```
├── main.py            # FastAPI server
├── streamlit_app.py   # Streamlit UI
├── data_loader.py     # PDF parsing + embeddings
├── vector_db.py       # Qdrant operations
├── custom_types.py    # Pydantic models
├── pyproject.toml
└── uv.lock
```
