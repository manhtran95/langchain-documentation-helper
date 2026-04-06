# LangChain Documentation Helper

A chatbot built with LangChain that answers questions about the LangChain documentation.

## How it works

1. **Ingestion** — Tavily crawls the LangChain documentation and the content is chunked and indexed into a Pinecone vector store.
2. **Retrieval** — When a user asks a question, relevant chunks are retrieved from Pinecone using semantic search.
3. **Chat** — A LangChain-powered LLM answers the question using the retrieved context.

## Frontend

Built with [Streamlit](https://streamlit.io).

## Environment variables

Create a `.env` file and fill in:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | From [platform.openai.com](https://platform.openai.com) |
| `PINECONE_API_KEY` | From [pinecone.io](https://www.pinecone.io) |
| `INDEX_NAME` | Your Pinecone index name |
| `LANGSMITH_API_KEY` | From [langchain.com/langsmith](https://www.langchain.com/langsmith) |
| `LANGSMITH_PROJECT` | Your LangSmith project name |
| `TAVILY_API_KEY` | From [app.tavily.com](https://app.tavily.com) |

## Usage

### 1. Install dependencies

```bash
uv sync
```

### 2. Run ingestion

```bash
uv run python ingestion.py
```

### 3. Start the app at localhost:8501
```bash
uv run streamlit run ingestion.py
```