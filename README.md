# Student Handbook Chatbot

A Streamlit RAG chatbot that answers questions about the undergraduate student handbook, using LlamaIndex, a local `BAAI/bge-small-en-v1.5` embedding model, and Gemini.

## Run it

1. Put `GEMINI_API_KEY=your-key-here` in a `.env` file at the project root.
2. Put the handbook (PDF) in `data/`.
3. Run `uv run streamlit run src/rag_chatbot/app.py`.

The index is saved to `storage/` after the first run. Delete `storage/` after changing anything in `data/` to force a rebuild.

## Robustness checks

| Check | What it protects against | Where in your code | Fail fast or fallback? |
|---|---|---|---|
| API key | Missing or misnamed `GEMINI_API_KEY` in `.env` | `get_api_key()`, lines 47–57 | Fail fast |
| Data folder exists | `data/` renamed, moved or missing | `check_data_dir()`, lines 62–68 | Fail fast |
| Data folder has files | Empty `data/` (hidden files like `.DS_Store` are ignored) | `check_data_dir()`, lines 70–78 | Fail fast |
| Engine build: key rejected | Invalid key or wrong model name (Gemini `ClientError`) | `try`/`except` around `get_query_engine()`, lines 115–124 | Fail fast |
| Engine build: no network | No internet at startup (`httpx.TransportError`) | lines 125–131 | Fail fast |
| Engine build: anything else | Corrupt PDF, failed embedding-model download, bad saved index | lines 132–139 | Fail fast |
| Question: rate limit / bad request | Gemini `ClientError` (429 rate limit, or another 4xx) | `try`/`except` around `query_engine.query()`, lines 167–177 | Fallback |
| Question: Gemini overloaded | Gemini `ServerError` (5xx, e.g. 503) | lines 178–180 | Fallback |
| Question: connection dropped | Wi-Fi off or network drop mid-session (`httpx.TransportError`) | lines 181–183 | Fallback |
| Question: anything else | Any other unexpected error while answering | lines 184–186 | Fallback |

Other changes:
- `get_query_engine()` is cached with `@st.cache_resource`, so documents are indexed and the embedding model is loaded only once. The `Settings` lines now live inside it, and the validated key is passed in directly: `GoogleGenAI(..., api_key=api_key)`.
- The index is built with `VectorStoreIndex.from_documents()` so documents are chunked before embedding.
- Every error shows a short fix-it message, plus a collapsed "Technical details" expander with the raw exception.
- Stretch goals: chat history (`st.session_state` + `st.chat_message`), a spinner while searching, and the index persisted to `storage/`.

Note: Gemini is contacted when the engine is built, so turning Wi-Fi off **before** starting the app triggers the fail-fast "Couldn't reach Gemini" message. Turning it off **after** the app has loaded and then asking a question triggers the fallback message, and the app keeps running.

## Screenshots

### 1. Renamed `GEMINI_API_KEY` in `.env`
<!-- screenshot -->

### 2. Renamed the `data` folder
<!-- screenshot -->

### 3. Emptied the `data` folder
<!-- screenshot -->

### 4. Wi-Fi off, then asked a question
<!-- screenshot -->
