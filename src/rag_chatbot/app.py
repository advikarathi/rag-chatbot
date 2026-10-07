import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ClientError, ServerError
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

# Handbook documents live here.
DATA_DIR = ROOT_DIR / "data"

# The built index is saved here so restarts skip re-embedding.
# Delete this folder after changing anything in DATA_DIR to force a rebuild.
STORAGE_DIR = ROOT_DIR / "storage"

load_dotenv(ENV_FILE)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Student Handbook Chatbot", page_icon="📘")


# --------------------------------------------------
# FAIL FAST: VALIDATE SETUP
# --------------------------------------------------

def get_api_key():
    """Return GEMINI_API_KEY from .env, or stop the app with a fix-it message."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.error(
            "**GEMINI_API_KEY is missing.**\n\n"
            f"Add a line like `GEMINI_API_KEY=your-key-here` to `{ENV_FILE}`, "
            "then restart the app."
        )
        st.stop()
    return api_key


def check_data_dir():
    """Stop the app unless DATA_DIR is a folder with at least one visible file."""
    if not DATA_DIR.is_dir():
        st.error(
            "**Handbook folder not found.**\n\n"
            f"The app expected a folder at `{DATA_DIR}`. "
            "Create it (or rename your folder to `data`) and put the handbook inside."
        )
        st.stop()

    # Ignore hidden files like .DS_Store; LlamaIndex skips them too.
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        st.error(
            "**The handbook folder is empty.**\n\n"
            f"Add at least one document (for example a PDF) to `{DATA_DIR}`, "
            "then restart the app."
        )
        st.stop()


# --------------------------------------------------
# BUILD / LOAD RAG INDEX (cached)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the handbook...")
def get_query_engine(api_key):
    """Configure the models and return a query engine, built once per server run."""
    Settings.llm = GoogleGenAI(model="gemini-3.8-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    if STORAGE_DIR.is_dir():
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    return index.as_query_engine(similarity_top_k=3, response_mode="compact")


def show_details(error):
    """Show the raw exception in a collapsed expander for debugging."""
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")


# --------------------------------------------------
# STARTUP: checks, then engine
# --------------------------------------------------

api_key = get_api_key()
check_data_dir()

try:
    query_engine = get_query_engine(api_key)
except ClientError as error:
    st.error(
        "**Gemini rejected the app's setup.**\n\n"
        f"Check that GEMINI_API_KEY in `{ENV_FILE}` is a valid key and that the "
        "model name in get_query_engine() exists, then restart the app."
    )
    show_details(error)
    st.stop()
except httpx.TransportError as error:
    st.error(
        "**Couldn't reach Gemini.**\n\n"
        "Check your internet connection, then refresh the page."
    )
    show_details(error)
    st.stop()
except Exception as error:
    st.error(
        "**The handbook couldn't be loaded.**\n\n"
        f"A file in `{DATA_DIR}` may be corrupt, or a model download failed. "
        f"Try deleting `{STORAGE_DIR}` and restarting the app."
    )
    show_details(error)
    st.stop()


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("📘 Student Handbook Chatbot")
st.caption("Ask questions about the student handbook.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# FALLBACK: ANSWER ONE QUESTION
# --------------------------------------------------

if question := st.chat_input("Ask a question about the handbook"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the handbook..."):
                answer = query_engine.query(question).response
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except ClientError as error:
            if error.code == 429:
                st.error("Gemini's rate limit was reached. Wait a minute, then ask again.")
            else:
                st.error("Gemini couldn't answer that question. Try rephrasing it.")
            show_details(error)
        except ServerError as error:
            st.error("Gemini is temporarily busy. Please try again in a few moments.")
            show_details(error)
        except httpx.TransportError as error:
            st.error("Couldn't reach Gemini. Check your internet connection and ask again.")
            show_details(error)
        except Exception as error:
            st.error("Something went wrong while answering. Please try again.")
            show_details(error)
