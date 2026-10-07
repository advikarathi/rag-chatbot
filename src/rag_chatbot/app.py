import os
import time

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ServerError
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

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data")
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")

load_dotenv(os.path.join(ROOT_DIR, ".env"))


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Student Handbook Chatbot",
    page_icon="📘",
)


# --------------------------------------------------
# CHECK API KEY
# --------------------------------------------------

if not (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
):
    st.error(
        "Google API key is missing. "
        "Add GOOGLE_API_KEY or GEMINI_API_KEY "
        "to your .env file."
    )
    st.stop()


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """Load Gemini and the embedding model once."""

    llm = GoogleGenAI(
        model="gemini-3.8-flash"
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return llm, embed_model


Settings.llm, Settings.embed_model = load_models()


# --------------------------------------------------
# BUILD / LOAD RAG INDEX
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading handbook...")
def get_query_engine():
    """Load saved vector index or build a new one."""

    if os.path.isdir(STORAGE_DIR):

        storage_context = StorageContext.from_defaults(
            persist_dir=STORAGE_DIR
        )

        index = load_index_from_storage(
            storage_context
        )

    else:

        if not os.path.isdir(DATA_DIR):
            raise FileNotFoundError(
                f"Data folder not found: {DATA_DIR}"
            )

        documents = SimpleDirectoryReader(
            DATA_DIR
        ).load_data()

        if not documents:
            raise ValueError(
                "No handbook documents were found "
                "inside the data folder."
            )

        index = VectorStoreIndex.from_documents(
            documents
        )

        index.storage_context.persist(
            persist_dir=STORAGE_DIR
        )

    return index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact"
    )


# --------------------------------------------------
# GEMINI RETRY FUNCTION
# --------------------------------------------------

def query_with_retry(
    query_engine,
    question,
    max_retries=4
):
    """
    Query Gemini.

    If Google temporarily returns a 503 error,
    retry using exponential backoff.
    """

    delay = 2

    for attempt in range(max_retries):

        try:

            response = query_engine.query(
                question
            )

            return response.response

        except ServerError as error:

            error_text = str(error)

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            )

            if not temporary_error:
                raise

            if attempt == max_retries - 1:
                raise

            time.sleep(delay)

            # Retry delays:
            # 2 sec
            # 4 sec
            # 8 sec
            delay *= 2


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title("📘 Student Handbook Chatbot")

st.caption(
    "Ask questions about the student handbook."
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the handbook"
)


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------
    # GET QUERY ENGINE
    # --------------------------------------------------

    try:

        query_engine = get_query_engine()

    except Exception as error:

        st.error(
            "The handbook could not be loaded."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(error))

        st.stop()


    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the handbook..."
        ):

            try:

                answer = query_with_retry(
                    query_engine,
                    question
                )

                st.markdown(answer)


            except ServerError:

                answer = (
                    "Gemini is temporarily experiencing "
                    "high demand. Please try your question "
                    "again in a few moments."
                )

                st.warning(answer)


            except Exception as error:

                answer = (
                    "Something went wrong while processing "
                    "your question."
                )

                st.error(answer)

                with st.expander(
                    "Technical details"
                ):
                    st.code(str(error))


    # --------------------------------------------------
    # SAVE ASSISTANT ANSWER
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )