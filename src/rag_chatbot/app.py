import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import streamlit as st

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Folder holding the handbook, resolved relative to this file so the app
# works no matter which directory it's launched from.
DATA_DIR = os.path.join(ROOT_DIR, "data")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to your .env file.")
    st.stop()

Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Indexing the handbook...")
def get_query_engine():
    """Build the query engine once and reuse it across Streamlit reruns."""
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Student Handbook Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the handbook"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    query_engine = get_query_engine()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = query_engine.query(question).response
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
