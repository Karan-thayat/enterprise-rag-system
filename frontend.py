import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_HISTORY_MESSAGES = 6

st.set_page_config(page_title="Enterprise RAG", page_icon="🧠", layout="centered")
st.title("Enterprise RAG System")
st.caption(
    "Chat with your PDFs: hybrid BM25 + vector retrieval, cross-encoder reranking "
    "and cited answers from GPT-OSS 20B on Groq."
)


def call_api(method: str, path: str, timeout: float, **kwargs) -> tuple[requests.Response | None, str | None]:
    """Calls the backend and returns (response, None) on success or (None, error message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the backend at {API_URL}. Is the FastAPI server running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond."
    if response.ok:
        return response, None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"Error {response.status_code}: {detail}"


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**[{source['number']}] {source['filename']}, page {source['page']}**")
            # Collapse the layout whitespace LiteParse preserves, for readability.
            snippet = " ".join(source["text"].split())
            st.caption(snippet[:600] + ("…" if len(snippet) > 600 else ""))


# --- Sidebar: document management ---
with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Process document", type="primary"):
        with st.spinner("Parsing, chunking and indexing… large or scanned PDFs can take a minute."):
            response, error = call_api(
                "POST",
                "/upload",
                timeout=600,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if error:
            st.error(error)
        else:
            st.success(response.json()["message"])

    response, error = call_api("GET", "/documents", timeout=10)
    if error:
        st.warning(error)
    elif not response.json():
        st.info("No documents indexed yet.")
    else:
        for document in response.json():
            name_column, delete_column = st.columns([5, 1])
            name_column.markdown(
                f"**{document['filename']}**  \n{document['pages']} pages · {document['chunks']} chunks"
            )
            if delete_column.button("🗑️", key=f"delete_{document['doc_id']}", help="Remove from the index"):
                _, error = call_api("DELETE", f"/documents/{document['doc_id']}", timeout=30)
                if error:
                    st.error(error)
                else:
                    st.rerun()

    health, _ = call_api("GET", "/health", timeout=10)
    if health is not None and not health.json()["llm_configured"]:
        st.warning("GROQ_API_KEY is not set on the server: search works, but answers are disabled.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("failed") and message["role"] == "assistant":
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if prompt := st.chat_input("Ask a question about your documents…"):
    # Send only completed turns, and only the most recent ones.
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
        if not message.get("failed")
    ][-MAX_HISTORY_MESSAGES:]

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating an answer…"):
            response, error = call_api(
                "POST", "/ask", timeout=120, json={"question": prompt, "history": history}
            )
        if error:
            st.error(error)
            user_message["failed"] = True
            st.session_state.messages.append({"role": "assistant", "content": error, "failed": True})
        else:
            data = response.json()
            st.markdown(data["answer"])
            if data["sources"]:
                render_sources(data["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
            )
