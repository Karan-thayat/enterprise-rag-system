import streamlit as st
import requests

# 1. App Configuration
st.set_page_config(page_title="Enterprise RAG", page_icon="🧠", layout="centered")
st.title("Enterprise RAG System ")
st.caption("Chat with your documents. Powered by Llama-3.1, ChromaDB, and FastAPI.")

# 2. API Endpoints
API_URL_ASK = "http://127.0.0.1:8000/ask"
API_URL_UPLOAD = "http://127.0.0.1:8000/upload"

# --- NEW: SIDEBAR DOCUMENT UPLOADER ---
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Uploading and processing... This may take a minute."):
                try:
                    # Prepare the file for HTTP transmission
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    # Send the POST request to FastAPI
                    response = requests.post(API_URL_UPLOAD, files=files)
                    
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Success!"))
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend server.")

# --- CHAT INTERFACE ---
# 3. Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Chat Input Box
if prompt := st.chat_input("Ask a question about the document..."):
    
    # Capture the history exactly as it is BEFORE we add the new question
    # This prevents the AI from seeing the new question twice
    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    # Immediately display the user's new question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Send the question AND history to FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating answer..."):
            try:
                # NEW: We are now sending the history array in the JSON payload!
                payload = {
                    "question": prompt,
                    "history": chat_history
                }
                response = requests.post(API_URL_ASK, json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is your FastAPI server running?")