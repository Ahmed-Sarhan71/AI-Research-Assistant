from pathlib import Path
import base64

import streamlit as st
import httpx
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = "http://localhost:8000"


def set_background(image_path: str):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{data});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="RAG", layout="centered")

set_background("bg.png")


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


st.title("📄 Add a Document")
uploaded = st.file_uploader("Select a PDF to add to your knowledge base", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Processing your document..."):
        path = save_uploaded_pdf(uploaded)
        resp = httpx.post(
            f"{BACKEND_URL}/ingest",
            json={"pdf_path": str(path.resolve()), "source_id": path.name},
            timeout=500,
        )
        resp.raise_for_status()
        data = resp.json()
    st.success(f"✅ Added {data['ingested']} chunks from: {path.name}")
    st.caption("You can upload more documents anytime.")

st.divider()
st.title("💬 Ask Anything")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

@st.cache_data(ttl=10)
def fetch_sources() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/sources", timeout=5)
        resp.raise_for_status()
        return resp.json().get("sources", [])
    except Exception:
        return []

available_sources = fetch_sources()
source_options = ["All documents"] + available_sources
selected_source = st.selectbox("Search in", source_options)

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            body = {"question": prompt, "top_k": 5}
            if selected_source != "All documents":
                body["source"] = selected_source
            resp = httpx.post(
                f"{BACKEND_URL}/query",
                json=body,
                timeout=500,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

        full = answer
        if sources:
            full += "\n\n📎 " + ", ".join(sources)
        st.markdown(full)
        st.session_state.messages.append({"role": "assistant", "content": full})

