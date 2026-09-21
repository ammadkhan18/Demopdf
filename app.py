"""
Chat with your PDF - a simple RAG (Retrieval-Augmented Generation) demo.

Stack:
  - Streamlit        : web front end
  - pypdf            : PDF text extraction
  - fastembed        : small local embedding model (Groq has no embeddings API)
  - numpy            : cosine-similarity search over chunk vectors
  - Groq             : fast LLM inference for the final answer

Secrets (Streamlit Cloud -> App settings -> Secrets, or .streamlit/secrets.toml):
  GROQ_API_KEY = "gsk_..."
"""

import hashlib
from io import BytesIO

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM = "openai/gpt-oss-120b"  # change in the sidebar if Groq retires it
HISTORY_TURNS = 3  # previous Q&A pairs sent to the model for follow-up questions

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's PDF "
    "documents. Use ONLY the provided context excerpts. If the answer is not in "
    "the context, say you could not find it in the document - do not guess. "
    "Be concise and, where useful, mention the page number(s) you relied on."
)

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings so a dot product equals cosine similarity."""
    vectors = np.array(list(get_embedder().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def extract_pages(name: str, data: bytes) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return []
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"source": name, "page": number, "text": text})
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring to break at spaces."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index(files, chunk_size: int, overlap: int):
    """Read PDFs -> chunk -> embed. Returns (chunks, embeddings)."""
    chunks = []
    for f in files:
        for page in extract_pages(f.name, f.getvalue()):
            for piece in split_text(page["text"], chunk_size, overlap):
                chunks.append(
                    {"source": page["source"], "page": page["page"], "text": piece}
                )
    if not chunks:
        return [], None
    embeddings = embed_texts([c["text"] for c in chunks])
    return chunks, embeddings


def retrieve(query: str, chunks: list[dict], embeddings: np.ndarray, top_k: int):
    query_vec = embed_texts([query])[0]
    scores = embeddings @ query_vec
    best = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best]


def stream_answer(client: Groq, model: str, temperature: float, messages: list[dict]):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for part in stream:
        delta = part.choices[0].delta.content
        if delta:
            yield delta


# --------------------------------------------------------------------------- #
# API key
# --------------------------------------------------------------------------- #
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    st.error(
        "GROQ_API_KEY not found. Add it to Streamlit secrets "
        '(e.g. `GROQ_API_KEY = "gsk_..."`) and reload.'
    )
    st.stop()

client = Groq(api_key=api_key)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    st.header("⚙️ Settings")
    model_name = st.text_input("Groq model", value=DEFAULT_LLM)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Chunks retrieved per question", 1, 10, 5)
    chunk_size = st.slider("Chunk size (characters)", 400, 2000, 1000, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 200, 50)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Build / refresh the vector index when the files or chunk settings change
# --------------------------------------------------------------------------- #
st.title("Chat with your PDF")
st.caption("Retrieval-Augmented Generation with Streamlit + Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded:
    st.session_state.pop("index_key", None)
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

digest = hashlib.sha256()
for f in uploaded:
    digest.update(f.name.encode())
    digest.update(f.getvalue())
digest.update(f"{chunk_size}-{overlap}".encode())
index_key = digest.hexdigest()

if st.session_state.get("index_key") != index_key:
    with st.spinner("Reading and indexing your PDF(s)..."):
        try:
            chunks, embeddings = build_index(uploaded, chunk_size, overlap)
        except Exception as exc:
            st.error(f"Could not process the PDF: {exc}")
            st.stop()
    if not chunks:
        st.warning(
            "No extractable text found. The PDF may be a scanned image - "
            "this demo does not include OCR."
        )
        st.stop()
    st.session_state.update(
        index_key=index_key, chunks=chunks, embeddings=embeddings, messages=[]
    )

chunks = st.session_state.chunks
embeddings = st.session_state.embeddings
st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} file(s). Ask away!")

# --------------------------------------------------------------------------- #
# Chat UI
# --------------------------------------------------------------------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']} - page {src['page']}** "
                        f"(similarity {src['score']:.2f})\n\n> {src['text'][:400]}..."
                    )

question = st.chat_input("Ask a question about your document(s)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question, chunks, embeddings, top_k)
    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}]\n{c['text']}" for c, _ in hits
    )

    # Recent plain Q&A history (without the bulky context) for follow-up questions
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1][-2 * HISTORY_TURNS :]
    ]
    api_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ]
    )

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(client, model_name, temperature, api_messages)
            )
        except Exception as exc:
            answer = f"Sorry, the Groq request failed: {exc}"
            st.error(answer)

        sources = [
            {
                "source": c["source"],
                "page": c["page"],
                "score": score,
                "text": c["text"],
            }
            for c, score in hits
        ]
        with st.expander("Sources"):
            for src in sources:
                st.markdown(
                    f"**{src['source']} - page {src['page']}** "
                    f"(similarity {src['score']:.2f})\n\n> {src['text'][:400]}..."
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
