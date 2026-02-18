import streamlit as st
import os
import requests
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — NVIDIA AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — refined dark SaaS aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg: #080c14;
  --surface: #0e1520;
  --surface2: #141d2e;
  --border: rgba(99,179,237,0.12);
  --border-active: rgba(99,179,237,0.35);
  --accent: #63b3ed;
  --accent2: #4ade80;
  --accent3: #f472b6;
  --text: #e2e8f0;
  --text-muted: #64748b;
  --text-dim: #94a3b8;
  --nvidia: #76b900;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Hide streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Main content ── */
.main .block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1100px !important;
}

/* ── Header ── */
.app-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 2rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
}
.app-logo {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, #76b900 0%, #4ade80 100%);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-family: 'Syne', sans-serif; font-weight: 800;
  color: #080c14;
}
.app-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 1.45rem;
  color: var(--text);
  margin: 0;
  line-height: 1;
}
.app-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  color: var(--nvidia);
  letter-spacing: 0.08em;
  margin-top: 3px;
}

/* ── Status badges ── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 99px;
  font-size: 0.7rem; font-family: 'DM Mono', monospace;
  font-weight: 500; letter-spacing: 0.04em;
}
.badge-green { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
.badge-blue  { background: rgba(99,179,237,0.1); color: #63b3ed; border: 1px solid rgba(99,179,237,0.25); }
.badge-amber { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

/* ── Upload zone ── */
.stFileUploader > div {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border-active) !important;
  border-radius: 14px !important;
  padding: 2rem !important;
  transition: border-color 0.2s;
}
.stFileUploader > div:hover { border-color: var(--accent) !important; }

/* ── Chat messages ── */
.msg-wrap { display: flex; gap: 12px; margin-bottom: 16px; animation: fadeUp 0.3s ease; }
@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

.msg-avatar {
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items:center; justify-content:center;
  font-size: 14px; flex-shrink: 0; margin-top: 2px;
}
.avatar-user { background: rgba(99,179,237,0.15); border: 1px solid rgba(99,179,237,0.3); }
.avatar-bot  { background: rgba(118,185,0,0.15);  border: 1px solid rgba(118,185,0,0.3);  }

.msg-bubble {
  padding: 12px 16px;
  border-radius: 14px;
  max-width: 85%;
  line-height: 1.65;
  font-size: 0.92rem;
}
.bubble-user {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-top-left-radius: 4px;
}
.bubble-bot {
  background: rgba(118,185,0,0.06);
  border: 1px solid rgba(118,185,0,0.18);
  border-top-left-radius: 4px;
}

.msg-meta {
  font-size: 0.65rem; font-family: 'DM Mono', monospace;
  color: var(--text-muted); margin-top: 5px;
}

/* ── Source cards ── */
.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 8px;
  font-size: 0.8rem;
  color: var(--text-dim);
  font-family: 'DM Mono', monospace;
  line-height: 1.6;
}
.source-label {
  font-size: 0.65rem;
  color: var(--accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 10px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
  flex: 1; min-width: 120px;
}
.stat-value {
  font-family: 'Syne', sans-serif;
  font-weight: 700; font-size: 1.5rem;
  color: var(--text); line-height: 1;
}
.stat-label {
  font-size: 0.7rem; font-family: 'DM Mono', monospace;
  color: var(--text-muted); margin-top: 4px;
  letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Dividers ── */
.section-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-active) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: rgba(99,179,237,0.1) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
  background: var(--nvidia) !important;
  border-color: var(--nvidia) !important;
  color: #080c14 !important;
  font-weight: 600 !important;
}

.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}

.stSlider > div { color: var(--text) !important; }

[data-testid="stChatInput"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-active) !important;
  border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

.stSuccess > div {
  background: rgba(74,222,128,0.07) !important;
  border: 1px solid rgba(74,222,128,0.25) !important;
  border-radius: 10px !important;
  color: #4ade80 !important;
}
.stWarning > div {
  background: rgba(251,191,36,0.07) !important;
  border: 1px solid rgba(251,191,36,0.25) !important;
  border-radius: 10px !important;
}
.stInfo > div {
  background: rgba(99,179,237,0.07) !important;
  border: 1px solid rgba(99,179,237,0.2) !important;
  border-radius: 10px !important;
  color: #93c5fd !important;
}

.stSpinner > div { color: var(--accent) !important; }
.stProgress > div > div { background: var(--nvidia) !important; }

/* Expanders */
details > summary {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.8rem !important;
  color: var(--text-dim) !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 0 10px 10px !important;
}

/* Sidebar labels */
.stSidebar label, .stSidebar .stMarkdown {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text-dim) !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

/* Chat input container */
.chat-input-area {
  position: sticky;
  bottom: 0;
  background: var(--bg);
  padding-top: 12px;
  padding-bottom: 8px;
  border-top: 1px solid var(--border);
  margin-top: 12px;
}

/* Empty state */
.empty-state {
  text-align: center;
  padding: 3rem 2rem;
  color: var(--text-muted);
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-state h3 {
  font-family: 'Syne', sans-serif;
  font-weight: 700; font-size: 1.1rem;
  color: var(--text-dim); margin-bottom: 8px;
}
.empty-state p { font-size: 0.85rem; line-height: 1.6; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-active); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

MODELS = {
    "Llama 3 8B Instruct":  "meta/llama3-8b-instruct",
    "Llama 3 70B Instruct": "meta/llama3-70b-instruct",
    "Mixtral 8x7B":         "mistralai/mixtral-8x7b-instruct-v0.1",
    "Gemma 7B":             "google/gemma-7b",
    "Phi-3 Mini":           "microsoft/phi-3-mini-128k-instruct",
}

RESPONSE_STYLES = {
    "Precise & Concise":  "Answer precisely and concisely. Avoid fluff. Stick to facts from the context.",
    "Detailed & Thorough":"Answer in detail, covering all relevant aspects found in the context.",
    "Bullet Points":      "Format your answer using clear bullet points and short headings where helpful.",
    "ELI5 (Simple)":      "Explain like I'm 5 — use very simple language and short sentences.",
}


# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "doc_stats": {},
        "pdf_name": None,
        "total_tokens": 0,
        "query_count": 0,
        "chat_history_ctx": [],   # for multi-turn context
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(file_bytes, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    embeddings = load_embeddings()
    vs = FAISS.from_documents(splits, embeddings)
    Path(tmp_path).unlink(missing_ok=True)
    return vs, {
        "pages": len(docs),
        "chunks": len(splits),
        "avg_chunk": int(sum(len(s.page_content) for s in splits) / max(len(splits),1)),
    }

def call_nvidia(api_key, model, messages, temperature, max_tokens, stream=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    return requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload, stream=stream, timeout=60)

def build_prompt(query, context, style_instruction, history, use_history):
    sys_msg = f"""You are a helpful AI assistant specialized in answering questions about uploaded documents.
{style_instruction}
Use ONLY the provided context to answer. If the answer is not in the context, say "I couldn't find that in the document."
Always be accurate and cite relevant details."""

    msgs = [{"role": "system", "content": sys_msg}]
    if use_history and history:
        msgs.extend(history[-6:])   # last 3 turns
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return msgs

def render_message(role, content, ts, sources=None):
    avatar_cls = "avatar-user" if role == "user" else "avatar-bot"
    bubble_cls = "bubble-user" if role == "user" else "bubble-bot"
    icon = "◈" if role == "user" else "⬡"
    st.markdown(f"""
<div class="msg-wrap">
  <div class="msg-avatar {avatar_cls}">{icon}</div>
  <div>
    <div class="msg-bubble {bubble_cls}">{content}</div>
    <div class="msg-meta">{ts}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    if sources:
        with st.expander(f"📎 {len(sources)} source chunk(s) retrieved"):
            for i, s in enumerate(sources, 1):
                page = s.metadata.get("page", "?")
                snippet = s.page_content[:280].replace("\n", " ") + "…"
                st.markdown(f"""<div class="source-card">
<div class="source-label">Source {i} · Page {page}</div>
{snippet}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;
color:#e2e8f0;margin-bottom:0.2rem;">⬡ DocMind</div>
<div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#76b900;
letter-spacing:0.1em;margin-bottom:1.5rem;">POWERED BY NVIDIA</div>
""", unsafe_allow_html=True)

    # API Key
    st.markdown("#### 🔑 API Key")
    api_key_input = ""
    try:
        api_key_input = st.secrets["NVIDIA_API_KEY"]
        st.markdown('<span class="badge badge-green">● Connected</span>', unsafe_allow_html=True)
    except:
        api_key_input = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-…")
        if api_key_input:
            st.markdown('<span class="badge badge-amber">● Manual Key</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-amber">● No Key Set</span>', unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Model selection
    st.markdown("#### 🧠 Model")
    model_label = st.selectbox("", list(MODELS.keys()), label_visibility="collapsed")
    selected_model = MODELS[model_label]

    # Response style
    st.markdown("#### 🎨 Response Style")
    style_label = st.selectbox("", list(RESPONSE_STYLES.keys()), label_visibility="collapsed")
    style_instruction = RESPONSE_STYLES[style_label]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Advanced settings
    st.markdown("#### ⚙️ Advanced")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
        help="Lower = more deterministic, Higher = more creative")
    max_tokens = st.slider("Max Output Tokens", 128, 2048, 512, 64)
    top_k = st.slider("Retrieved Chunks (k)", 1, 8, 3,
        help="How many document chunks to retrieve per query")
    chunk_size = st.slider("Chunk Size", 400, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, 25)
    use_history = st.toggle("Multi-turn Memory", value=True,
        help="Include previous Q&A in context for follow-up questions")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Stats
    if st.session_state.doc_stats:
        d = st.session_state.doc_stats
        st.markdown("#### 📊 Document Stats")
        st.markdown(f"""
<div class="stat-row">
  <div class="stat-card"><div class="stat-value">{d['pages']}</div><div class="stat-label">Pages</div></div>
  <div class="stat-card"><div class="stat-value">{d['chunks']}</div><div class="stat-label">Chunks</div></div>
</div>
<div class="stat-row">
  <div class="stat-card"><div class="stat-value">{st.session_state.query_count}</div><div class="stat-label">Queries</div></div>
  <div class="stat-card"><div class="stat-value">{st.session_state.total_tokens}</div><div class="stat-label">Tokens used</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history_ctx = []
            st.rerun()
    with col2:
        if st.button("↺ Reset All", use_container_width=True):
            for k in ["messages","vectorstore","doc_stats","pdf_name","total_tokens","query_count","chat_history_ctx"]:
                st.session_state[k] = [] if k in ["messages","chat_history_ctx"] else (None if k in ["vectorstore","pdf_name"] else 0) if k != "doc_stats" else {}
            st.rerun()

    # Export chat
    if st.session_state.messages:
        chat_text = "\n\n".join(
            f"[{m['role'].upper()}] {m['content']}" for m in st.session_state.messages
        )
        st.download_button("⬇ Export Chat (.txt)", chat_text,
            file_name=f"docmind_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True)


# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">⬡</div>
  <div>
    <div class="app-title">DocMind</div>
    <div class="app-subtitle">NVIDIA AI · RAG · FAISS · LangChain</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload zone ──
upload_col, info_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        label_visibility="collapsed",
        help="Upload any PDF — research papers, contracts, manuals, reports…"
    )

with info_col:
    if st.session_state.pdf_name:
        st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);
border-radius:12px;padding:14px 16px;margin-top:4px;">
  <div style="font-size:0.65rem;font-family:'DM Mono',monospace;color:var(--accent);
  letter-spacing:0.07em;text-transform:uppercase;margin-bottom:6px;">Active Doc</div>
  <div style="font-size:0.82rem;color:var(--text);font-weight:500;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{st.session_state.pdf_name}</div>
  <div style="margin-top:8px;"><span class="badge badge-green">● Ready</span></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:var(--surface);border:1px dashed var(--border);
border-radius:12px;padding:14px 16px;margin-top:4px;text-align:center;">
  <div style="font-size:1.5rem;margin-bottom:6px;">📄</div>
  <div style="font-size:0.75rem;color:var(--text-muted);">No PDF loaded</div>
</div>
""", unsafe_allow_html=True)


# ── Process PDF ──
if uploaded_file and (uploaded_file.name != st.session_state.pdf_name):
    if not api_key_input:
        st.warning("⚠️ Please enter your NVIDIA API key in the sidebar first.")
    else:
        prog = st.progress(0, text="Reading PDF…")
        with st.spinner(""):
            prog.progress(20, text="Splitting into chunks…")
            vs, stats = process_pdf(uploaded_file.read(), chunk_size, chunk_overlap)
            prog.progress(70, text="Building FAISS index…")
            time.sleep(0.3)
            prog.progress(100, text="Done!")
            time.sleep(0.3)
            prog.empty()

        st.session_state.vectorstore = vs
        st.session_state.doc_stats = stats
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.messages = []
        st.session_state.chat_history_ctx = []
        st.success(f"✓ **{uploaded_file.name}** processed — {stats['pages']} pages, {stats['chunks']} chunks")
        st.rerun()


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ── Suggested Questions ──
if st.session_state.vectorstore and not st.session_state.messages:
    st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--accent);
letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">Quick Start</div>
""", unsafe_allow_html=True)
    suggestions = [
        "Summarize the main points of this document",
        "What are the key findings or conclusions?",
        "List the most important topics covered",
        "What problem does this document address?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(f"↗  {s}", key=f"sug_{i}", use_container_width=True):
            st.session_state["prefill"] = s
            st.rerun()


# ── Chat History Display ──
chat_container = st.container()
with chat_container:
    if not st.session_state.vectorstore:
        st.markdown("""
<div class="empty-state">
  <div class="icon">⬡</div>
  <h3>Upload a PDF to begin</h3>
  <p>DocMind will index your document and let you ask<br>anything about it using NVIDIA AI models.</p>
</div>
""", unsafe_allow_html=True)
    elif not st.session_state.messages:
        st.markdown("""
<div class="empty-state">
  <div class="icon">💬</div>
  <h3>Ask your first question</h3>
  <p>Type below or use a quick-start prompt above.</p>
</div>
""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(
                msg["role"], msg["content"], msg.get("ts",""),
                sources=msg.get("sources") if msg["role"] == "assistant" else None
            )


# ── Chat Input ──
if st.session_state.vectorstore:
    prefill_val = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask anything about your PDF…", key="chat_input")
    if not user_input and prefill_val:
        user_input = prefill_val

    if user_input:
        if not api_key_input:
            st.warning("⚠️ Please enter your NVIDIA API key in the sidebar.")
            st.stop()

        ts = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": user_input, "ts": ts})

        with st.spinner("Retrieving context and generating answer…"):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            source_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n---\n\n".join([d.page_content for d in source_docs])

            msgs = build_prompt(
                user_input, context, style_instruction,
                st.session_state.chat_history_ctx, use_history
            )

            response = call_nvidia(api_key_input, selected_model, msgs, temperature, max_tokens)

            if response.status_code != 200:
                answer = f"❌ API Error {response.status_code}: {response.text[:200]}"
            else:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                used_tokens = data.get("usage", {}).get("total_tokens", 0)
                st.session_state.total_tokens += used_tokens
                st.session_state.query_count += 1

                # Update multi-turn context
                if use_history:
                    st.session_state.chat_history_ctx.append({"role":"user","content":user_input})
                    st.session_state.chat_history_ctx.append({"role":"assistant","content":answer})

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "ts": datetime.now().strftime("%H:%M"),
            "sources": source_docs,
        })
        st.rerun()
