# ⬡ DocMind

**RAG-powered PDF chat using NVIDIA API Catalog, FAISS, LangChain, and Streamlit.**

No GPU. No Docker. No database. Runs free on Streamlit Cloud.

---

## What it actually does

You upload a PDF. It chunks it, embeds it with `all-MiniLM-L6-v2` (384-dim, runs on CPU fine), builds a FAISS flat index in memory, and wires that retriever into a LangChain prompt that hits NVIDIA's hosted inference API. You ask questions. It pulls the top-k most relevant chunks, injects them as context, and gets an answer back.

That's it. No magic. Standard RAG pattern — the value is in the defaults being sane, the UI not being garbage, and the whole thing deploying in one click.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM backend | NVIDIA API Catalog | Free credits, OpenAI-compatible endpoint, access to Llama 3 / Mixtral / Gemma without running anything locally |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, 384-dim, CPU-only, good enough for most retrieval tasks |
| Vector store | FAISS (flat, in-memory) | No infra. For documents under ~500 pages the flat index is fine. Swap to IVF if you're chunking large corpora. |
| Orchestration | LangChain community | Mostly for `PyPDFLoader` and `RecursiveCharacterTextSplitter`. The actual API call is raw `requests` — no need to pull in `langchain-openai` for something this simple. |
| UI | Streamlit | Fastest path to something that looks reasonable without writing React |

---

## Supported models

Swap between these from the sidebar — all hosted on NVIDIA's API Catalog, no local resources needed:

- `meta/llama3-8b-instruct` — default, fast, good for most docs
- `meta/llama3-70b-instruct` — better reasoning, higher latency
- `mistralai/mixtral-8x7b-instruct-v0.1` — good for structured outputs
- `google/gemma-7b` — lighter, decent for simple Q&A
- `microsoft/phi-3-mini-128k-instruct` — 128k context window, useful for long docs if you increase chunk sizes

---

## Features

- **Multi-turn memory** — last 3 Q&A turns included in context. Togglable. Don't leave it on for unrelated follow-ups, it'll pollute the context window.
- **Response styles** — Precise, Detailed, Bullet Points, ELI5. These are system prompt modifiers, nothing fancy.
- **Source attribution** — every answer expands to show the exact chunks retrieved with page numbers. Useful for verifying the model isn't hallucinating.
- **Tunable retrieval** — k (chunks retrieved), chunk size, chunk overlap all exposed as sliders. The defaults (k=3, size=1000, overlap=150) work well for standard documents. For dense technical PDFs bump k to 5 and drop chunk size to 600.
- **Token counter** — live count of tokens consumed in session. Useful for staying within free tier limits.
- **Chat export** — downloads the full session as `.txt`. Timestamped filename.

---

## Getting started

### 1. Clone

```bash
git clone https://github.com/yourname/docmind.git
cd docmind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download `all-MiniLM-L6-v2` (~90MB) and cache it locally. Subsequent runs are instant.

### 3. Get an NVIDIA API key

Go to [build.nvidia.com](https://build.nvidia.com), sign in, and generate an API key. Free tier gives you enough credits to experiment comfortably.

### 4. Set the key

**Local:**
```bash
# .streamlit/secrets.toml
NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxx"
```

Or just paste it directly in the sidebar input — there's a manual key field for dev use.

**Streamlit Cloud:**
Settings → Secrets → add `NVIDIA_API_KEY = "nvapi-..."`.

### 5. Run

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public or private, both work)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point it at `app.py`
4. Under **Advanced settings → Secrets**, add your API key
5. Deploy

First deploy takes ~3-4 minutes (installing `sentence-transformers` is slow). Cold starts after that are ~30s.

---

## Project structure

```
docmind/
├── app.py              # everything — intentionally single-file for portability
├── requirements.txt
├── .streamlit/
│   └── secrets.toml    # local only, never commit this
└── README.md
```

Single-file by design. If you're extending this into something real, the natural split is:

```
docmind/
├── app.py
├── core/
│   ├── ingest.py       # PDF loading, chunking, FAISS indexing
│   ├── retriever.py    # vectorstore wrapper, search logic
│   └── llm.py          # NVIDIA API client, prompt construction
├── ui/
│   └── components.py   # reusable Streamlit HTML components
└── config.py           # model registry, defaults
```

---

## Known limitations and honest notes

**FAISS is in-memory.** Every time a user uploads a PDF it gets re-indexed from scratch. For a production setup you'd want to persist the index (FAISS supports `save_local` / `load_local`) and key it by a hash of the file content. Not done here because it adds complexity without being necessary for the use case.

**No streaming.** The `stream=False` flag is set intentionally — Streamlit's `st.write_stream` works but requires careful handling of the rerun cycle to avoid duplicate renders. Worth adding if latency on large responses bothers you.

**Chunk size is a real tradeoff.** Bigger chunks = more context per retrieval hit but noisier signal. Smaller chunks = tighter retrieval but you might split sentences mid-thought. The overlap parameter exists to mitigate boundary artifacts. There's no universal correct answer here — it depends on the document structure. Dense academic papers do better with smaller chunks (~500 chars, overlap ~100). Manuals and reports do fine at the defaults.

**Multi-turn context is naïve.** It appends the last 3 turns verbatim to the messages array. This works for simple follow-up questions but breaks down if the conversation drifts topic. A proper implementation would run a query rewriting step to merge the follow-up with prior context before retrieval. That adds a second LLM call — trading latency for accuracy.

**The system prompt is English-only tuned.** If your documents are in another language you'll want to update the system message and probably swap the embedding model for a multilingual one (`paraphrase-multilingual-MiniLM-L12-v2` is a decent drop-in, same vector dim).

---

## Potential extensions

Things worth building if this becomes a real tool:

- **FAISS persistence** — hash the uploaded file, check if index exists before re-embedding. Saves ~5-10s per upload.
- **Multi-document support** — merge multiple FAISS indexes or build a single index with document-level metadata filtering.
- **Query rewriting** — small LLM call to reformulate follow-up questions before retrieval. Meaningfully improves multi-turn accuracy.
- **Reranker** — run a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the retrieved chunks before injecting context. Helps when k is high and some chunks are weakly relevant.
- **Confidence scoring** — FAISS returns L2 distances, you can expose these as relevance scores and warn when the retrieval signal is weak (i.e. the model is likely to hallucinate).
- **Table and figure extraction** — `PyPDFLoader` is text-only. For PDFs with important tables, swap to `pdfplumber` or `unstructured` for better extraction fidelity.

---

## Requirements

```
streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-text-splitters>=0.2.0
pypdf>=4.0.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
requests>=2.31.0
```

Python 3.10+ recommended. Tested on 3.10 and 3.11.

---

## License

MIT.
