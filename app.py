import streamlit as st
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="NVIDIA AI PDF Chat",
    page_icon="🟢",
    layout="wide"
)

st.markdown("""
<style>
body {background-color: #0f172a;}
.chat-user {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.chat-bot {
    background-color: #1e40af;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🟢 NVIDIA AI PDF Chat")
st.caption("Powered by NVIDIA API • LangChain • FAISS")

# -------------------------------
# Load NVIDIA API Key
# -------------------------------

NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "meta/llama3-8b-instruct"


# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("PDF Processed Successfully!")

    # -------------------------------
    # Chat Section
    # -------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):

            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer ONLY using the following context:

            {context}

            Question: {user_input}
            """

            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 512
            }

            response = requests.post(
                NVIDIA_ENDPOINT,
                headers=headers,
                json=payload
            )

            result = response.json()

            answer = result["choices"][0]["message"]["content"]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
