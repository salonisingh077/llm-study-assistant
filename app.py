import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="LLM Study Assistant", page_icon="📚")
st.title("📚 LLM Study Assistant")
st.caption("Upload your study notes and ask anything!")

# Load embeddings and LLM once
@st.cache_resource
def load_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = OllamaLLM(model="tinyllama", stop=["Question:", "Context:", "Chat History:"])
    return embeddings, llm

embeddings, llm = load_base()

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload Your Notes")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs", use_container_width=True):
            with st.spinner("Processing your PDFs..."):
                all_chunks = []
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load and chunk it
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    os.unlink(tmp_path)

                # Build FAISS index from uploaded docs
                vectorstore = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success(f"✅ {len(uploaded_files)} PDF(s) processed! {len(all_chunks)} chunks indexed.")

    st.divider()
    if "vectorstore" in st.session_state:
        st.success("✅ Chatbot is ready!")
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            del st.session_state.vectorstore
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.warning("⚠️ Please upload PDFs first")

# Prompt template
template = """You are a helpful study assistant.
Use the context and chat history to answer the question.
If you don't know, say you don't know.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show welcome message if no PDFs uploaded yet
if "vectorstore" not in st.session_state:
    st.info("👈 Upload your PDF study notes from the sidebar to get started!")
else:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question from your notes..."):

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Format history
                history_str = ""
                for human, ai in st.session_state.chat_history:
                    history_str += f"Human: {human}\nAI: {ai}\n"

                # Get context from uploaded docs
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)

                # Run chain
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "chat_history": history_str,
                    "context": context,
                    "question": question
                })

                st.markdown(answer)

                # Show sources
                with st.expander("📄 Sources used"):
                    for doc in docs:
                        st.write(f"**Page {doc.metadata['page']}** from `{doc.metadata['source']}`")
                        st.caption(doc.page_content[:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append((question, answer))