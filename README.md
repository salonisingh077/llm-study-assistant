\# 📚 LLM Study Assistant



A fully offline RAG-based chatbot that answers questions from your own study notes using semantic search.



\## 🚀 Features

\- Upload any PDF study notes directly from the browser

\- Semantic search across all uploaded documents using FAISS

\- Conversational memory — remembers previous questions

\- Shows source pages used for each answer

\- 100% free and offline — powered by Ollama (no API key needed)



\## 🛠️ Tech Stack

Python, LangChain, FAISS, HuggingFace Embeddings, Ollama, Streamlit



\## ⚙️ Setup



1\. Clone the repo

&#x20;  git clone https://github.com/salonisingh077/llm-study-assistant.git

&#x20;  cd llm-study-assistant



2\. Create virtual environment

&#x20;  python -m venv venv

&#x20;  venv\\Scripts\\activate



3\. Install dependencies

&#x20;  pip install langchain langchain-community langchain-huggingface

&#x20;  pip install langchain-ollama langchain-text-splitters

&#x20;  pip install faiss-cpu pypdf python-dotenv streamlit sentence-transformers



4\. Install Ollama from ollama.com and pull model

&#x20;  ollama pull tinyllama



5\. Run the app

&#x20;  streamlit run app.py



\## 💡 How It Works

1\. Upload your PDF notes from the sidebar

2\. Click Process PDFs — builds a FAISS vector index

3\. Ask any question in the chat

4\. The app finds the most relevant chunks from your notes

5\. Ollama LLM generates an answer based on your documents

