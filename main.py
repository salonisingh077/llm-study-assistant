from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load FAISS index
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index_multi",
    embeddings,
    allow_dangerous_deserialization=True
)
print("Index loaded!")

# Set up LLM
llm = OllamaLLM(model="tinyllama")

# Prompt that includes chat history
template = """You are a helpful study assistant. Use the context and chat history to answer the question.
If you don't know the answer, just say you don't know.

Chat History:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Store chat history manually
chat_history = []

def ask(question):
    # Format history as string
    history_str = ""
    for human, ai in chat_history:
        history_str += f"Human: {human}\nAI: {ai}\n"

    # Get context from FAISS
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build input and run chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "chat_history": history_str,
        "context": context,
        "question": question
    })

    # Save to history
    chat_history.append((question, answer))
    return answer

# Interactive chat loop
print("\n" + "="*50)
print("Study Assistant with Memory - Ready!")
print("Type 'quit' to exit")
print("="*50 + "\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        print("Goodbye!")
        break
    print("Thinking...")
    answer = ask(question)
    print(f"Assistant: {answer}\n")