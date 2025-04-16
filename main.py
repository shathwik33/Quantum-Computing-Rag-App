import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Constants
LLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=LLM_MODEL,
)

# Embeddings & Vectorstore
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt Template
rag_prompt = PromptTemplate(
    input_variables=["retrieved_chunks", "conversation", "user_prompt"],
    template="""
You are a helpful assistant specialized in quantum computing.
Use the context below only if it's relevant. If not, say: "I'm unable to answer that."

Context:
{retrieved_chunks}

Conversation:
{conversation}

Question:
{user_prompt}

Answer:
""",
)

# In-memory chat history
chat_history = []


def format_chat_history(history) -> str:
    return "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in history
    )


def retrieve_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs) if docs else ""


def build_prompt(user_input: str, context: str, history: str) -> str:
    return rag_prompt.format(
        retrieved_chunks=context, conversation=history, user_prompt=user_input
    )


def generate_response(user_input: str) -> str:
    context = retrieve_context(user_input)
    history = format_chat_history(chat_history)
    full_prompt = build_prompt(user_input, context, history)

    response = llm.invoke(
        [SystemMessage("You're a quantum AI assistant."), HumanMessage(full_prompt)]
    )

    chat_history.append(HumanMessage(user_input))
    chat_history.append(AIMessage(response.content))

    return response.content


def main():
    print(
        "Quantum Assistant RAG Chatbot\nType 'exit' to quit or 'clear' to reset memory.\n"
    )
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Bye!")
                break
            elif user_input.lower() == "clear":
                chat_history.clear()
                print("Chat history cleared.\n")
                continue

            reply = generate_response(user_input)
            print(f"{reply}\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
