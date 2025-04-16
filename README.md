# Quantum Computing RAG App

A simple Retrieval-Augmented Generation (RAG) application designed for answering questions related to quantum computing. It leverages a local document-based vector store to enhance LLM-based responses using your own data.

## 🧠 What It Does

This app allows you to:

- Load documents (e.g., quantum computing books or notes)
- Generate vector embeddings from text
- Store and query them via a local vector store (ChromaDB)
- Use the most relevant chunks as context for generating accurate, informed responses via an LLM

## 📂 Project Structure

```
├── main.py             # Runs the app and query logic
├── vector_store.py     # Handles ChromaDB vector indexing
├── chroma_db/          # Directory where the vector DB is stored
├── book.pdf            # Sample document
├── text.txt            # Sample raw text input
├── .gitignore
```

## ⚙️ How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/shathwik33/Quantum-Computing-Rag-App.git
   cd Quantum-Computing-Rag-App
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   python main.py
   ```

You'll be prompted to ask a question related to quantum computing. The app fetches relevant document chunks and generates a response using an LLM.

## ✅ Requirements

- Python 3.8+
- `chromadb`
- `langchain`
- `PyPDF2` (if using PDFs)
- An OpenAI key (or other LLM backend) if extended for full RAG

## 🚀 Future Improvements

- Add UI for ease of use
- Support for more file formats
- Integration with Hugging Face or OpenAI API
