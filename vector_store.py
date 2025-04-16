import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
PDF_PATH = "book.pdf"
TEXT_PATH = "text.txt"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0


def extract_text_from_pdf(pdf_path: str) -> str:
    print("Extracting text from PDF...")
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""


def save_text_to_file(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved to {path}")


def load_and_split_text(file_path: str) -> list:
    print("Splitting text into chunks...")
    try:
        documents = TextLoader(file_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_documents(documents)
    except Exception as e:
        print(f"Error during splitting: {e}")
        return []


def create_vector_store(documents: list):
    print("📦 Creating vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
        print("Vector store saved.")
    except Exception as e:
        print(f"Vector store creation failed: {e}")


def main():
    text = extract_text_from_pdf(PDF_PATH)
    if not text:
        return

    save_text_to_file(text, TEXT_PATH)
    docs = load_and_split_text(TEXT_PATH)
    if docs:
        create_vector_store(docs)


if __name__ == "__main__":
    main()
