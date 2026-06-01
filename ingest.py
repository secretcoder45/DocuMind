"""
Step 1 & 2: Load PDF and split into chunks
Step 3 & 4: Embed chunks and store in FAISS vector store
"""
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def ingest_pdf(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        print(f"Error: file not found — {pdf_path}")
        sys.exit(1)

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding chunks and saving vector store...")
    # runs locally — no API key needed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"  Vector store saved to ./{VECTORSTORE_PATH}/")
    print("Done! Run `python chat.py` to start chatting.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path/to/file.pdf>")
        sys.exit(1)
    ingest_pdf(sys.argv[1])
