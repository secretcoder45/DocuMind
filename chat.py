"""
Step 5: Retrieve relevant chunks and generate an answer (RAG pipeline)
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def load_chain() -> ConversationalRetrievalChain:
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            "No vector store found. Run `python ingest.py <your.pdf>` first."
        )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def main() -> None:
    print("PDF Q&A Chatbot — type 'quit' to exit\n")
    try:
        chain = load_chain()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        result = chain.invoke({"question": question})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        print(f"\nAssistant: {answer}")

        if sources:
            pages = sorted({(doc.metadata.get("page", 0) + 1) for doc in sources})
            print(f"  [Sources: pages {pages}]")
        print()


if __name__ == "__main__":
    main()
