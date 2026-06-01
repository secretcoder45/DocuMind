# PDF Q&A Chatbot — LangChain + Claude + RAG

Chat with any PDF using Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions in plain English, and get answers grounded in the document with source page references.

---

## Tech Stack

| Component | Library / Model |
|---|---|
| PDF Loading | `PyPDFLoader` (langchain-community) |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Vector Store | `FAISS` (Facebook AI Similarity Search) |
| LLM | `claude-haiku-4-5` (Anthropic) |
| Memory | `ConversationBufferMemory` |
| Orchestration | `ConversationalRetrievalChain` (LangChain) |

---

## Project Structure

```
pdf-reader_langchain/
├── ingest.py        ← Phase 1: Load PDF → chunk → embed → save index
├── chat.py          ← Phase 2: Load index → retrieve → generate answer
├── requirements.txt
├── .env.example     ← Copy to .env and add your Anthropic key
├── .gitignore
└── vectorstore/     ← Created after running ingest.py (git-ignored)
```

---

## Architecture Overview

The system runs in two distinct phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — INGESTION  (run once per PDF)     ingest.py          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   your_file.pdf  ──►  Load  ──►  Chunk  ──►  Embed  ──►  Save  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — QUERYING  (every question)        chat.py            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Question  ──►  Condense  ──►  Embed  ──►  Retrieve  ──►  LLM │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Ingestion (`ingest.py`)

Run this once to process your PDF and build the searchable index.

```
your_file.pdf
      │
      ▼
┌─────────────────────────────────────┐
│  PyPDFLoader                        │
│                                     │
│  Reads each page of the PDF into a  │
│  LangChain Document object:         │
│  { page_content: "...", metadata:   │
│    { page: 0, source: "file.pdf" }} │
└──────────────────┬──────────────────┘
                   │  e.g. 30 pages → 30 Documents
                   ▼
┌──────────────────────────────────────────────────────┐
│  RecursiveCharacterTextSplitter                      │
│  chunk_size=1000  |  chunk_overlap=200               │
│                                                      │
│  Page text                                           │
│  ┌────────────────────────────────────────────┐      │
│  │← ─ ─ ─ ─ chunk 1 (1000 chars) ─ ─ ─ ─ ─►│      │
│  │                          ├──overlap──┤     │      │
│  │                ├─ ─ ─ ─ ─ chunk 2 ─ ─ ─ ─►      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  Overlap ensures sentences at boundaries are not     │
│  silently cut and lost from both chunks.             │
└──────────────────┬───────────────────────────────────┘
                   │  e.g. 30 pages → ~120 chunks
                   ▼
┌──────────────────────────────────────────────────────┐
│  HuggingFaceEmbeddings  (all-MiniLM-L6-v2)          │
│  Runs LOCALLY — no API key, no cost                  │
│                                                      │
│  "Payment is due in 30 days"                         │
│           │                                          │
│           ▼                                          │
│   [0.12, -0.84, 0.33, 0.07, ...]  ← 384 numbers    │
│                                                      │
│  Each number encodes a dimension of meaning.         │
│  Similar sentences produce similar vectors.          │
└──────────────────┬───────────────────────────────────┘
                   │  ~120 vectors  (each 384-dimensional)
                   ▼
┌──────────────────────────────────────────────────────┐
│  FAISS Vector Store                                  │
│  (Facebook AI Similarity Search)                     │
│                                                      │
│   Vector 1  ●──────────────────────────              │
│   Vector 2  ●───────────────────                     │
│   Vector 3  ●──────────────────────────────          │
│   ...        (flat index on disk)                    │
│                                                      │
│  Saved to  ./vectorstore/                            │
│  Enables millisecond nearest-neighbour lookups.      │
└──────────────────────────────────────────────────────┘
```

---

## Phase 2 — Querying (`chat.py`)

Runs every time you ask a question. Four steps happen internally.

```
You type: "What are the payment terms?"
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1 — Conversation Memory                                │
│  ConversationBufferMemory                                    │
│                                                              │
│  history = [                                                 │
│    Human: "Summarise the contract"                           │
│    AI:    "The contract covers..."                           │
│    Human: "What are the payment terms?"   ← current         │
│  ]                                                           │
│                                                              │
│  Keeps the full dialogue so follow-up questions work.        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2 — Question Condensation  (LLM call #1)               │
│                                                              │
│  Sends history + question to Claude:                         │
│  "Rewrite this as a single standalone question."             │
│                                                              │
│  "What are the payment terms?"  (after context)             │
│            ▼                                                 │
│  "What are the payment terms specified in the contract?"     │
│                                                              │
│  Why? So vague follow-ups like "what about clause 3?"        │
│  become fully self-contained search queries.                 │
└──────────────────────────┬───────────────────────────────────┘
                           │  standalone question
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3 — Embed + Retrieve                                   │
│                                                              │
│  Same local HuggingFace model embeds the question:          │
│  "What are the payment terms..." → [0.09, -0.77, ...]       │
│                                                              │
│  FAISS finds the 4 most similar chunk vectors (k=4):        │
│                                                              │
│  Query ●                                                     │
│         ╲                                                    │
│  Chunk1  ●  ← closest  (similarity: 0.91)                   │
│  Chunk2  ●  ← 2nd      (similarity: 0.88)                   │
│  Chunk7  ●  ← 3rd      (similarity: 0.84)                   │
│  Chunk9  ●  ← 4th      (similarity: 0.81)                   │
│  Chunk3  ○  far away   (similarity: 0.31)  ← not retrieved  │
│                                                              │
│  Only those 4 chunks are sent to the LLM — not the whole    │
│  PDF. This keeps cost low and answers precise.              │
└──────────────────────────┬───────────────────────────────────┘
                           │  4 relevant text chunks
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4 — Generation  (LLM call #2)                          │
│  Claude Haiku  (claude-haiku-4-5, temperature=0)             │
│                                                              │
│  Prompt sent to Claude:                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Use the context below to answer the question.          │  │
│  │                                                        │  │
│  │ Context:                                               │  │
│  │   [chunk 1 text]                                       │  │
│  │   [chunk 2 text]                                       │  │
│  │   [chunk 7 text]                                       │  │
│  │   [chunk 9 text]                                       │  │
│  │                                                        │  │
│  │ Question: What are the payment terms...?               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  temperature=0 → deterministic, fact-grounded answers.       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
Assistant: "Payment is due within 30 days of invoice date.
            A 2% late fee applies after that period."
  [Sources: pages [4, 7]]
```

---

## Data Flow Summary

```
                      INGESTION (once)
  ┌──────┐   pages   ┌───────┐  chunks  ┌─────────┐  vectors  ┌───────┐
  │ PDF  │ ────────► │Loader │ ───────► │Splitter │ ────────► │ FAISS │
  └──────┘           └───────┘          └─────────┘           └───────┘
                                             ▲                     │
                                    HuggingFace                 saved to
                                    Embeddings                  ./vectorstore
                                    (local)

                      QUERYING (each question)
  ┌──────────┐        ┌──────────┐        ┌───────┐        ┌────────┐
  │ Question │──────► │ Condense │──────► │ FAISS │──────► │ Claude │
  └──────────┘  LLM   └──────────┘ embed  │retrieve        └────────┘
       ▲         #1         │       local  └───────┘            │
       │                    │                  k=4 chunks        │
  Memory adds          standalone                               ▼
  chat history         question                            Answer + pages
```

---

## Why RAG?

| Challenge | How RAG solves it |
|---|---|
| LLMs have a token limit — a 100-page PDF won't fit | Only the 4 most relevant chunks are sent, not the whole PDF |
| Keyword search misses synonyms and paraphrases | Vector embeddings capture *meaning*, so "invoice due date" matches "payment deadline" |
| Follow-up questions lose context | `ConversationBufferMemory` keeps the full chat history |
| Vague references like "what about that?" | Question condensation rewrites them into standalone queries before searching |
| Cloud embedding APIs cost money | `all-MiniLM-L6-v2` runs 100% locally for free |

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Anthropic API key
cp .env.example .env
# Edit .env and paste your key: ANTHROPIC_API_KEY=sk-ant-...

# 3. Ingest your PDF (run once, or again for a new PDF)
python ingest.py path/to/your.pdf

# 4. Start chatting
python chat.py
```

**Example session:**
```
PDF Q&A Chatbot — type 'quit' to exit

You: What is this document about?
Assistant: This document is a software licensing agreement between...
  [Sources: pages [1, 2]]

You: What are the payment terms?
Assistant: Payment is due within 30 days of invoice date...
  [Sources: pages [4, 7]]

You: quit
Bye!
```
