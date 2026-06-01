# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Requires an `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

```bash
# Ingest a PDF (run once per PDF, overwrites ./vectorstore/)
python ingest.py path/to/your.pdf

# Start the chat REPL
python chat.py
```

Type `quit`, `exit`, or `q` to leave the chat session.

## Architecture

Two-phase pipeline — ingestion is a one-time offline step; querying runs per question.

**Phase 1 — `ingest.py`**  
Loads a PDF via `PyPDFLoader`, splits pages into overlapping 1000-char chunks (`RecursiveCharacterTextSplitter`, overlap=200), embeds them locally with `all-MiniLM-L6-v2` (HuggingFace, no API key needed), and saves the FAISS index to `./vectorstore/`.

**Phase 2 — `chat.py`**  
Loads the saved FAISS index, then for each question runs `ConversationalRetrievalChain`:
1. **Condense** — LLM rewrites the question + chat history into a standalone query (LLM call #1).
2. **Retrieve** — question is embedded locally; FAISS returns the 4 nearest chunks (`k=4`).
3. **Generate** — Claude Haiku (`claude-haiku-4-5-20251001`, `temperature=0`) answers from those 4 chunks (LLM call #2).

`ConversationBufferMemory` keeps the full chat history so follow-up questions resolve correctly. Source page numbers are extracted from chunk metadata and printed after each answer.

The `vectorstore/` directory is git-ignored; re-run `ingest.py` whenever you switch to a different PDF.
