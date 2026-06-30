# AI Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) pipeline built using LangChain, ChromaDB, and HuggingFace embeddings.

## Features

* Loads text from a local document
* Splits the text into manageable chunks
* Generates embeddings using the HuggingFace `all-MiniLM-L6-v2` model
* Stores embeddings locally using ChromaDB
* No paid APIs or API keys required

## Tech Stack

* Python
* LangChain
* ChromaDB
* HuggingFace Sentence Transformers

## Project Structure

```
ai-knowledge-assistant/
│
├── sample_data.txt
├── chunker.py
├── vector_store.py
├── README.md
└── .gitignore
```

## Setup

1. Create a virtual environment.
2. Install the required packages.
3. Run `vector_store.py` to generate the local ChromaDB database.

## Notes

The `chroma_db` directory and the virtual environment are excluded from version control because they are generated locally.
