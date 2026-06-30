# AI Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) pipeline built using LangChain, ChromaDB, and HuggingFace embeddings.

## Features

- Loads text from a local document
- Splits text into chunks
- Generates embeddings using HuggingFace Sentence Transformers
- Stores embeddings locally in ChromaDB
- Works without paid APIs

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers

## Files

- `sample_data.txt`
- `chunker.py`
- `vector_store.py`