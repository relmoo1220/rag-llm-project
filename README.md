# Simple Finance RAG LLM ChatBot

A simple chatbot that enables users to query financial statements of major companies from 2009 to 2022. This Retrieval-Augmented Generation (RAG) application integrates a SvelteKit frontend, FastAPI backend, and ChromaDB as a vector database to power efficient and relevant financial queries. Using Hugging Face’s Transformers, the app leverages a question-answering model to provide accurate responses based on extracted document data.

[![FastAPI](https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white)](#)
[![SvelteKit](https://img.shields.io/badge/SvelteKit-%23f1413d.svg?logo=svelte&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)

### Technology Stack

- Frontend: SvelteKit for an interactive and responsive UI.
- Backend: FastAPI for fast, asynchronous API handling.
- Database: ChromaDB as a high-performance vector store.
- Model: Hugging Face Transformers for financial Q&A capabilities.

### Installation

Creating a virtual environment

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Starting the backend

```
cd .\fastapi-backend\
fastapi dev main.py
```

Starting the frontend

```
cd .\frontend\
npm install
npm run dev
```

### Current Models In-Use

- **Alibaba-NLP/gte-large-en-v1.5**: A sentence similarity model used to generate embeddings for the financial documents. These embeddings are stored in ChromaDB, enabling the chatbot to perform efficient, contextually relevant search and retrieval of information.

- **deepset/roberta-base-squad2**: A question-answering model fine-tuned on SQuAD2.0, used to interpret user queries. It processes relevant sections retrieved from ChromaDB to generate accurate responses to finance-related questions, providing an interactive, LLM-powered Q&A experience.
