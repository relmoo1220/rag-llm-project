from typing import Union
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

app = FastAPI()

persist_directory = "./chroma_persistence"

# Initialize Chroma DB with persistence
client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Load the collection by name
collection = client.get_collection("document_collection")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")

pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Embedding Function
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to query ChromaDB
def query_db(question, collection, top_k=2):
    q_embeddings = get_embeddings([question])
    results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k)
    return ' '.join(results['documents'][0])

@app.get("/")
def read_root():
    question = "In 2013, what is AAPL's revenue?"

    # First pass: Initial context retrieval
    context = query_db(question, collection)

    # Define the template for financial metric units
    template = (
        "Please note the following units for the financial metrics: "
        "Market Cap is in billions of USD, "
        "Revenue is in millions of USD, "
        "Gross Profit is in millions of USD, "
        "Net Income is in millions of USD, "
        "Earning Per Share is in USD. "
        "EBITDA is in millions of USD"
        "Share Holder Equity is in millions of USD"
        "Cash Flow from Operating is in millions of USD"
        "Cash Flow from Investing is in millions of USD"
        "Cash Flow from Financial Activities is in millions of USD"
        "Free Cash Flow per Share is in USD"
        "Now, based on the provided context, please answer the question and include the units."
    )

    # First pass: Generate initial answer
    input_text = (
        f"{template}\n\nContext: {context}\n\nQuestion: {question}"
    )
    result = pipe(input_text, max_new_tokens=300, do_sample=True, temperature=0.5, top_k=50)
    answer = result[0]['generated_text']
    print(context)
    print(answer)

    return {"answer": answer}
