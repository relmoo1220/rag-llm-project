from typing import Union
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Embedding Function
def get_embeddings(texts):
     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
     with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
     return embeddings

# Function to query ChromaDB
def query_db(question, collection, top_k=5):
     q_embeddings = get_embeddings([question])
     results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k)
     return ' '.join(results['documents'][0])

@app.get("/")
def read_root():
    question = "Did harry potter beat voldermort?"
    
    # First pass: Initial context retrieval
    context = query_db(question, collection)
    
    # First pass: Generate initial answer
    input_text = f"Given the following context, please provide a detailed answer to the question: '{question}'. Context: {context}."
    result = pipe(input_text, max_new_tokens=300, temperature=0.7, top_k=50)
    first_answer = result[0]['generated_text']

    # Second pass: Use the initial answer to refine the query
    refined_question = f"{first_answer} Can you provide more details?"
    refined_context = query_db(refined_question, collection)
    
    # Second pass: Generate the final answer with more refined context
    refined_input_text = f"Based on the refined context, elaborate on the question: '{refined_question}'. Refined Context: {refined_context}."
    refined_result = pipe(refined_input_text, max_new_tokens=300, do_sample=True, temperature=0.7, top_k=50)
    final_answer = refined_result[0]['generated_text']

    return {"initial_answer": first_answer, "final_answer": final_answer}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}