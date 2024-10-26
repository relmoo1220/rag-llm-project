from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, pipeline
import json
import torch
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],  # Replace with the frontend origin
    allow_credentials=True,
    allow_methods=['*'],  # Allow all methods (GET, POST, etc.)
    allow_headers=['*'],  # Allow all headers
)

class QueryRequest(BaseModel):
    query: str

persist_directory = './chroma_persistence'

# Initialize Chroma DB with persistence
client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Load the collection by name
collection = client.get_collection('document_collection')

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Embedding Function
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to query ChromaDB
def query_db(question, collection, top_k=2):
    q_embeddings = get_embeddings([question])
    results = collection.query(query_embeddings=q_embeddings.numpy().tolist(), n_results=top_k)

    # Extract documents and parse JSON strings
    documents = []
    for doc_list in results['documents']:
        for doc in doc_list:  # Each doc is a JSON string
            documents.append(json.loads(doc))  # Parse the JSON string

    # Format the documents into a readable context
    context = ""
    for doc in documents:
        context += (
            f"Year: {doc['Year']}, "
            f"Company: {doc['Company']}, "
            f"Category: {doc['Category']}, "
            f"Market Cap (B USD): {doc['Market Cap (B USD)']} billion USD, "
            f"Revenue: {doc['Revenue']} million USD, "
            f"Gross Profit: {doc['Gross Profit']} million USD, "
            f"Net Income: {doc['Net Income']} million USD, "
            f"Earning Per Share: {doc['Earning Per Share']} USD, "
            f"EBITDA: {doc['EBITDA']} million USD, "
            f"Share Holder Equity: {doc['Share Holder Equity']} million USD, "
            f"Cash Flow from Operating: {doc['Cash Flow from Operating']} million USD, "
            f"Cash Flow from Investing: {doc['Cash Flow from Investing']} million USD, "
            f"Cash Flow from Financial Activities: {doc['Cash Flow from Financial Activities']} million USD, "
            f"Current Ratio: {doc['Current Ratio']}, "
            f"Debt/Equity Ratio: {doc['Debt/Equity Ratio']}, "
            f"ROE: {doc['ROE']}, "
            f"ROA: {doc['ROA']}, "
            f"ROI: {doc['ROI']}, "
            f"Net Profit Margin: {doc['Net Profit Margin']} %, "
            f"Free Cash Flow per Share: {doc['Free Cash Flow per Share']} USD, "
            f"Return on Tangible Equity: {doc['Return on Tangible Equity']}, "
            f"Number of Employees: {doc['Number of Employees']}, "
            f"Inflation Rate (US): {doc['Inflation Rate (US)']}\n"
        )

    return context.strip()  # Return the formatted context


@app.post("/query")
def query_llm(request: QueryRequest):
    question = request.query  # Get the user query from the request body

    context = query_db(question, collection)

    print("Context:", context)  # Output the structured context

    # Directly use the question and context in the question-answering model
    result = pipe(question=question, context=context)

    # Extract the answer from the result
    answer = result['answer']

    return {"answer": answer}
