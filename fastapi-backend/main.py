from typing import Union
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import chromadb
import fitz
import re
import string

app = FastAPI()

datasets = [
    "./dataset/monopoly.pdf",
]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to normalize text
def normalize_text(text):
    # Replace newlines with a space and remove extra spaces
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)  # Replace multiple spaces with a single space
    
    # Remove punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove any extra spaces again, just in case
    text = re.sub(' +', ' ', text)
    
    return text.strip()  # Remove leading/trailing whitespace

# Initialize LangChain's RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,         # Set your desired chunk size
    chunk_overlap=20,       # Set the overlap between chunks
    length_function=len     # Define the length function to use
)

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

# Initialize Chroma DB
client = chromadb.Client()
collection = client.create_collection("document_collection")

# Process each dataset
for i, dataset in enumerate(datasets):
    # Extract and normalize text
    non_normalized_text = extract_text_from_pdf(pdf_path=dataset)
    normalized_text = normalize_text(non_normalized_text)

    # Create chunks using LangChain's RecursiveCharacterTextSplitter
    chunks = text_splitter.split_text(normalized_text)

    # Get embeddings for each chunk and insert them into ChromaDB
    for j, chunk in enumerate(chunks):
        embedding = get_embeddings([chunk])  # Get the embedding for the chunk
        # Create a unique ID for each chunk
        unique_id = f"manual_{i + 1}:chunk_{j}"
        # Insert the chunk and its embedding into ChromaDB
        collection.add(
            ids=unique_id,
            documents=[chunk],              # Add the text chunk
            embeddings=embedding.numpy(),    # Add the corresponding embedding
            metadatas=[{"source": dataset}]  # Metadata for the chunk
        )

@app.get("/")
def read_root():
    question = "When does a player go to jail in monopoly?"
    context = query_db(question, collection)

    input_text = f"Given the following context, please provide a elaborate answer to the question: '{question}'. Context: {context} Please include examples and explanations where relevant."

    result = pipe(input_text, max_new_tokens=300, temperature=0.7, top_k=50)

    # Extract the generated answer
    answer = result[0]['generated_text']

    return {"items": answer}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}