from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import pandas as pd
import re

# Dataset paths
dataset = [
    {"path": "./dataset/financial-statements.csv"}
]
persist_directory = "./chroma_persistence"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")

# Function to read CSV and normalize text row by row with headers
def extract_text_from_csv(csv_path):
    rows_text = []
    try:
        df = pd.read_csv(csv_path)
        headers = df.columns.tolist()  # Get the headers

        for index, row in df.iterrows():
            # Combine the header and corresponding row values into a single string
            row_text = ' '.join(f"{header}: {row[header]}" for header in headers)
            rows_text.append(row_text)  # Append the formatted row text
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return rows_text  # Return a list of row texts

# Function to normalize text
def normalize_text(text):
    # Replace newlines with a space and remove extra spaces
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading/trailing whitespace

# Embedding Function
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Initialize Chroma DB with persistence
client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Create a collection (this will be saved in the persistent storage)
collection = client.create_collection(name="document_collection")

# Loop over dataset to process each CSV
for doc in dataset:
    csv_path = doc["path"]

    # Extract text from the CSV row by row with headers
    rows_text = extract_text_from_csv(csv_path=csv_path)

    # Process each row
    for i, row_text in enumerate(rows_text):
        # Normalize the text for each row
        normalized_text = normalize_text(row_text)

        # Get embeddings for the row text directly
        embedding = get_embeddings([normalized_text])  # Get the embedding for the entire row

        # Create a unique ID for each row
        unique_id = f"row_{i}"

        # Insert the row and its embedding into ChromaDB
        collection.add(
            ids=[unique_id],
            documents=[normalized_text],     # Add the normalized row text
            embeddings=embedding.numpy()     # Add the corresponding embedding
        )
        print(unique_id)

print("ChromaDB persistence complete!")
