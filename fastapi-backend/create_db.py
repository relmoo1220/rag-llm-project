from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import pandas as pd
import re
import json

# Dataset paths
dataset = [
    {"path": "./dataset/financial-statements.csv"}
]
persist_directory = "./chroma_persistence"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

# Function to read CSV and convert it to a DataFrame
def load_csv_to_dataframe(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None  # Return None if there is an error

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

    # Load the CSV data into a DataFrame
    df = load_csv_to_dataframe(csv_path=csv_path)

    if df is not None:
        # Process each row in the DataFrame
        for i, row in df.iterrows():
            # Create a dictionary for the current row
            row_data = {
                "Year": int(row["Year"]),  # Ensure Year is an integer
                "Company": row["Company"],
                "Category": row["Category"],
                "Market Cap (B USD)": float(row["Market Cap(in B USD)"]),
                "Revenue": float(row["Revenue"]),
                "Gross Profit": float(row["Gross Profit"]),
                "Net Income": float(row["Net Income"]),
                "Earning Per Share": float(row["Earning Per Share"]),
                "EBITDA": float(row["EBITDA"]),
                "Share Holder Equity": float(row["Share Holder Equity"]),
                "Cash Flow from Operating": float(row["Cash Flow from Operating"]),
                "Cash Flow from Investing": float(row["Cash Flow from Investing"]),
                "Cash Flow from Financial Activities": float(row["Cash Flow from Financial Activities"]),
                "Current Ratio": float(row["Current Ratio"]),
                "Debt/Equity Ratio": float(row["Debt/Equity Ratio"]),
                "ROE": float(row["ROE"]),
                "ROA": float(row["ROA"]),
                "ROI": float(row["ROI"]),
                "Net Profit Margin": float(row["Net Profit Margin"]),
                "Free Cash Flow per Share": float(row["Free Cash Flow per Share"]),
                "Return on Tangible Equity": float(row["Return on Tangible Equity"]),
                "Number of Employees": int(row["Number of Employees"]),
                "Inflation Rate (US)": float(row["Inflation Rate(in US)"])
            }

            # Convert the row data dictionary to a JSON string
            normalized_text = normalize_text(json.dumps(row_data))  # Convert dictionary to JSON string

            # Get embeddings for the normalized text
            embedding = get_embeddings([normalized_text])

            # Create a unique ID for each row
            unique_id = f"row_{i}"

            # Insert the row and its embedding into ChromaDB
            collection.add(
                ids=[unique_id],
                documents=[normalized_text],     # Add the normalized JSON string
                embeddings=embedding.numpy()     # Add the corresponding embedding
            )
            print(f"Inserted: {unique_id}")
            print(row_data)

print("ChromaDB persistence complete!")
