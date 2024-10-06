from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import fitz
import re
import string

dataset = "./dataset/harrypotter.pdf"
persist_directory = "./chroma_persistence"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

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
    chunk_size=512,         # Set your desired chunk size
    chunk_overlap=20,       # Set the overlap between chunks
    length_function=len     # Define the length function to use
)

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

# Extract and normalize text
non_normalized_text = extract_text_from_pdf(pdf_path=dataset)
normalized_text = normalize_text(non_normalized_text)

# Create chunks using LangChain's RecursiveCharacterTextSplitter
chunks = text_splitter.split_text(normalized_text)

# Get embeddings for each chunk and insert them into ChromaDB
for i, chunk in enumerate(chunks):
    embedding = get_embeddings([chunk])  # Get the embedding for the chunk
    # Create a unique ID for each chunk
    unique_id = f"harrypotter:chunk_{i}"
    # Insert the chunk and its embedding into ChromaDB
    collection.add(
        ids=[unique_id],
        documents=[chunk],              # Add the text chunk
        embeddings=embedding.numpy(),   # Add the corresponding embedding
        metadatas=[{"source": dataset}]  # Metadata for the chunk
    )
    print(unique_id)

print("ChromaDB persistence complete!")