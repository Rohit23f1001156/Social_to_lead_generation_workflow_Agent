import json
import os
from dotenv import load_dotenv
from utils.embeddings import RESTGeminiEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def ingest_knowledge_base():
    """
    Loads the JSON, converts to string format, and creates a FAISS vector index.
    """
    print("Starting ingestion process...")
    
    # 1. Load JSON data
    try:
        with open("data/knowledge_base.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: knowledge_base.json not found in data/ folder.")
        return

    # 2. Convert structured data to text chunks
    docs = []
    
    # Add Metadata to each chunk
    for plan in data.get("plans", []):
        features_str = ", ".join(plan["features"])
        text = f"{plan['name']} costs {plan['price']} and includes: {features_str}."
        docs.append(Document(page_content=text, metadata={"type": "plan", "name": plan["name"]}))
        
    for policy in data.get("policies", []):
        text = f"Company Policy: {policy}"
        docs.append(Document(page_content=text, metadata={"type": "policy"}))

    # Implementation of Recursive Splitting for interview depth
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print(f"Created {len(split_docs)} structured document chunks.")

    # 3. Setup Embeddings
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY missing.")
        return
        
    # Use custom REST embeddings to bypass gRPC Deadline Exceeded issues
    embeddings = RESTGeminiEmbeddings(google_api_key=api_key, model="models/gemini-embedding-001")

    # 4. Create and save FAISS index
    print("Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    
    # Create rag directory if it doesn't exist
    os.makedirs("rag", exist_ok=True)
    vectorstore.save_local("rag/faiss_index")
    print("Ingestion complete! FAISS index saved to rag/faiss_index/")

if __name__ == "__main__":
    ingest_knowledge_base()