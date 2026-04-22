import os
from dotenv import load_dotenv
from utils.embeddings import RESTGeminiEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def retrieve_context(query: str) -> str:
    """
    Loads the local FAISS index and retrieves top relevant chunks for the query.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    # Use custom REST embeddings to bypass gRPC Deadline Exceeded issues
    embeddings = RESTGeminiEmbeddings(google_api_key=api_key, model="models/gemini-embedding-001")
    
    try:
        # allow_dangerous_deserialization=True is required by newer LangChain versions to load local files
        vectorstore = FAISS.load_local(
            "rag/faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Perform similarity search (k=3 fetches the top 3 most relevant results)
        docs = vectorstore.similarity_search(query, k=3)
        
        # Combine the results into one text block
        context = "\n".join([doc.page_content for doc in docs])
        return context
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""