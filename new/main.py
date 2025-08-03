# main.py

import os
import logging
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from mistralai import Mistral
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader
from io import BytesIO

# --- Configuration & Initialization ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (for MISTRAL_API_KEY)
load_dotenv()

# Authentication Token (as specified in the problem description)
EXPECTED_AUTH_TOKEN = "Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

# --- Initialize Models (Global Singleton Pattern) ---
# This ensures models are loaded only once on startup, improving latency.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on application startup and cleanup on shutdown."""
    # Startup
    logging.info("Loading lightweight embedding model...")
    
    # Using the recommended lightweight model for better performance and lower memory usage
    # This model is only ~90MB compared to ~440MB of BGE, perfect for resource-constrained environments
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    app.state.embedding_model = SentenceTransformer(model_name)
    logging.info(f"'{model_name}' model loaded successfully - optimized for speed and low memory usage.")

    logging.info("Initializing Mistral LLM Client...")
    # Initialize the Mistral client with the API key from .env
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env file.")
    app.state.mistral_client = Mistral(api_key=api_key)
    logging.info("Mistral Client initialized successfully.")
    
    yield
    
    # Shutdown (cleanup if needed)
    logging.info("Application shutdown complete.")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="Processes documents to answer contextual questions using RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API Data Validation ---

class SubmissionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- Core Service Functions ---

def download_and_parse_pdf(pdf_url: str) -> List[str]:
    """
    Downloads a PDF from a URL, parses it, and splits it into text chunks.
    
    Args:
        pdf_url: The URL of the PDF document.

    Returns:
        A list of text chunks from the document.
    """
    try:
        logging.info(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Read PDF from in-memory bytes
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        logging.info(f"Successfully parsed PDF. Total characters: {len(full_text)}")
        
        # Simple but effective chunking strategy
        # A more advanced strategy could use RecursiveCharacterTextSplitter from LangChain
        chunk_size = 1000  # Characters per chunk
        overlap = 200      # Overlap between chunks to maintain context
        
        chunks = [
            full_text[i:i + chunk_size] 
            for i in range(0, len(full_text), chunk_size - overlap)
        ]
        logging.info(f"Document split into {len(chunks)} chunks.")
        return chunks

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not download or process the PDF from the provided URL.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during PDF processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse the PDF document.")


def retrieve_relevant_context(query: str, corpus_chunks: List[str], corpus_embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 5) -> str:
    """
    Retrieves the most relevant context chunks for a given query using semantic search.
    
    Args:
        query: The user's question.
        corpus_chunks: The list of text chunks from the document.
        corpus_embeddings: The pre-computed embeddings for the corpus chunks.
        model: The embedding model instance.
        top_k: The number of top relevant chunks to retrieve.

    Returns:
        A single string containing the concatenated relevant context.
    """
    # Encode the query to get its embedding (simplified for SentenceTransformer)
    query_embedding = model.encode([query])
    
    # Perform semantic search using sentence-transformers utility which is compatible with FAISS-like operations
    # This finds the 'top_k' most similar chunks from the corpus
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    
    # Collate the context from the retrieved chunks
    context_chunks = [corpus_chunks[hit['corpus_id']] for hit in hits]
    relevant_context = "\n---\n".join(context_chunks)
    
    logging.info(f"Retrieved {len(hits)} relevant chunks for the query: '{query}'")
    return relevant_context

def generate_answer_with_llm(context: str, question: str, client: Mistral) -> str:
    """
    Generates an answer using the Mistral LLM based on the provided context and question.
    This function is designed for token efficiency and explainability.

    Args:
        context: The relevant text retrieved from the document.
        question: The user's original question.
        client: The Mistral API client.

    Returns:
        The generated answer string.
    """
    # A carefully crafted prompt to guide the LLM
    prompt = f"""
    **Role:** You are a highly intelligent AI assistant specializing in document analysis for insurance, legal, and HR domains.
    
    **Task:** Answer the user's question based *exclusively* on the provided context below. Do not use any external knowledge or make assumptions.
    
    **Context from Document:**
    ---
    {context}
    ---
    
    **User's Question:**
    {question}
    
    **Instruction:**
    1.  Read the context carefully.
    2.  Formulate a clear, concise, and direct answer.
    3.  If the answer is not found in the context, state explicitly: "The information is not available in the provided document context."
    """

    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest", # Using a powerful model for high accuracy
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Low temperature for factual, deterministic answers
        )
        answer = chat_response.choices[0].message.content.strip()
        logging.info("Successfully generated answer with Mistral LLM.")
        return answer
    except Exception as e:
        logging.error(f"Error communicating with Mistral API: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The LLM service is currently unavailable.")

# --- API Endpoint ---

@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission: SubmissionRequest):
    """
    Main API endpoint to process documents and answer questions.
    It follows the complete RAG workflow:
    1. Authenticates the request.
    2. Downloads and parses the document.
    3. Creates embeddings for document chunks.
    4. For each question, retrieves relevant context.
    5. Generates an answer using the LLM.
    6. Returns all answers in a structured JSON response.
    """
    # 1. Authentication
    auth_header = request.headers.get("Authorization")
    if auth_header != EXPECTED_AUTH_TOKEN:
        logging.warning("Authentication failed. Invalid token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    logging.info("Authentication successful.")

    # 2. Document Processing
    doc_chunks = download_and_parse_pdf(submission.documents)
    
    # 3. Embedding Generation
    logging.info("Generating embeddings for document chunks...")
    # FAISS index is implicitly created and used by the semantic_search utility under the hood.
    # For very large documents, an explicit FAISS index build would be more scalable.
    corpus_embeddings = app.state.embedding_model.encode(doc_chunks)
    logging.info("Embeddings generated and indexed successfully.")
    
    # 4. & 5. Loop through questions to Retrieve and Generate
    final_answers = []
    for question in submission.questions:
        logging.info(f"Processing question: '{question}'")
        
        # Retrieve context relevant to the current question
        relevant_context = retrieve_relevant_context(
            query=question,
            corpus_chunks=doc_chunks,
            corpus_embeddings=corpus_embeddings,
            model=app.state.embedding_model
        )
        
        # Generate the answer using the LLM with the retrieved context
        answer = generate_answer_with_llm(
            context=relevant_context,
            question=question,
            client=app.state.mistral_client
        )
        final_answers.append(answer)
        
    # 6. Return Structured JSON Output
    return SubmissionResponse(answers=final_answers)

# --- To run the server ---
if __name__ == "__main__":
    # Use PORT environment variable from Render, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)