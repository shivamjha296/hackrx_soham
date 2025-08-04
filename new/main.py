# main.py

import os
import logging
import random
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity
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

# Mistral API Configuration
MISTRAL_EMBEDDING_BATCH_SIZE = 20  # Conservative batch size for Mistral API
MISTRAL_EMBEDDING_MODEL = "mistral-embed"

# --- API Key Rotation Manager ---
class MistralAPIKeyManager:
    """Manages multiple Mistral API keys with intelligent rotation and fallback."""
    
    def __init__(self):
        self.api_keys = []
        self.clients = {}
        self.current_key_index = 0
        self.key_usage_count = {}
        self.failed_keys = set()
        self.last_used_time = {}
        
        # Load all API keys from environment
        self._load_api_keys()
        self._initialize_clients()
        
    def _load_api_keys(self):
        """Load all available API keys from environment variables."""
        # Try to load keys in order
        for i in range(1, 11):  # MISTRAL_API_KEY_1 to MISTRAL_API_KEY_10
            key = os.getenv(f"MISTRAL_API_KEY_{i}")
            if key and key != "your_api_key_here" and key.strip() != "" and len(key.strip()) > 10:
                self.api_keys.append(key.strip())
                self.key_usage_count[key.strip()] = 0
                self.last_used_time[key.strip()] = 0
                
        # Also check the primary key
        primary_key = os.getenv("MISTRAL_API_KEY")
        if primary_key and primary_key not in self.api_keys and len(primary_key.strip()) > 10:
            self.api_keys.append(primary_key)
            self.key_usage_count[primary_key] = 0
            self.last_used_time[primary_key] = 0
            
        if not self.api_keys:
            raise ValueError("No valid Mistral API keys found in environment variables.")
            
        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in self.api_keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        self.api_keys = unique_keys
            
        logging.info(f"Loaded {len(self.api_keys)} unique API keys for rotation")
        
    def _initialize_clients(self):
        """Initialize Mistral clients for each API key."""
        for api_key in self.api_keys:
            try:
                self.clients[api_key] = Mistral(api_key=api_key)
                logging.info(f"Initialized client for API key: ...{api_key[-8:]}")
            except Exception as e:
                logging.error(f"Failed to initialize client for API key ...{api_key[-8:]}: {e}")
                self.failed_keys.add(api_key)
                
    def get_next_client(self) -> Mistral:
        """Get the next available Mistral client with ultra-fast rotation."""
        available_keys = [key for key in self.api_keys if key not in self.failed_keys]
        
        if not available_keys:
            # Reset failed keys if all have failed (maybe temporary issues)
            logging.warning("All API keys marked as failed. Resetting and retrying...")
            self.failed_keys.clear()
            available_keys = self.api_keys
            
        # Ultra-fast round-robin rotation - no complex calculations
        self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        best_key = available_keys[self.current_key_index]
        
        # Quick update tracking (minimal overhead)
        current_time = time.time()
        self.key_usage_count[best_key] += 1
        self.last_used_time[best_key] = current_time
        
        # Only log every 10th call to reduce logging overhead
        if self.key_usage_count[best_key] % 10 == 1:
            logging.info(f"Using API key ...{best_key[-8:]} (usage: {self.key_usage_count[best_key]})")
        
        return self.clients[best_key]
            
    def mark_key_failed(self, client: Mistral):
        """Mark an API key as failed if it encounters rate limiting or errors."""
        for api_key, stored_client in self.clients.items():
            if stored_client == client:
                self.failed_keys.add(api_key)
                logging.warning(f"Marked API key ...{api_key[-8:]} as failed")
                break
                
    def get_stats(self) -> dict:
        """Get usage statistics for all API keys."""
        return {
            "total_keys": len(self.api_keys),
            "failed_keys": len(self.failed_keys),
            "usage_counts": {f"...{key[-8:]}": count for key, count in self.key_usage_count.items()}
        }

# --- Initialize Models (Global Singleton Pattern) ---
# This ensures models are loaded only once on startup, improving latency.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on application startup and cleanup on shutdown."""
    # Startup
    logging.info("Initializing Mistral API Key Manager...")
    app.state.mistral_key_manager = MistralAPIKeyManager()
    logging.info("Mistral API Key Manager initialized successfully.")
    
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

def generate_embeddings(texts: List[str], key_manager: MistralAPIKeyManager, batch_size: int = MISTRAL_EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Mistral's embedding API with key rotation.
    Processes texts in batches to avoid API limits.
    
    Args:
        texts: List of text strings to embed.
        key_manager: The Mistral API key manager for rotation.
        batch_size: Maximum number of texts to process in one API call.
        
    Returns:
        A numpy array of embeddings.
    """
    try:
        logging.info(f"Generating embeddings for {len(texts)} text chunks using Mistral API...")
        
        all_embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Fast retry with minimal delays
            max_retries = min(2, len(key_manager.api_keys))  # Reduced retries for speed
            success = False
            
            for attempt in range(max_retries):
                try:
                    client = key_manager.get_next_client()
                    embeddings_response = client.embeddings.create(
                        model=MISTRAL_EMBEDDING_MODEL,
                        inputs=batch
                    )
                    
                    # Extract embeddings from the response
                    batch_embeddings = [item.embedding for item in embeddings_response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    logging.info(f"Successfully processed batch {batch_num}/{total_batches}")
                    success = True
                    break
                    
                except Exception as e:
                    error_message = str(e).lower()
                    if any(term in error_message for term in ["rate_limit", "quota", "exceeded", "throttled"]):
                        logging.warning(f"Rate limit hit, switching keys (attempt {attempt + 1})")
                        key_manager.mark_key_failed(client)
                        # No delay - immediate switch to next key
                    else:
                        logging.error(f"Error in batch {batch_num}: {e}")
                        if attempt == max_retries - 1:
                            raise
                        
            if not success:
                raise Exception(f"Failed to process batch {batch_num} after {max_retries} attempts")
        
        embeddings_array = np.array(all_embeddings)
        logging.info(f"Successfully generated all embeddings: shape {embeddings_array.shape}")
        return embeddings_array
        
    except Exception as e:
        logging.error(f"Error generating embeddings with Mistral API: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The embedding service is currently unavailable.")

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


def retrieve_relevant_context(query: str, corpus_chunks: List[str], corpus_embeddings: np.ndarray, key_manager: MistralAPIKeyManager, top_k: int = 5) -> str:
    """
    Retrieves the most relevant context chunks for a given query using semantic search.
    
    Args:
        query: The user's question.
        corpus_chunks: The list of text chunks from the document.
        corpus_embeddings: The pre-computed embeddings for the corpus chunks.
        key_manager: The Mistral API key manager for rotation.
        top_k: The number of top relevant chunks to retrieve.

    Returns:
        A single string containing the concatenated relevant context.
    """
    # Generate embedding for the query using Mistral
    query_embedding = generate_embeddings([query], key_manager)
    
    # Calculate cosine similarity between query and all corpus chunks
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    # Get top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort in descending order
    
    # Collate the context from the retrieved chunks
    context_chunks = [corpus_chunks[idx] for idx in top_indices]
    relevant_context = "\n---\n".join(context_chunks)
    
    logging.info(f"Retrieved {len(top_indices)} relevant chunks for the query: '{query}'")
    return relevant_context

def generate_answer_with_llm(context: str, question: str, key_manager: MistralAPIKeyManager) -> str:
    """
    Generates an answer using the Mistral LLM based on the provided context and question.
    This function is designed for token efficiency and explainability.

    Args:
        context: The relevant text retrieved from the document.
        question: The user's original question.
        key_manager: The Mistral API key manager for rotation.

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

    max_retries = min(2, len(key_manager.api_keys))  # Reduced for speed
    for attempt in range(max_retries):
        try:
            client = key_manager.get_next_client()
            chat_response = client.chat.complete(
                model="mistral-large-latest", # Using a powerful model for high accuracy
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # Low temperature for factual, deterministic answers
            )
            answer = chat_response.choices[0].message.content.strip()
            logging.info("Successfully generated answer with Mistral LLM.")
            return answer
            
        except Exception as e:
            error_message = str(e).lower()
            if any(term in error_message for term in ["rate_limit", "quota", "exceeded", "throttled"]):
                logging.warning(f"Rate limit hit, switching keys (attempt {attempt + 1})")
                key_manager.mark_key_failed(client)
                # No delay - immediate switch
            else:
                logging.error(f"Error communicating with Mistral API: {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The LLM service is currently unavailable.")
    
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="All API keys exhausted or unavailable.")

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
    corpus_embeddings = generate_embeddings(doc_chunks, app.state.mistral_key_manager)
    logging.info("Embeddings generated successfully.")
    
    # 4. & 5. Loop through questions to Retrieve and Generate
    final_answers = []
    for question in submission.questions:
        logging.info(f"Processing question: '{question}'")
        
        # Retrieve context relevant to the current question
        relevant_context = retrieve_relevant_context(
            query=question,
            corpus_chunks=doc_chunks,
            corpus_embeddings=corpus_embeddings,
            key_manager=app.state.mistral_key_manager
        )
        
        # Generate the answer using the LLM with the retrieved context
        answer = generate_answer_with_llm(
            context=relevant_context,
            question=question,
            key_manager=app.state.mistral_key_manager
        )
        final_answers.append(answer)
        
    # 6. Return Structured JSON Output
    return SubmissionResponse(answers=final_answers)

# --- Additional Monitoring Endpoint ---
@app.get("/api/stats")
async def get_api_stats(request: Request):
    """
    Get statistics about API key usage and performance.
    Useful for monitoring during the hackathon.
    """
    # Simple authentication check
    auth_header = request.headers.get("Authorization")
    if auth_header != EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    
    stats = app.state.mistral_key_manager.get_stats()
    return {
        "api_key_stats": stats,
        "status": "API key rotation system active"
    }

# --- To run the server ---
if __name__ == "__main__":
    # Use PORT environment variable from Render, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)