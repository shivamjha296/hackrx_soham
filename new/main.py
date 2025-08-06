# main.py

import os
import logging
import time
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import requests
from langchain_nomic import NomicEmbeddings
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader
from io import BytesIO
import json
import hashlib
import pickle
import os
from pathlib import Path

# --- API Key Management ---
class APIKeyManager:
    def __init__(self):
        # Load all API keys from environment
        self.mistral_keys = []
        self.nomic_keys = []
        
        # Load Mistral API keys
        for i in range(1, 5):  # Support up to 4 keys
            key = os.getenv(f"MISTRAL_API_KEY_{i}")
            if key and key != "your_backup_mistral_key_" + str(i):
                self.mistral_keys.append(key)
        
        # Load Nomic API keys  
        for i in range(1, 5):  # Support up to 4 keys
            key = os.getenv(f"NOMIC_API_KEY_{i}")
            if key and key != "your_backup_nomic_key_" + str(i):
                self.nomic_keys.append(key)
        
        # Current key indices
        self.current_mistral_index = 0
        self.current_nomic_index = 0
        
        # Failed key tracking
        self.failed_mistral_keys = set()
        self.failed_nomic_keys = set()
        
        logging.info(f"Loaded {len(self.mistral_keys)} Mistral API keys")
        logging.info(f"Loaded {len(self.nomic_keys)} Nomic API keys")
        
        if not self.mistral_keys:
            raise ValueError("No valid Mistral API keys found in .env file")
        if not self.nomic_keys:
            raise ValueError("No valid Nomic API keys found in .env file")
    
    def get_current_mistral_key(self) -> Optional[str]:
        """Get the current active Mistral API key."""
        if self.current_mistral_index < len(self.mistral_keys):
            return self.mistral_keys[self.current_mistral_index]
        return None
    
    def get_current_nomic_key(self) -> Optional[str]:
        """Get the current active Nomic API key."""
        if self.current_nomic_index < len(self.nomic_keys):
            return self.nomic_keys[self.current_nomic_index]
        return None
    
    def mark_mistral_key_failed(self, key: str):
        """Mark a Mistral API key as failed and switch to next available key."""
        self.failed_mistral_keys.add(key)
        self.current_mistral_index += 1
        if self.current_mistral_index < len(self.mistral_keys):
            logging.warning(f"Mistral API key failed, switching to backup key #{self.current_mistral_index + 1}")
        else:
            logging.error("All Mistral API keys have been exhausted!")
    
    def mark_nomic_key_failed(self, key: str):
        """Mark a Nomic API key as failed and switch to next available key."""
        self.failed_nomic_keys.add(key)
        self.current_nomic_index += 1
        if self.current_nomic_index < len(self.nomic_keys):
            logging.warning(f"Nomic API key failed, switching to backup key #{self.current_nomic_index + 1}")
        else:
            logging.error("All Nomic API keys have been exhausted!")
    
    def has_available_mistral_key(self) -> bool:
        """Check if there are any available Mistral API keys."""
        return self.current_mistral_index < len(self.mistral_keys)
    
    def has_available_nomic_key(self) -> bool:
        """Check if there are any available Nomic API keys."""
        return self.current_nomic_index < len(self.nomic_keys)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Initialize global API key manager after loading .env
api_key_manager = APIKeyManager()

EXPECTED_AUTH_TOKEN = "Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"
MISTRAL_LLM_MODEL = "mistral-large-latest"

# Set Nomic API key from key manager
current_nomic_key = api_key_manager.get_current_nomic_key()
if current_nomic_key:
    os.environ["NOMIC_API_KEY"] = current_nomic_key

# Create embeddings cache directory
CACHE_DIR = Path("embeddings_cache")
try:
    CACHE_DIR.mkdir(exist_ok=True)
    logging.info(f"Cache directory created/confirmed at: {CACHE_DIR.absolute()}")
except Exception as e:
    logging.error(f"Failed to create cache directory: {e}")
    # Fallback to current directory
    CACHE_DIR = Path(".")
    logging.warning("Using current directory as cache location")

# --- Initialize Models (Global Singleton Pattern) ---
# This ensures models are loaded only once on startup, improving latency.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on application startup and cleanup on shutdown."""
    # Startup
    logging.info("Initializing Mistral LLM Client...")
    # Initialize the Mistral client with the current API key
    current_key = api_key_manager.get_current_mistral_key()
    if not current_key:
        raise ValueError("No valid Mistral API key available.")
    app.state.mistral_client = create_mistral_client(current_key)
    logging.info("Mistral Client initialized successfully.")
    
    yield
    
    # Shutdown (cleanup if needed)
    logging.info("Application shutdown complete.")

def create_mistral_client(api_key: str) -> Mistral:
    """Create a new Mistral client with the given API key."""
    return Mistral(api_key=api_key)

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

def get_pdf_hash(pdf_content: bytes) -> str:
    """Generate a unique hash for PDF content."""
    return hashlib.md5(pdf_content).hexdigest()

def get_url_hash(url: str) -> str:
    """Generate a hash for the URL to check if we've seen this URL before."""
    # Remove query parameters that might change (like timestamps)
    parsed = urlparse(url)
    # Use just the scheme, netloc, and path (not query params)
    stable_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return hashlib.md5(stable_url.encode()).hexdigest()

def save_pdf_metadata_to_cache(url_hash: str, pdf_url: str, pdf_hash: str, chunks: List[str]):
    """Save PDF metadata and chunks to cache for quick lookup by URL."""
    try:
        metadata_file = CACHE_DIR / f"url_{url_hash}.json"
        metadata = {
            'pdf_url': pdf_url,
            'pdf_hash': pdf_hash,
            'chunk_count': len(chunks),
            'total_characters': sum(len(chunk) for chunk in chunks),
            'timestamp': time.time(),
            'cached_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved PDF metadata to cache: {metadata_file}")
    except Exception as e:
        logging.warning(f"Failed to save PDF metadata to cache: {e}")

def load_pdf_metadata_from_cache(url_hash: str) -> dict:
    """Load PDF metadata from cache by URL hash."""
    try:
        metadata_file = CACHE_DIR / f"url_{url_hash}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Found PDF metadata in cache for URL: {metadata.get('cached_at', 'unknown time')}")
            return metadata
        return None
    except Exception as e:
        logging.warning(f"Failed to load PDF metadata from cache: {e}")
        return None

def save_embeddings_to_cache(pdf_hash: str, chunks: List[str], embeddings: np.ndarray, pdf_url: str = None):
    """Save document chunks, embeddings, and PDF URL to cache."""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        cache_data = {
            'chunks': chunks,
            'embeddings': embeddings,
            'pdf_url': pdf_url,  # Store the PDF URL for reference
            'timestamp': time.time(),  # Use current timestamp
            'total_characters': sum(len(chunk) for chunk in chunks),
            'chunk_count': len(chunks)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info(f"Saved embeddings and chunks to cache: {cache_file}")
        logging.info(f"Cache contains {len(chunks)} chunks with {cache_data['total_characters']} total characters")
    except Exception as e:
        logging.warning(f"Failed to save embeddings to cache: {e}")

def load_embeddings_from_cache(pdf_hash: str) -> tuple:
    """Load document chunks and embeddings from cache."""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            logging.info(f"Loaded embeddings from cache: {cache_file}")
            logging.info(f"Cache contains {cache_data.get('chunk_count', len(cache_data['chunks']))} chunks with {cache_data.get('total_characters', 'unknown')} total characters")
            return cache_data['chunks'], cache_data['embeddings']
        return None, None
    except Exception as e:
        logging.warning(f"Failed to load embeddings from cache: {e}")
        return None, None

def generate_embeddings_nomic(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using Nomic's LangChain integration with fallback mechanism.
    """
    max_retries = len(api_key_manager.nomic_keys)
    
    for attempt in range(max_retries):
        current_key = api_key_manager.get_current_nomic_key()
        if not current_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="All Nomic API keys have been exhausted."
            )
        
        try:
            logging.info(f"Generating embeddings for {len(texts)} text chunks using Nomic API (attempt {attempt + 1}/{max_retries})...")
            
            # Set the current key in environment
            os.environ["NOMIC_API_KEY"] = current_key
            
            embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5")
            embeddings = embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings)
            logging.info(f"Successfully generated all embeddings: shape {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            error_message = str(e).lower()
            logging.error(f"Error generating embeddings with Nomic API (key #{attempt + 1}): {e}")
            
            # Check if it's an API exhaustion or auth error
            if any(keyword in error_message for keyword in ['rate limit', 'quota', 'exhausted', 'unauthorized', '401', '429', '403']):
                logging.warning(f"API key #{attempt + 1} appears to be exhausted or unauthorized, trying next key...")
                api_key_manager.mark_nomic_key_failed(current_key)
                if attempt < max_retries - 1:
                    continue
            
            # If it's not an API key issue or we've exhausted all keys, raise the error
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                    detail="Nomic embedding service is currently unavailable - all API keys exhausted."
                )
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
        detail="Failed to generate embeddings with any available API key."
    )

def download_and_parse_pdf(pdf_url) -> tuple:
    """
    Downloads a PDF from a URL, parses it, and splits it into text chunks.
    Uses URL-based caching to skip downloading and parsing if we've seen this PDF before.
    Returns both the chunks and the PDF content hash for caching.
    
    Args:
        pdf_url: The URL of the PDF document (can be HttpUrl object or string).

    Returns:
        A tuple of (chunks, pdf_hash, pdf_content_bytes)
    """
    
    # Convert HttpUrl object to string if needed
    pdf_url_str = str(pdf_url)
    
    # First, check if we've seen this URL before (ignoring query params)
    url_hash = get_url_hash(pdf_url_str)
    cached_metadata = load_pdf_metadata_from_cache(url_hash)
    
    if cached_metadata:
        pdf_hash = cached_metadata['pdf_hash']
        logging.info(f"Found URL in cache, checking for chunks and embeddings (PDF hash: {pdf_hash})")
        
        # Try to load the chunks and embeddings
        cached_chunks, cached_embeddings = load_embeddings_from_cache(pdf_hash)
        if cached_chunks is not None:
            logging.info(f"✅ CACHE HIT: Skipping PDF download and parsing! Using cached data from {cached_metadata.get('cached_at', 'unknown time')}")
            logging.info(f"📄 Loaded {len(cached_chunks)} chunks with {cached_metadata.get('total_characters', 'unknown')} total characters")
            
            # Return the cached data - we'll set pdf_content to None since we don't need it
            return cached_chunks, pdf_hash, None
    
    # If not in cache, proceed with download and parsing
    try:
        logging.info(f"📥 Downloading PDF from {pdf_url_str}")
        response = requests.get(pdf_url_str)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get PDF content and generate hash
        pdf_content = response.content
        pdf_hash = get_pdf_hash(pdf_content)
        logging.info(f"PDF hash: {pdf_hash}")

        # Read PDF from in-memory bytes
        pdf_file = BytesIO(pdf_content)
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
        
        # Save the URL metadata for future use
        save_pdf_metadata_to_cache(url_hash, pdf_url_str, pdf_hash, chunks)
        
        return chunks, pdf_hash, pdf_content

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not download or process the PDF from the provided URL.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during PDF processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse the PDF document.")


def retrieve_relevant_context(query: str, corpus_chunks: List[str], corpus_embeddings: np.ndarray, top_k: int = 5) -> str:
    """
    Retrieves the most relevant context chunks for a given query using semantic search with fallback.
    """
    max_retries = len(api_key_manager.nomic_keys)
    
    for attempt in range(max_retries):
        current_key = api_key_manager.get_current_nomic_key() 
        if not current_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="All Nomic API keys have been exhausted."
            )
        
        try:
            # Set the current key in environment
            os.environ["NOMIC_API_KEY"] = current_key
            
            # Generate embedding for the query using Nomic LangChain
            embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5")
            query_embedding = np.array([embeddings_model.embed_query(query)])
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            context_chunks = [corpus_chunks[idx] for idx in top_indices]
            relevant_context = "\n---\n".join(context_chunks)
            logging.info(f"Retrieved {len(top_indices)} relevant chunks for the query: '{query}'")
            return relevant_context
            
        except Exception as e:
            error_message = str(e).lower()
            logging.error(f"Error in semantic search with Nomic API (key #{attempt + 1}): {e}")
            
            # Check if it's an API exhaustion or auth error
            if any(keyword in error_message for keyword in ['rate limit', 'quota', 'exhausted', 'unauthorized', '401', '429', '403']):
                logging.warning(f"API key #{attempt + 1} appears to be exhausted, trying next key...")
                api_key_manager.mark_nomic_key_failed(current_key)
                if attempt < max_retries - 1:
                    continue
            
            # If it's not an API key issue or we've exhausted all keys, raise the error
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Semantic search service is currently unavailable."
            )
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Failed to perform semantic search with any available API key."
    )

def generate_answer_with_llm(context: str, question: str, client: Mistral) -> str:
    """
    Generates an answer using the Mistral LLM with fallback mechanism for API key rotation.
    """
    max_retries = len(api_key_manager.mistral_keys)
    
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
    - Answer in a single, direct sentence.
    - Do not quote or reference the source.
    - If the answer is not found, reply: "The information is not available in the provided document context."
    """

    for attempt in range(max_retries):
        current_key = api_key_manager.get_current_mistral_key()
        if not current_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="All Mistral API keys have been exhausted."
            )
        
        try:
            # Always create a new client with the current key for this attempt
            current_client = create_mistral_client(current_key)
            logging.info(f"Using Mistral API key #{api_key_manager.current_mistral_index + 1}")
            
            chat_response = current_client.chat.complete(
                model=MISTRAL_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer = chat_response.choices[0].message.content.strip()
            logging.info("Successfully generated answer with Mistral LLM.")
            return answer
            
        except Exception as e:
            error_message = str(e).lower()
            logging.error(f"Error communicating with Mistral API (key #{api_key_manager.current_mistral_index + 1}): {e}")
            
            # Check if it's an API exhaustion, auth error, or rate limit
            if any(keyword in error_message for keyword in ['rate limit', 'quota', 'exhausted', 'unauthorized', '401', '429', '403', 'status 401']):
                logging.warning(f"Mistral API key #{api_key_manager.current_mistral_index + 1} appears to be exhausted or unauthorized, trying next key...")
                api_key_manager.mark_mistral_key_failed(current_key)
                if attempt < max_retries - 1:
                    continue
            
            # If it's not an API key issue or we've exhausted all keys, raise the error
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The LLM service is currently unavailable - all API keys exhausted."
                )
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Failed to generate answer with any available API key."
    )

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "LLM-Powered Intelligent Query–Retrieval System",
        "status": "online",
        "version": "1.0.0",
        "configuration": {
            "embeddings": "Nomic API (nomic-embed-text-v1.5) with fallback",
            "llm": "Mistral API with fallback",
            "caching": "enabled",
            "api_keys": {
                "mistral_total": len(api_key_manager.mistral_keys),
                "nomic_total": len(api_key_manager.nomic_keys)
            }
        },
        "endpoints": {
            "main": "/hackrx/run (POST)",
            "health": "/health (GET)",
            "cache": "/cache/status (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get("/cache/status")
async def cache_status():
    """Get cache status and information."""
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        url_cache_files = list(CACHE_DIR.glob("url_*.json"))
        
        cache_info = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                cache_info.append({
                    "file": cache_file.name,
                    "pdf_hash": cache_file.stem,
                    "chunk_count": cache_data.get('chunk_count', len(cache_data.get('chunks', []))),
                    "total_characters": cache_data.get('total_characters', 'unknown'),
                    "cached_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_data.get('timestamp', 0))),
                    "pdf_url": cache_data.get('pdf_url', 'unknown')[:100] + "..." if cache_data.get('pdf_url') and len(cache_data.get('pdf_url', '')) > 100 else cache_data.get('pdf_url', 'unknown')
                })
            except Exception as e:
                cache_info.append({
                    "file": cache_file.name,
                    "error": f"Failed to read cache file: {e}"
                })
        
        url_cache_info = []
        for url_file in url_cache_files:
            try:
                with open(url_file, 'r') as f:
                    url_data = json.load(f)
                url_cache_info.append({
                    "file": url_file.name,
                    "url_hash": url_file.stem.replace('url_', ''),
                    "pdf_url": url_data.get('pdf_url', 'unknown')[:100] + "..." if len(url_data.get('pdf_url', '')) > 100 else url_data.get('pdf_url', 'unknown'),
                    "pdf_hash": url_data.get('pdf_hash', 'unknown'),
                    "chunk_count": url_data.get('chunk_count', 'unknown'),
                    "cached_at": url_data.get('cached_at', 'unknown')
                })
            except Exception as e:
                url_cache_info.append({
                    "file": url_file.name,
                    "error": f"Failed to read URL cache file: {e}"
                })
        
        return {
            "cache_directory": str(CACHE_DIR.absolute()),
            "embedding_cache_files": len(cache_files),
            "url_cache_files": len(url_cache_files),
            "embedding_cache_details": cache_info,
            "url_cache_details": url_cache_info,
            "total_cached_pdfs": len(cache_files)
        }
    except Exception as e:
        return {
            "error": f"Failed to read cache status: {e}",
            "cache_directory": str(CACHE_DIR.absolute())
        }

@app.get("/health")
async def health_check():
    """Health check endpoint with API key status."""
    
    return {
        "status": "healthy", 
        "timestamp": "2025-08-05",
        "services": {
            "mistral_api": {
                "total_keys": len(api_key_manager.mistral_keys),
                "current_key_index": api_key_manager.current_mistral_index + 1,
                "failed_keys": len(api_key_manager.failed_mistral_keys),
                "available_keys": len(api_key_manager.mistral_keys) - len(api_key_manager.failed_mistral_keys),
                "status": "available" if api_key_manager.has_available_mistral_key() else "exhausted"
            },
            "nomic_api": {
                "total_keys": len(api_key_manager.nomic_keys),
                "current_key_index": api_key_manager.current_nomic_index + 1,
                "failed_keys": len(api_key_manager.failed_nomic_keys),
                "available_keys": len(api_key_manager.nomic_keys) - len(api_key_manager.failed_nomic_keys),
                "status": "available" if api_key_manager.has_available_nomic_key() else "exhausted"
            }
        }
    }

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
    doc_chunks, pdf_hash, pdf_content = download_and_parse_pdf(submission.documents)
    
    # 3. Embedding Generation with Caching
    logging.info("Checking for cached embeddings...")
    cached_chunks, cached_embeddings = load_embeddings_from_cache(pdf_hash)
    
    if cached_chunks is not None and cached_embeddings is not None:
        logging.info("Using cached embeddings!")
        corpus_embeddings = cached_embeddings
        doc_chunks = cached_chunks
    else:
        logging.info("Generating new embeddings for document chunks...")
        corpus_embeddings = generate_embeddings_nomic(doc_chunks)
        # Save with PDF URL for reference
        save_embeddings_to_cache(pdf_hash, doc_chunks, corpus_embeddings, str(submission.documents))
        logging.info("Embeddings generated and cached successfully.")
    
    # 4. & 5. Loop through questions to Retrieve and Generate
    final_answers = []
    for question in submission.questions:
        logging.info(f"Processing question: '{question}'")
        
        # Retrieve context relevant to the current question
        relevant_context = retrieve_relevant_context(
            query=question,
            corpus_chunks=doc_chunks,
            corpus_embeddings=corpus_embeddings
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