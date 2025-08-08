# main.py

import os
import logging
import time
import asyncio
import hashlib
import pickle
import json
import csv
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

import numpy as np
import aiohttp
import aiofiles
import requests  # Keep for fallback synchronous operations
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, falling back to file-based caching only")

from langchain_nomic import NomicEmbeddings
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader

# --- Advanced Cache Manager ---
class AdvancedCacheManager:
    """
    Advanced caching system supporting both Redis and file-based caching
    with different cache types for different data patterns.
    """
    
    def __init__(self, redis_url: str = None, file_cache_dir: str = "embeddings_cache"):
        self.file_cache_dir = Path(file_cache_dir)
        self.file_cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for frequently accessed data
        self.memory_cache = {}
        self.memory_cache_ttl = {}
        self.max_memory_cache_size = 100
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Redis connection
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                logging.info("Redis cache initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to connect to Redis: {e}")
                
        # Keep backward compatibility with memory_cache_timestamps
        self.memory_cache_timestamps = self.memory_cache_ttl
    
    async def get_redis_value(self, key: str) -> Optional[bytes]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            # Only log once per minute to avoid spam
            if not hasattr(self, '_last_redis_error') or time.time() - self._last_redis_error > 60:
                logging.warning(f"Redis connection failed, falling back to file cache: {e}")
                self._last_redis_error = time.time()
            return None
    
    async def set_redis_value(self, key: str, value: bytes, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL."""
        if not self.redis_client:
            return False
        try:
            await self.redis_client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            # Only log once per minute to avoid spam
            if not hasattr(self, '_last_redis_error') or time.time() - self._last_redis_error > 60:
                logging.warning(f"Redis connection failed, falling back to file cache: {e}")
                self._last_redis_error = time.time()
            return False
    
    def get_memory_cache(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        if key in self.memory_cache:
            if time.time() < self.memory_cache_ttl.get(key, 0):
                return self.memory_cache[key]
            else:
                # Expired
                del self.memory_cache[key]
                if key in self.memory_cache_ttl:
                    del self.memory_cache_ttl[key]
        return None
    
    def set_memory_cache(self, key: str, value: Any, ttl: int = 300):
        """Set value in in-memory cache with TTL."""
        # Manage cache size
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove oldest entries
            oldest_key = min(self.memory_cache_ttl.keys(), key=lambda k: self.memory_cache_ttl[k])
            del self.memory_cache[oldest_key]
            del self.memory_cache_ttl[oldest_key]
        
        self.memory_cache[key] = value
        self.memory_cache_ttl[key] = time.time() + ttl
    
    async def get_llm_answer_cache(self, context_hash: str, question_hash: str) -> Optional[str]:
        """Get cached LLM answer."""
        cache_key = f"llm_answer:{context_hash}:{question_hash}"
        
        # Try memory cache first
        result = self.get_memory_cache(cache_key)
        if result:
            return result
        
        # Try Redis
        redis_data = await self.get_redis_value(cache_key)
        if redis_data:
            answer = pickle.loads(redis_data)
            self.set_memory_cache(cache_key, answer)  # Cache in memory too
            return answer
        
        # Try file cache
        file_path = self.file_cache_dir / f"llm_{context_hash}_{question_hash}.pkl"
        if file_path.exists():
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                    answer = pickle.loads(content)
                    self.set_memory_cache(cache_key, answer)
                    return answer
            except Exception as e:
                logging.warning(f"Failed to load LLM cache from file: {e}")
        
        return None
    
    async def set_llm_answer_cache(self, context_hash: str, question_hash: str, answer: str):
        """Cache LLM answer with multiple storage layers."""
        cache_key = f"llm_answer:{context_hash}:{question_hash}"
        
        # Set in memory cache
        self.set_memory_cache(cache_key, answer)
        
        # Set in Redis (with longer TTL)
        data = pickle.dumps(answer)
        await self.set_redis_value(cache_key, data, ttl=7200)  # 2 hours
        
        # Set in file cache (permanent until manually cleared)
        file_path = self.file_cache_dir / f"llm_{context_hash}_{question_hash}.pkl"
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
        except Exception as e:
            logging.warning(f"Failed to save LLM cache to file: {e}")
    
    async def get_retrieved_context_cache(self, query_hash: str, doc_hash: str) -> Optional[str]:
        """Get cached retrieved context."""
        cache_key = f"context:{doc_hash}:{query_hash}"
        
        # Try memory cache first
        result = self.get_memory_cache(cache_key)
        if result:
            return result
        
        # Try Redis
        redis_data = await self.get_redis_value(cache_key)
        if redis_data:
            context = pickle.loads(redis_data)
            self.set_memory_cache(cache_key, context)
            return context
        
        return None
    
    async def set_retrieved_context_cache(self, query_hash: str, doc_hash: str, context: str):
        """Cache retrieved context."""
        cache_key = f"context:{doc_hash}:{query_hash}"
        
        # Set in memory cache
        self.set_memory_cache(cache_key, context)
        
        # Set in Redis
        data = pickle.dumps(context)
        await self.set_redis_value(cache_key, data, ttl=3600)  # 1 hour
    
    def cleanup_old_cache_files(self, max_age_days: int = 7):
        """Clean up old cache files to manage disk space."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        for cache_file in self.file_cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    logging.info(f"Cleaned up old cache file: {cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to clean up cache file {cache_file}: {e}")
    
    async def close(self):
        """Close connections and cleanup."""
        if self.redis_client:
            await self.redis_client.close()
        self.thread_pool.shutdown(wait=True)

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

# --- CSV Query Logger ---
class CSVQueryLogger:
    """
    Logs every query and response to a CSV file with timestamps for analysis and auditing.
    """
    
    def __init__(self, log_file: str = "query_response_log.csv"):
        self.log_file = Path(log_file)
        self.lock = asyncio.Lock()
        self._ensure_csv_header()
    
    def _ensure_csv_header(self):
        """Ensure CSV file exists with proper headers"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'document_url',
                    'document_hash',
                    'question',
                    'answer',
                    'response_time_ms',
                    'endpoint_type',
                    'cache_used',
                    'chunk_count',
                    'confidence_score'
                ])
    
    async def log_query_response(self, 
                                document_url: str,
                                document_hash: str,
                                question: str, 
                                answer: str,
                                response_time_ms: float,
                                endpoint_type: str = "standard",
                                cache_used: bool = False,
                                chunk_count: int = 0,
                                confidence_score: float = None):
        """Log a query-response pair to CSV"""
        async with self.lock:
            try:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        document_url,
                        document_hash,
                        question.replace('\n', ' ').replace('\r', ''),  # Clean newlines
                        answer.replace('\n', ' ').replace('\r', ''),    # Clean newlines
                        round(response_time_ms, 2),
                        endpoint_type,
                        cache_used,
                        chunk_count,
                        round(confidence_score, 3) if confidence_score else None
                    ])
            except Exception as e:
                logging.error(f"Failed to log query-response to CSV: {e}")

# Initialize global instances
api_key_manager = APIKeyManager()
cache_manager = AdvancedCacheManager(
    redis_url=os.getenv("REDIS_URL"),  # Optional Redis URL
    file_cache_dir="embeddings_cache"
)
csv_logger = CSVQueryLogger()  # Initialize CSV logger

EXPECTED_AUTH_TOKEN = "Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"
MISTRAL_LLM_MODEL = "ministral-8b-2410"

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
    
    # Start background cache cleanup task
    app.state.cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    
    yield
    
    # Shutdown (cleanup if needed)
    logging.info("Shutting down application...")
    app.state.cleanup_task.cancel()
    try:
        await app.state.cleanup_task
    except asyncio.CancelledError:
        pass
    await cache_manager.close()
    logging.info("Application shutdown complete.")

async def periodic_cache_cleanup():
    """Periodic task to clean up old cache files."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            cache_manager.cleanup_old_cache_files()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error during cache cleanup: {e}")

def create_mistral_client(api_key: str) -> Mistral:
    """Create a new Mistral client with the given API key."""
    return Mistral(api_key=api_key)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System (Enhanced with Aggressive Caching & Async)",
    description="Processes documents to answer contextual questions using RAG with advanced caching and parallel processing.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API Data Validation ---

class SubmissionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

class EnhancedSubmissionResponse(BaseModel):
    answers: List[str]
    detailed_answers: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

# --- Utility Functions for Hashing ---

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

def get_text_hash(text: str) -> str:
    """Generate a hash for text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_context_hash(context: str) -> str:
    """Generate a hash for context content for LLM caching."""
    return hashlib.md5(context.encode('utf-8')).hexdigest()

def get_question_hash(question: str) -> str:
    """Generate a hash for question content for LLM caching."""
    return hashlib.md5(question.encode('utf-8')).hexdigest()

# --- Async File-based Cache Functions ---

async def save_pdf_metadata_to_cache(url_hash: str, pdf_url: str, pdf_hash: str, chunks: List[str], chunks_data: List[Dict[str, Any]] = None):
    """Save PDF metadata and chunks to cache for quick lookup by URL."""
    try:
        metadata_file = CACHE_DIR / f"url_{url_hash}.json"
        metadata = {
            'pdf_url': pdf_url,
            'pdf_hash': pdf_hash,
            'chunk_count': len(chunks),
            'total_characters': sum(len(chunk) for chunk in chunks),
            'timestamp': time.time(),
            'cached_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'has_enhanced_chunks': chunks_data is not None
        }
        
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        logging.info(f"Saved PDF metadata to cache: {metadata_file}")
    except Exception as e:
        logging.warning(f"Failed to save PDF metadata to cache: {e}")

async def load_pdf_metadata_from_cache(url_hash: str) -> dict:
    """Load PDF metadata from cache by URL hash."""
    try:
        metadata_file = CACHE_DIR / f"url_{url_hash}.json"
        if metadata_file.exists():
            async with aiofiles.open(metadata_file, 'r') as f:
                content = await f.read()
                metadata = json.loads(content)
            logging.info(f"Found PDF metadata in cache for URL: {metadata.get('cached_at', 'unknown time')}")
            return metadata
        return None
    except Exception as e:
        logging.warning(f"Failed to load PDF metadata from cache: {e}")
        return None

async def save_embeddings_to_cache(pdf_hash: str, chunks: List[str], embeddings: np.ndarray, pdf_url: str = None, chunks_data: List[Dict[str, Any]] = None):
    """Save document chunks, embeddings, and enhanced chunk data to cache."""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        cache_data = {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunks_data': chunks_data,  # Enhanced chunk metadata
            'pdf_url': pdf_url,  # Store the PDF URL for reference
            'timestamp': time.time(),  # Use current timestamp
            'total_characters': sum(len(chunk) for chunk in chunks),
            'chunk_count': len(chunks)
        }
        
        async with aiofiles.open(cache_file, 'wb') as f:
            await f.write(pickle.dumps(cache_data))
        logging.info(f"Saved embeddings and enhanced chunks to cache: {cache_file}")
        logging.info(f"Cache contains {len(chunks)} chunks with {cache_data['total_characters']} total characters")
    except Exception as e:
        logging.warning(f"Failed to save embeddings to cache: {e}")

async def load_embeddings_from_cache(pdf_hash: str) -> tuple:
    """Load document chunks, embeddings, and enhanced chunk data from cache."""
    try:
        cache_file = CACHE_DIR / f"{pdf_hash}.pkl"
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'rb') as f:
                content = await f.read()
                cache_data = pickle.loads(content)
            logging.info(f"Loaded embeddings from cache: {cache_file}")
            logging.info(f"Cache contains {cache_data.get('chunk_count', len(cache_data['chunks']))} chunks with {cache_data.get('total_characters', 'unknown')} total characters")
            
            # Return chunks, embeddings, and enhanced chunk data
            chunks = cache_data['chunks']
            embeddings = cache_data['embeddings']
            chunks_data = cache_data.get('chunks_data', None)
            
            return chunks, embeddings, chunks_data
        return None, None, None
    except Exception as e:
        logging.warning(f"Failed to load embeddings from cache: {e}")
        return None, None, None

# --- Async Embedding Generation with Parallel Processing ---

async def generate_embeddings_nomic_async(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using Nomic's LangChain integration with fallback mechanism and parallel processing.
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
            
            # Use asyncio to run the sync embedding generation in a thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_embeddings_sync,
                texts
            )
            
            logging.info(f"Successfully generated all embeddings: shape {embeddings.shape}")
            return embeddings
            
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

def _generate_embeddings_sync(texts: List[str]) -> np.ndarray:
    """Synchronous embedding generation for use in thread pool."""
    embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5")
    embeddings = embeddings_model.embed_documents(texts)
    return np.array(embeddings)

async def generate_embeddings_parallel(texts: List[str], batch_size: int = 50) -> np.ndarray:
    """
    Generate embeddings with parallel processing for large text collections.
    """
    if len(texts) <= batch_size:
        return await generate_embeddings_nomic_async(texts)
    
    # Split texts into batches for parallel processing
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Process batches in parallel
    tasks = [generate_embeddings_nomic_async(batch) for batch in batches]
    
    try:
        results = await asyncio.gather(*tasks)
        # Concatenate all embeddings
        all_embeddings = np.vstack(results)
        logging.info(f"Generated embeddings for {len(texts)} texts using {len(batches)} parallel batches")
        return all_embeddings
    except Exception as e:
        logging.error(f"Error in parallel embedding generation: {e}")
        # Fallback to sequential processing
        return await generate_embeddings_nomic_async(texts)

# --- Async PDF Download and Processing ---

async def download_and_parse_pdf_async(pdf_url) -> tuple:
    """
    Asynchronously downloads a PDF from a URL, parses it, and splits it into text chunks.
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
    cached_metadata = await load_pdf_metadata_from_cache(url_hash)
    
    if cached_metadata:
        pdf_hash = cached_metadata['pdf_hash']
        logging.info(f"Found URL in cache, checking for chunks and embeddings (PDF hash: {pdf_hash})")
        
        # Try to load the chunks and embeddings
        cached_chunks, cached_embeddings, cached_chunks_data = await load_embeddings_from_cache(pdf_hash)
        if cached_chunks is not None:
            logging.info(f"✅ CACHE HIT: Skipping PDF download and parsing! Using cached data from {cached_metadata.get('cached_at', 'unknown time')}")
            logging.info(f"📄 Loaded {len(cached_chunks)} chunks with {cached_metadata.get('total_characters', 'unknown')} total characters")
            
            # Return the cached data with enhanced chunks if available
            return cached_chunks, pdf_hash, None, cached_chunks_data
    
    # If not in cache, proceed with download and parsing
    try:
        logging.info(f"📥 Downloading PDF from {pdf_url_str}")
        
        # Use aiohttp for async download
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url_str) as response:
                response.raise_for_status()
                pdf_content = await response.read()

        # Get PDF content and generate hash
        pdf_hash = get_pdf_hash(pdf_content)
        logging.info(f"PDF hash: {pdf_hash}")

        # Parse PDF in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        full_text = await loop.run_in_executor(
            cache_manager.thread_pool,
            _parse_pdf_sync,
            pdf_content
        )
        
        logging.info(f"Successfully parsed PDF. Total characters: {len(full_text)}")
        
        # Advanced structure-aware chunking
        chunks_data = advanced_chunking(full_text, chunk_size=1000, overlap=200)
        chunks = [chunk["text"] for chunk in chunks_data]  # Extract text for compatibility
        logging.info(f"Document split into {len(chunks)} chunks using advanced chunking.")
        
        # Save the URL metadata for future use (include chunks_data for enhanced retrieval)
        await save_pdf_metadata_to_cache(url_hash, pdf_url_str, pdf_hash, chunks, chunks_data)
        
        return chunks, pdf_hash, pdf_content, chunks_data

    except aiohttp.ClientError as e:
        logging.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not download or process the PDF from the provided URL.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during PDF processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse the PDF document.")

def _parse_pdf_sync(pdf_content: bytes) -> str:
    """Synchronous PDF parsing for use in thread pool."""
    pdf_file = BytesIO(pdf_content)
    reader = PdfReader(pdf_file)
    
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        # Add page metadata
        full_text += f"\n[PAGE {page_num + 1}]\n{page_text}\n"
    
    return full_text

# --- Advanced Chunking with Structure Awareness ---

def advanced_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Advanced chunking with structure awareness and metadata.
    Preserves section headers, page boundaries, and sentence integrity.
    """
    chunks = []
    
    # Split by pages first
    pages = text.split('[PAGE ')
    
    for page_idx, page_content in enumerate(pages):
        if not page_content.strip():
            continue
            
        # Extract page number
        if page_idx == 0:
            page_num = 1
            page_text = page_content
        else:
            lines = page_content.split('\n', 1)
            try:
                page_num = int(lines[0].split(']')[0]) if ']' in lines[0] else page_idx + 1
                page_text = lines[1] if len(lines) > 1 else ""
            except:
                page_num = page_idx + 1
                page_text = page_content
        
        # Structure-aware chunking within each page
        page_chunks = structure_aware_chunk(page_text, chunk_size, overlap, page_num)
        chunks.extend(page_chunks)
    
    return chunks

def structure_aware_chunk(text: str, chunk_size: int, overlap: int, page_num: int) -> List[Dict[str, Any]]:
    """
    Creates chunks while preserving document structure (headers, sections, sentences).
    """
    chunks = []
    
    # Detect section headers (lines with ALL CAPS or numbered sections)
    lines = text.split('\n')
    sections = []
    current_section = {"header": "", "content": "", "start_line": 0}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        is_header = (
            (len(line) > 5 and line.isupper() and len(line) < 100) or  # ALL CAPS headers
            (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(line) < 100) or  # Numbered sections
            (line.startswith(('Article', 'Section', 'Chapter', 'CLAUSE', 'DEFINITION')) and len(line) < 100)
        )
        
        if is_header and current_section["content"]:
            sections.append(current_section)
            current_section = {"header": line, "content": "", "start_line": i}
        elif is_header:
            current_section["header"] = line
            current_section["start_line"] = i
        else:
            current_section["content"] += line + " "
    
    # Add the last section
    if current_section["content"]:
        sections.append(current_section)
    
    # If no clear sections found, treat entire text as one section
    if not sections:
        sections = [{"header": f"Page {page_num}", "content": text, "start_line": 0}]
    
    # Create chunks from sections
    chunk_id = 0
    for section in sections:
        section_text = section["content"]
        section_header = section["header"]
        
        if len(section_text) <= chunk_size:
            # Small section - make it one chunk
            chunks.append({
                "id": chunk_id,
                "text": f"{section_header}\n{section_text}".strip(),
                "page": page_num,
                "section": section_header,
                "char_count": len(section_text),
                "chunk_type": "complete_section"
            })
            chunk_id += 1
        else:
            # Large section - split while preserving sentences
            section_chunks = sentence_aware_split(section_text, chunk_size, overlap)
            for i, chunk_text in enumerate(section_chunks):
                chunks.append({
                    "id": chunk_id,
                    "text": f"{section_header}\n{chunk_text}".strip() if i == 0 else chunk_text,
                    "page": page_num,
                    "section": section_header,
                    "char_count": len(chunk_text),
                    "chunk_type": "section_part",
                    "part_number": i + 1
                })
                chunk_id += 1
    
    return chunks

def sentence_aware_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks while preserving sentence boundaries.
    """
    import re
    
    # Simple sentence splitting (can be enhanced with NLP libraries)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle overlap
            if chunks and overlap > 0:
                # Get last few sentences for overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# --- Query Understanding & Intent Detection ---

def analyze_query_intent(question: str) -> Dict[str, Any]:
    """
    Analyze query to understand intent and extract key entities.
    """
    question_lower = question.lower()
    
    # Intent classification
    intent_patterns = {
        "definition": ["what is", "define", "meaning of", "definition"],
        "waiting_period": ["waiting period", "wait", "how long", "period", "duration"],
        "coverage": ["cover", "coverage", "included", "benefit", "eligible"],
        "exclusion": ["exclude", "not covered", "limitation", "restriction"],
        "procedure": ["how to", "process", "procedure", "steps", "apply"],
        "amount": ["amount", "cost", "price", "fee", "limit", "maximum"],
        "time": ["when", "date", "deadline", "grace period", "due"],
        "eligibility": ["eligible", "qualify", "criteria", "requirement"],
        "comparison": ["difference", "compare", "versus", "better", "option"]
    }
    
    detected_intents = []
    for intent, patterns in intent_patterns.items():
        if any(pattern in question_lower for pattern in patterns):
            detected_intents.append(intent)
    
    # Extract key entities
    entities = extract_entities(question)
    
    # Generate query variations for better retrieval
    query_variations = generate_query_variations(question, detected_intents, entities)
    
    return {
        "original_query": question,
        "intents": detected_intents,
        "entities": entities,
        "query_variations": query_variations,
        "search_terms": extract_search_terms(question)
    }

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract relevant entities from the query.
    """
    text_lower = text.lower()
    
    # Domain-specific entity patterns
    entities = {
        "medical_conditions": [],
        "procedures": [],
        "time_periods": [],
        "financial_terms": [],
        "policy_terms": []
    }
    
    # Medical conditions and procedures
    medical_terms = [
        "cataract", "surgery", "maternity", "pregnancy", "diabetes", "cancer",
        "heart", "dental", "vision", "prescription", "hospitalization",
        "emergency", "outpatient", "inpatient", "ayush", "alternative"
    ]
    
    for term in medical_terms:
        if term in text_lower:
            if term in ["surgery", "procedure"]:
                entities["procedures"].append(term)
            else:
                entities["medical_conditions"].append(term)
    
    # Time-related entities
    time_patterns = [
        "grace period", "waiting period", "30 days", "60 days", "90 days",
        "1 year", "2 years", "annual", "monthly", "quarterly"
    ]
    
    for pattern in time_patterns:
        if pattern in text_lower:
            entities["time_periods"].append(pattern)
    
    # Financial terms
    financial_terms = [
        "premium", "deductible", "copay", "coinsurance", "out-of-pocket",
        "maximum", "limit", "discount", "ncd", "no claim"
    ]
    
    for term in financial_terms:
        if term in text_lower:
            entities["financial_terms"].append(term)
    
    # Policy-specific terms
    policy_terms = [
        "coverage", "benefit", "exclusion", "rider", "add-on", "base policy",
        "family", "individual", "group", "employer"
    ]
    
    for term in policy_terms:
        if term in text_lower:
            entities["policy_terms"].append(term)
    
    return entities

def generate_query_variations(original_query: str, intents: List[str], entities: Dict[str, List[str]]) -> List[str]:
    """
    Generate query variations to improve retrieval recall.
    """
    variations = [original_query]
    
    # Add intent-specific variations
    if "waiting_period" in intents:
        variations.extend([
            original_query.replace("waiting period", "wait time"),
            original_query.replace("waiting period", "waiting time"),
            original_query + " duration",
            original_query + " time limit"
        ])
    
    if "coverage" in intents:
        variations.extend([
            original_query.replace("cover", "include"),
            original_query.replace("coverage", "benefit"),
            original_query + " eligible",
            original_query + " included"
        ])
    
    # Add entity-focused variations
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            if entity not in original_query.lower():
                variations.append(f"{original_query} {entity}")
    
    return list(set(variations))  # Remove duplicates

def extract_search_terms(query: str) -> List[str]:
    """
    Extract important search terms from the query.
    """
    import re
    
    # Remove common stop words but keep domain-specific ones
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
    
    # Extract words and phrases
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Keep important terms
    search_terms = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            search_terms.append(word)
    
    # Extract quoted phrases
    quoted_phrases = re.findall(r'"([^"]*)"', query)
    search_terms.extend(quoted_phrases)
    
    return search_terms

# --- Enhanced Context Retrieval with Hybrid Search & Reranking ---

async def retrieve_relevant_context_enhanced(query: str, corpus_chunks: List[Dict[str, Any]], corpus_embeddings: np.ndarray, doc_hash: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Enhanced retrieval with query understanding, hybrid search, and reranking.
    """
    # Step 1: Analyze query intent and extract entities
    query_analysis = analyze_query_intent(query)
    
    # Check cache first (using original query)
    query_hash = get_text_hash(query)
    cached_context = await cache_manager.get_retrieved_context_cache(query_hash, doc_hash)
    if cached_context:
        logging.info(f"Retrieved enhanced context from cache for query: '{query[:50]}...'")
        return json.loads(cached_context)
    
    # Step 2: Multi-query retrieval using query variations
    all_candidate_chunks = set()
    
    for query_variant in query_analysis["query_variations"][:3]:  # Limit to top 3 variations
        semantic_candidates = await semantic_search(query_variant, corpus_chunks, corpus_embeddings, top_k=top_k*2)
        keyword_candidates = keyword_search(query_variant, corpus_chunks, query_analysis["search_terms"], top_k=top_k)
        
        all_candidate_chunks.update(semantic_candidates)
        all_candidate_chunks.update(keyword_candidates)
    logging.info(f"Candidate chunks: {all_candidate_chunks}")
    # Step 3: Convert to list and rerank
    candidate_chunks = list(all_candidate_chunks)
    
    # Step 4: Intent-aware boosting
    boosted_chunks = apply_intent_boosting(candidate_chunks, query_analysis, corpus_chunks)
    
    # Step 5: Rerank using cross-encoder approach
    reranked_chunks = rerank_chunks(query, boosted_chunks, top_k)
    
    # Step 6: Build context with metadata
    context_result = build_enhanced_context(reranked_chunks, query_analysis, corpus_chunks)
    
    # Cache the result
    await cache_manager.set_retrieved_context_cache(query_hash, doc_hash, json.dumps(context_result))
    
    logging.info(f"Enhanced retrieval: {len(candidate_chunks)} candidates → {len(reranked_chunks)} final chunks")
    return context_result

async def semantic_search(query: str, corpus_chunks: List[Dict[str, Any]], corpus_embeddings: np.ndarray, top_k: int = 10) -> List[int]:
    """
    Semantic search using embeddings.
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
            os.environ["NOMIC_API_KEY"] = current_key
            
            # Generate embedding for the query using async execution
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_query_embedding_sync,
                query
            )
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return top_indices.tolist()
            
        except Exception as e:
            error_message = str(e).lower()
            logging.error(f"Error in semantic search with Nomic API (key #{attempt + 1}): {e}")
            
            if any(keyword in error_message for keyword in ['rate limit', 'quota', 'exhausted', 'unauthorized', '401', '429', '403']):
                logging.warning(f"API key #{attempt + 1} appears to be exhausted, trying next key...")
                api_key_manager.mark_nomic_key_failed(current_key)
                if attempt < max_retries - 1:
                    continue
            
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Semantic search service is currently unavailable."
            )

def keyword_search(query: str, corpus_chunks: List[Dict[str, Any]], search_terms: List[str], top_k: int = 10) -> List[int]:
    """
    Keyword-based search using TF-IDF like scoring.
    """
    from collections import Counter
    import re
    
    chunk_scores = []
    
    for i, chunk_data in enumerate(corpus_chunks):
        chunk_text = chunk_data["text"].lower()
        score = 0
        
        # Exact phrase matching (higher weight)
        if query.lower() in chunk_text:
            score += 10
        
        # Individual term matching
        for term in search_terms:
            term_count = len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', chunk_text))
            # TF-IDF like scoring: term frequency / document length
            tf = term_count / len(chunk_text.split()) if chunk_text.split() else 0
            score += tf * 5
        
        # Boost based on chunk metadata
        if chunk_data.get("chunk_type") == "complete_section":
            score *= 1.2  # Prefer complete sections
        
        chunk_scores.append((i, score))
    
    # Sort by score and return top_k indices
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in chunk_scores[:top_k] if score > 0]

def apply_intent_boosting(candidate_indices: List[int], query_analysis: Dict[str, Any], corpus_chunks: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """
    Apply intent-aware boosting to candidate chunks.
    """
    boosted_chunks = []
    
    for idx in candidate_indices:
        chunk_data = corpus_chunks[idx]
        chunk_text = chunk_data["text"].lower()
        boost_score = 1.0
        
        # Intent-specific boosting
        if "waiting_period" in query_analysis["intents"]:
            if any(term in chunk_text for term in ["waiting period", "wait", "period", "duration", "days", "months", "years"]):
                boost_score *= 1.5
        
        if "coverage" in query_analysis["intents"]:
            if any(term in chunk_text for term in ["cover", "benefit", "include", "eligible", "entitled"]):
                boost_score *= 1.3
        
        if "exclusion" in query_analysis["intents"]:
            if any(term in chunk_text for term in ["exclude", "not covered", "limitation", "restriction", "except"]):
                boost_score *= 1.4
        
        if "amount" in query_analysis["intents"]:
            if any(term in chunk_text for term in ["amount", "limit", "maximum", "minimum", "rupees", "rs", "$", "cost"]):
                boost_score *= 1.3
        
        # Entity-based boosting
        for entity_type, entities in query_analysis["entities"].items():
            for entity in entities:
                if entity.lower() in chunk_text:
                    boost_score *= 1.2
        
        # Section-based boosting
        section_name = chunk_data.get("section", "").lower()
        if any(intent_word in section_name for intent_word in ["benefit", "coverage", "exclusion", "limitation", "waiting"]):
            boost_score *= 1.1
        
        boosted_chunks.append((idx, boost_score))
    
    return boosted_chunks

def rerank_chunks(query: str, boosted_chunks: List[Tuple[int, float]], top_k: int) -> List[int]:
    """
    Simple reranking based on boosted scores and additional relevance signals.
    """
    # Sort by boosted scores
    boosted_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Take top candidates and apply final ranking
    top_candidates = boosted_chunks[:min(top_k * 2, len(boosted_chunks))]
    
    # For now, return top_k based on boosted scores
    # In future, this can be enhanced with cross-encoder models
    return [idx for idx, score in top_candidates[:top_k]]

def build_enhanced_context(chunk_indices: List[int], query_analysis: Dict[str, Any], corpus_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build enhanced context with metadata and structure.
    """
    context_chunks = []
    pages_referenced = set()
    sections_referenced = set()
    
    for idx in chunk_indices:
        chunk_data = corpus_chunks[idx]
        context_chunks.append({
            "text": chunk_data["text"],
            "page": chunk_data.get("page", "unknown"),
            "section": chunk_data.get("section", "unknown"),
            "chunk_type": chunk_data.get("chunk_type", "unknown"),
            "relevance_score": 1.0  # This would come from reranking
        })
        
        pages_referenced.add(chunk_data.get("page", "unknown"))
        sections_referenced.add(chunk_data.get("section", "unknown"))
    logging.info(f"Context chunks: {context_chunks}")
    # Create structured context
    context_text = "\n---\n".join([chunk["text"] for chunk in context_chunks])
    
    return {
        "context": context_text,
        "metadata": {
            "query_analysis": query_analysis,
            "chunks_count": len(context_chunks),
            "pages_referenced": list(pages_referenced),
            "sections_referenced": list(sections_referenced),
            "retrieval_method": "enhanced_hybrid"
        },
        "chunks": context_chunks
    }

# --- Async Context Retrieval with Caching ---

async def retrieve_relevant_context_async(query: str, corpus_chunks: List[str], corpus_embeddings: np.ndarray, doc_hash: str, top_k: int = 5) -> str:
    """
    Retrieves the most relevant context chunks for a given query using semantic search with aggressive caching.
    """
    # Check cache first
    query_hash = get_text_hash(query)
    cached_context = await cache_manager.get_retrieved_context_cache(query_hash, doc_hash)
    if cached_context:
        logging.info(f"Retrieved context from cache for query: '{query[:50]}...'")
        return cached_context
    
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
            
            # Generate embedding for the query using async execution
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_query_embedding_sync,
                query
            )
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            context_chunks = [corpus_chunks[idx] for idx in top_indices]
            relevant_context = "\n---\n".join(context_chunks)
            
            # Cache the result
            await cache_manager.set_retrieved_context_cache(query_hash, doc_hash, relevant_context)
            
            logging.info(f"Retrieved {len(top_indices)} relevant chunks for the query: '{query[:50]}...'")
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

def _generate_query_embedding_sync(query: str) -> np.ndarray:
    """Synchronous query embedding generation for use in thread pool."""
    embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1.5")
    return np.array(embeddings_model.embed_query(query))

def _generate_llm_answer_sync(client: Mistral, prompt: str) -> str:
    """Synchronous LLM answer generation for use in thread pool."""
    chat_response = client.chat.complete(
        model=MISTRAL_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return chat_response.choices[0].message.content.strip()

# --- Query Analysis and Intent Detection ---

def analyze_query_intent(query: str) -> Dict[str, Any]:
    """
    Analyze query for intent detection and entity extraction.
    """
    query_lower = query.lower()
    
    # Intent detection based on keywords
    intents = []
    if any(word in query_lower for word in ['grace period', 'waiting period', 'time', 'days', 'months']):
        intents.append('time_period')
    if any(word in query_lower for word in ['coverage', 'cover', 'covered', 'benefit']):
        intents.append('coverage')
    if any(word in query_lower for word in ['cost', 'premium', 'price', 'amount']):
        intents.append('cost')
    if any(word in query_lower for word in ['exclude', 'exclusion', 'not covered']):
        intents.append('exclusion')
    if any(word in query_lower for word in ['claim', 'process', 'procedure']):
        intents.append('process')
    
    # Entity extraction (simple keyword-based)
    entities = []
    medical_terms = ['maternity', 'cataract', 'surgery', 'hospital', 'ayush', 'organ donor', 'pre-existing']
    insurance_terms = ['ncd', 'no claim discount', 'premium', 'policy', 'deductible']
    
    for term in medical_terms + insurance_terms:
        if term in query_lower:
            entities.append(term)
    
    return {
        "intents": intents,
        "entities": entities,
        "query_type": "complex" if len(intents) > 1 else "simple"
    }

# --- Hybrid Retrieval Implementation ---

async def perform_hybrid_retrieval(query: str, query_embedding: np.ndarray, chunks_data: List[Dict[str, Any]], 
                                 corpus_embeddings: np.ndarray, query_analysis: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Perform hybrid retrieval combining semantic search with keyword matching and intent-aware boosting.
    """
    # Semantic similarity scores
    similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    
    # Keyword matching scores (simple TF-IDF style)
    keyword_scores = calculate_keyword_scores(query, chunks_data)
    
    # Intent-aware boosting
    intent_scores = calculate_intent_scores(query_analysis, chunks_data)
    
    # Combine scores with weights
    semantic_weight = 0.6
    keyword_weight = 0.3
    intent_weight = 0.1
    
    combined_scores = (semantic_weight * similarities + 
                      keyword_weight * keyword_scores + 
                      intent_weight * intent_scores)
    
    # Get top-k indices
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    # Build context and collect metadata
    context_chunks = []
    pages_referenced = set()
    sections_referenced = set()
    
    for idx in top_indices:
        chunk_data = chunks_data[idx]
        context_chunks.append(chunk_data['text'])
        
        if 'page' in chunk_data:
            pages_referenced.add(f"Page {chunk_data['page']}")
        if 'section' in chunk_data:
            sections_referenced.add(chunk_data['section'])
    
    relevant_context = "\n---\n".join(context_chunks)
    
    return {
        "context": relevant_context,
        "metadata": {
            "query_analysis": query_analysis,
            "retrieval_method": "hybrid_semantic_keyword",
            "chunks_count": len(top_indices),
            "pages_referenced": list(pages_referenced),
            "sections_referenced": list(sections_referenced),
            "similarity_scores": similarities[top_indices].tolist(),
            "combined_scores": combined_scores[top_indices].tolist()
        }
    }

def calculate_keyword_scores(query: str, chunks_data: List[Dict[str, Any]]) -> np.ndarray:
    """
    Calculate keyword matching scores for chunks.
    """
    query_words = set(query.lower().split())
    scores = []
    
    for chunk_data in chunks_data:
        chunk_words = set(chunk_data['text'].lower().split())
        # Simple Jaccard similarity
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        score = intersection / union if union > 0 else 0
        scores.append(score)
    
    return np.array(scores)

def calculate_intent_scores(query_analysis: Dict[str, Any], chunks_data: List[Dict[str, Any]]) -> np.ndarray:
    """
    Calculate intent-aware boosting scores for chunks.
    """
    scores = np.zeros(len(chunks_data))
    
    if not query_analysis.get('entities'):
        return scores
    
    for i, chunk_data in enumerate(chunks_data):
        chunk_text_lower = chunk_data['text'].lower()
        
        # Boost chunks containing query entities
        entity_matches = sum(1 for entity in query_analysis['entities'] if entity in chunk_text_lower)
        scores[i] = entity_matches / len(query_analysis['entities']) if query_analysis['entities'] else 0
    
    return scores

# --- Enhanced Context Retrieval with Hybrid Search ---

async def retrieve_relevant_context_enhanced(query: str, chunks_data: List[Dict[str, Any]], corpus_embeddings: np.ndarray, doc_hash: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Enhanced retrieval with query understanding, hybrid search, and metadata extraction.
    """
    # Check cache first
    query_hash = get_text_hash(query)
    cached_context = await cache_manager.get_retrieved_context_cache(query_hash, doc_hash)
    if cached_context:
        logging.info(f"Retrieved enhanced context from cache for query: '{query[:50]}...'")
        try:
            return json.loads(cached_context)
        except json.JSONDecodeError:
            # Fallback to simple context if cached data is in old format
            return {
                "context": cached_context,
                "metadata": {
                    "query_analysis": {"intents": [], "entities": []},
                    "retrieval_method": "cached_simple",
                    "chunks_count": 0,
                    "pages_referenced": [],
                    "sections_referenced": []
                }
            }
    
    # Analyze query for intent and entities
    query_analysis = analyze_query_intent(query)
    
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
            
            # Generate embedding for the query using async execution
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_query_embedding_sync,
                query
            )
            
            # Perform hybrid retrieval (semantic + keyword matching)
            context_result = await perform_hybrid_retrieval(
                query=query,
                query_embedding=query_embedding,
                chunks_data=chunks_data,
                corpus_embeddings=corpus_embeddings,
                query_analysis=query_analysis,
                top_k=top_k
            )
            
            # Cache the result
            await cache_manager.set_retrieved_context_cache(query_hash, doc_hash, json.dumps(context_result))
            
            logging.info(f"Retrieved {context_result['metadata']['chunks_count']} enhanced chunks for query: '{query[:50]}...'")
            return context_result
            
        except Exception as e:
            error_message = str(e).lower()
            logging.error(f"Error in enhanced semantic search with Nomic API (key #{attempt + 1}): {e}")
            
            # Check if it's an API exhaustion or auth error
            if any(keyword in error_message for keyword in ['rate limit', 'quota', 'exhausted', 'unauthorized', '401', '429', '403']):
                logging.warning(f"API key #{attempt + 1} appears to be exhausted, trying next key...")
                api_key_manager.mark_nomic_key_failed(current_key)
                if attempt < max_retries - 1:
                    continue
            
            # If it's not an API key issue or we've exhausted all keys, raise the error
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Enhanced semantic search service is currently unavailable."
            )
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Failed to perform enhanced semantic search with any available API key."
    )

# --- Async LLM Answer Generation with Caching ---

# --- Enhanced LLM Answer Generation with Evidence & Reasoning ---

async def generate_answer_with_llm_enhanced(context_result: Dict[str, Any], question: str, client: Mistral) -> Dict[str, Any]:
    """
    Enhanced answer generation with evidence tracking and reasoning.
    """
    context = context_result["context"]
    metadata = context_result["metadata"]
    
    # Check cache first
    context_hash = get_context_hash(context)
    question_hash = get_question_hash(question)
    
    cached_answer = await cache_manager.get_llm_answer_cache(context_hash, question_hash)
    if cached_answer:
        logging.info(f"Retrieved enhanced LLM answer from cache for question: '{question[:50]}...'")
        try:
            # Try to parse as JSON (new format)
            return json.loads(cached_answer)
        except json.JSONDecodeError:
            # Fallback for old string format - convert to new format
            logging.info("Converting old cache format to enhanced format")
            return {
                "answer": cached_answer,
                "evidence": [],
                "page_references": metadata.get("pages_referenced", []),
                "section_references": metadata.get("sections_referenced", []),
                "confidence": "medium",
                "reasoning": "Answer retrieved from cache (old format)",
                "metadata": {
                    "query_analysis": metadata.get("query_analysis", {"intents": [], "entities": []}),
                    "retrieval_method": "cached_legacy",
                    "chunks_used": metadata.get("chunks_count", 0)
                }
            }
    
    max_retries = len(api_key_manager.mistral_keys)
    
    # Enhanced prompt with structure awareness
    prompt = f"""
    **Role:** You are a highly intelligent AI assistant specializing in document analysis for insurance, legal, and HR domains.
    
    **Task:** Answer the user's question based *exclusively* on the provided context below. Provide evidence and reasoning for your answer.
    
    **Context from Document:**
    ---
    {context}
    ---
    
    **Query Analysis:**
    - Intent: {', '.join(metadata['query_analysis']['intents']) if metadata['query_analysis']['intents'] else 'General inquiry'}
    - Entities: {metadata['query_analysis']['entities']}
    - Pages Referenced: {metadata['pages_referenced']}
    - Sections Referenced: {metadata['sections_referenced']}
    
    **User's Question:**
    {question}
    
    **Instructions:**
    1. Answer the question directly and concisely
    2. Cite specific evidence from the context (mention page/section when available)
    3. If multiple pieces of information are relevant, organize them clearly
    4. Indicate confidence level in your answer
    5. If the answer is not found, clearly state: "The information is not available in the provided document context."
    
    **Response Format:**
    Provide your response as a JSON object with the following structure:
    {{
        "answer": "Your direct answer here",
        "evidence": ["Specific quote 1 from document", "Specific quote 2 from document"],
        "page_references": ["Page X", "Page Y"],
        "section_references": ["Section Name 1", "Section Name 2"],
        "confidence": "high/medium/low",
        "reasoning": "Brief explanation of how you arrived at this answer"
    }}
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
            
            # Use asyncio to run the sync LLM call in a thread pool
            loop = asyncio.get_event_loop()
            raw_answer = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_llm_answer_sync,
                current_client,
                prompt
            )
            
            # Parse the JSON response
            try:
                # Clean the response to extract JSON from markdown code blocks
                clean_answer = raw_answer.strip()
                
                # Remove markdown code block markers if present
                if clean_answer.startswith('```json'):
                    clean_answer = clean_answer[7:]  # Remove ```json
                if clean_answer.startswith('```'):
                    clean_answer = clean_answer[3:]   # Remove ```
                if clean_answer.endswith('```'):
                    clean_answer = clean_answer[:-3]  # Remove closing ```
                
                # Remove any leading/trailing whitespace and newlines
                clean_answer = clean_answer.strip()
                
                # Try to parse the cleaned JSON
                answer_data = json.loads(clean_answer)
                
                # Validate required fields and add defaults if missing
                enhanced_answer = {
                    "answer": answer_data.get("answer", raw_answer),
                    "evidence": answer_data.get("evidence", []),
                    "page_references": answer_data.get("page_references", []),
                    "section_references": answer_data.get("section_references", []),
                    "confidence": answer_data.get("confidence", "medium"),
                    "reasoning": answer_data.get("reasoning", ""),
                    "metadata": {
                        "query_analysis": metadata["query_analysis"],
                        "retrieval_method": metadata["retrieval_method"],
                        "chunks_used": metadata["chunks_count"]
                    }
                }
                
            except json.JSONDecodeError as e:
                # Fallback if LLM doesn't return valid JSON
                logging.warning(f"Failed to parse LLM response as JSON: {e}")
                logging.warning(f"Raw response: {raw_answer[:200]}...")
                enhanced_answer = {
                    "answer": raw_answer,
                    "evidence": [],
                    "page_references": metadata["pages_referenced"],
                    "section_references": metadata["sections_referenced"],
                    "confidence": "medium",
                    "reasoning": "Answer provided without structured evidence (parsing failed)",
                    "metadata": {
                        "query_analysis": metadata["query_analysis"],
                        "retrieval_method": metadata["retrieval_method"],
                        "chunks_used": metadata["chunks_count"]
                    }
                }
            
            # Cache the enhanced answer
            await cache_manager.set_llm_answer_cache(context_hash, question_hash, json.dumps(enhanced_answer))
            
            logging.info("Successfully generated enhanced answer with Mistral LLM.")
            return enhanced_answer
            
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

async def generate_answer_with_llm_async(context: str, question: str, client: Mistral) -> str:
    """
    Generates an answer using the Mistral LLM with fallback mechanism and aggressive caching.
    """
    # Check cache first
    context_hash = get_context_hash(context)
    question_hash = get_question_hash(question)
    
    cached_answer = await cache_manager.get_llm_answer_cache(context_hash, question_hash)
    if cached_answer:
        logging.info(f"Retrieved LLM answer from cache for question: '{question[:50]}...'")
        return cached_answer
    
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
            
            # Use asyncio to run the sync LLM call in a thread pool
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                cache_manager.thread_pool,
                _generate_llm_answer_sync,
                current_client,
                prompt
            )
            
            # Cache the answer
            await cache_manager.set_llm_answer_cache(context_hash, question_hash, answer)
            
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

# --- Enhanced Parallel Question Processing ---

async def process_questions_parallel_enhanced(questions: List[str], chunks_data: List[Dict[str, Any]], corpus_embeddings: np.ndarray, doc_hash: str, client: Mistral) -> List[Dict[str, Any]]:
    """
    Process multiple questions in parallel with enhanced retrieval and detailed responses.
    """
    if len(questions) == 1:
        # Single question - no need for parallelization overhead
        context_result = await retrieve_relevant_context_enhanced(
            query=questions[0],
            chunks_data=chunks_data,
            corpus_embeddings=corpus_embeddings,
            doc_hash=doc_hash
        )
        answer_result = await generate_answer_with_llm_enhanced(
            context_result=context_result,
            question=questions[0],
            client=client
        )
        return [answer_result]
    
    # Multiple questions - process in parallel
    async def process_single_question_enhanced(question: str) -> Dict[str, Any]:
        logging.info(f"Processing enhanced question: '{question[:50]}...'")
        
        # Retrieve context relevant to the current question with enhanced features
        context_result = await retrieve_relevant_context_enhanced(
            query=question,
            chunks_data=chunks_data,
            corpus_embeddings=corpus_embeddings,
            doc_hash=doc_hash
        )
        
        # Generate the answer using enhanced LLM with evidence tracking
        answer_result = await generate_answer_with_llm_enhanced(
            context_result=context_result,
            question=question,
            client=client
        )
        return answer_result
    
    # Process all questions in parallel
    tasks = [process_single_question_enhanced(question) for question in questions]
    answer_results = await asyncio.gather(*tasks)
    
    logging.info(f"Processed {len(questions)} questions in parallel with enhanced features")
    return answer_results

async def process_questions_parallel(questions: List[str], doc_chunks: List[str], corpus_embeddings: np.ndarray, doc_hash: str, client: Mistral) -> List[str]:
    """
    Process multiple questions in parallel for improved performance.
    """
    if len(questions) == 1:
        # Single question - no need for parallelization overhead
        context = await retrieve_relevant_context_async(
            query=questions[0],
            corpus_chunks=doc_chunks,
            corpus_embeddings=corpus_embeddings,
            doc_hash=doc_hash
        )
        answer = await generate_answer_with_llm_async(
            context=context,
            question=questions[0],
            client=client
        )
        return [answer]
    
    # Multiple questions - process in parallel
    async def process_single_question(question: str) -> str:
        logging.info(f"Processing question: '{question[:50]}...'")
        
        # Retrieve context relevant to the current question
        context = await retrieve_relevant_context_async(
            query=question,
            corpus_chunks=doc_chunks,
            corpus_embeddings=corpus_embeddings,
            doc_hash=doc_hash
        )
        
        # Generate the answer using the LLM
        answer = await generate_answer_with_llm_async(
            context=context,
            question=question,
            client=client
        )
        return answer
    
    # Process all questions in parallel
    tasks = [process_single_question(question) for question in questions]
    answers = await asyncio.gather(*tasks)
    
    logging.info(f"Processed {len(questions)} questions in parallel")
    return answers

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "LLM-Powered Intelligent Query–Retrieval System (Enhanced with Aggressive Caching & Async)",
        "status": "online",
        "version": "2.0.0",
        "enhancements": {
            "aggressive_caching": {
                "layers": ["memory", "redis", "file_based"],
                "cached_data": ["llm_answers", "retrieved_contexts", "embeddings", "pdf_metadata"]
            },
            "async_processing": {
                "pdf_download": "aiohttp",
                "file_operations": "aiofiles", 
                "parallel_embeddings": "ThreadPoolExecutor",
                "parallel_questions": "asyncio.gather"
            },
            "performance_optimizations": [
                "multi_layer_caching",
                "parallel_question_processing",
                "async_io_operations",
                "background_cache_cleanup",
                "in_memory_cache_with_ttl"
            ]
        },
        "configuration": {
            "embeddings": "Nomic API (nomic-embed-text-v1.5) with fallback",
            "llm": "Mistral API with fallback",
            "caching": "Redis + File-based + Memory",
            "api_keys": {
                "mistral_total": len(api_key_manager.mistral_keys),
                "nomic_total": len(api_key_manager.nomic_keys)
            }
        },
        "endpoints": {
            "main": "/hackrx/run (POST) - Standard RAG processing",
            "enhanced": "/hackrx/run/enhanced (POST) - Enhanced RAG with evidence & reasoning",
            "health": "/health (GET)",
            "cache": "/cache/status (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get("/cache/status")
async def cache_status():
    """Get cache status and information including Redis, file-based, and memory cache."""
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        url_cache_files = list(CACHE_DIR.glob("url_*.json"))
        
        cache_info = []
        for cache_file in cache_files:
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    content = await f.read()
                    cache_data = pickle.loads(content)
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
                async with aiofiles.open(url_file, 'r') as f:
                    content = await f.read()
                    url_data = json.loads(content)
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
        
        # Redis status
        redis_status = "not_configured"
        redis_info = {}
        if cache_manager.redis_client:
            try:
                # Test Redis connection
                await cache_manager.redis_client.ping()
                redis_status = "connected"
                redis_info = {
                    "connection": "active",
                    "url": "configured" if os.getenv("REDIS_URL") else "not_configured"
                }
            except Exception as e:
                redis_status = "connection_failed"
                redis_info = {"error": str(e)}
        
        return {
            "cache_directory": str(CACHE_DIR.absolute()),
            "file_based_cache": {
                "embedding_cache_files": len(cache_files),
                "url_cache_files": len(url_cache_files),
                "embedding_cache_details": cache_info,
                "url_cache_details": url_cache_info,
                "total_cached_pdfs": len(cache_files)
            },
            "redis_cache": {
                "status": redis_status,
                "info": redis_info
            },
            "memory_cache": {
                "entries": len(cache_manager.memory_cache),
                "max_size": cache_manager.max_memory_cache_size
            },
            "caching_layers": {
                "memory": "enabled",
                "redis": redis_status,
                "file_based": "enabled"
            }
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
    Main API endpoint to process documents and answer questions with aggressive caching and parallel processing.
    Enhanced with:
    - Aggressive caching (Redis + file-based + memory)
    - Asynchronous I/O operations
    - Parallel question processing
    - Parallel embedding generation
    - CSV logging for all queries and responses
    
    It follows the complete RAG workflow:
    1. Authenticates the request.
    2. Asynchronously downloads and parses the document.
    3. Creates embeddings for document chunks with parallel processing.
    4. For each question, retrieves relevant context with caching.
    5. Generates answers using the LLM with caching and parallel processing.
    6. Returns all answers in a structured JSON response.
    7. Logs all query-response pairs to CSV with timestamps.
    """
    start_time = time.time()
    
    # 1. Authentication
    auth_header = request.headers.get("Authorization")
    if auth_header != EXPECTED_AUTH_TOKEN:
        logging.warning("Authentication failed. Invalid token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    logging.info("Authentication successful.")

    # 2. Document Processing (Async)
    doc_chunks, pdf_hash, pdf_content, chunks_data = await download_and_parse_pdf_async(submission.documents)
    
    # 3. Embedding Generation with Aggressive Caching
    logging.info("Checking for cached embeddings...")
    cached_chunks, cached_embeddings, cached_chunks_data = await load_embeddings_from_cache(pdf_hash)
    
    if cached_chunks is not None and cached_embeddings is not None:
        logging.info("Using cached embeddings!")
        corpus_embeddings = cached_embeddings
        doc_chunks = cached_chunks
        
        # Handle enhanced chunk data
        if cached_chunks_data is not None:
            chunks_data = cached_chunks_data
            logging.info("Using cached enhanced chunk metadata")
        else:
            # Fallback: create basic chunk data from cached chunks
            chunks_data = [{"id": i, "text": chunk, "page": "unknown", "section": "unknown", "chunk_type": "legacy"} 
                          for i, chunk in enumerate(cached_chunks)]
            logging.info("Created basic chunk metadata for legacy cache")
    else:
        logging.info("Generating new embeddings for document chunks with parallel processing...")
        # Use parallel embedding generation for better performance
        corpus_embeddings = await generate_embeddings_parallel(doc_chunks)
        # Save with PDF URL for reference and enhanced chunk data
        await save_embeddings_to_cache(pdf_hash, doc_chunks, corpus_embeddings, str(submission.documents), chunks_data)
        logging.info("Embeddings generated and cached successfully.")
    
    # 4. & 5. Process Questions with Enhanced Parallel Processing and Caching
    answer_results = await process_questions_parallel_enhanced(
        questions=submission.questions,
        chunks_data=chunks_data,
        corpus_embeddings=corpus_embeddings,
        doc_hash=pdf_hash,
        client=app.state.mistral_client
    )
    
    # Extract simple answers for backward compatibility with proper formatting
    final_answers = []
    for i, result in enumerate(answer_results):
        answer = result["answer"]
        
        # If the answer looks like raw markdown JSON, extract the actual answer
        if answer.startswith('```json') or answer.startswith('```'):
            try:
                # Clean the markdown and extract the JSON
                clean_answer = answer.strip()
                if clean_answer.startswith('```json'):
                    clean_answer = clean_answer[7:]
                if clean_answer.startswith('```'):
                    clean_answer = clean_answer[3:]
                if clean_answer.endswith('```'):
                    clean_answer = clean_answer[:-3]
                
                clean_answer = clean_answer.strip()
                parsed_data = json.loads(clean_answer)
                answer = parsed_data.get("answer", answer)
                
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, keep the original answer
                pass
        
        final_answers.append(answer)
        
        # Log each question-answer pair to CSV
        question_time = time.time()
        response_time_ms = (question_time - start_time) * 1000
        
        # Extract confidence score if available - handle different data structures
        confidence_score = result.get("confidence_score")
        if not confidence_score and "evidence" in result:
            evidence = result["evidence"]
            if isinstance(evidence, dict):
                confidence_score = evidence.get("confidence_score")
            elif isinstance(evidence, list) and evidence:
                # If evidence is a list, look for confidence in the first item
                if isinstance(evidence[0], dict):
                    confidence_score = evidence[0].get("confidence_score")
        
        # Log to CSV asynchronously
        asyncio.create_task(csv_logger.log_query_response(
            document_url=str(submission.documents),
            document_hash=pdf_hash,
            question=submission.questions[i],
            answer=answer,
            response_time_ms=response_time_ms,
            endpoint_type="standard",
            cache_used=cached_chunks is not None,
            chunk_count=len(chunks_data),
            confidence_score=confidence_score
        ))
        
    # 6. Return Structured JSON Output
    return SubmissionResponse(answers=final_answers)

@app.post("/hackrx/run/enhanced", response_model=EnhancedSubmissionResponse)
async def run_submission_enhanced(request: Request, submission: SubmissionRequest):
    """
    Enhanced API endpoint that returns detailed answers with evidence, reasoning, and metadata.
    Includes all the enhanced features: advanced chunking, hybrid retrieval, query understanding,
    intent detection, evidence tracking, structured responses, and CSV logging.
    """
    start_time = time.time()
    
    # 1. Authentication
    auth_header = request.headers.get("Authorization")
    if auth_header != EXPECTED_AUTH_TOKEN:
        logging.warning("Authentication failed. Invalid token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    logging.info("Authentication successful - Enhanced endpoint")

    # 2. Document Processing (Async)
    doc_chunks, pdf_hash, pdf_content, chunks_data = await download_and_parse_pdf_async(submission.documents)
    
    # 3. Embedding Generation with Aggressive Caching
    logging.info("Checking for cached embeddings...")
    cached_chunks, cached_embeddings, cached_chunks_data = await load_embeddings_from_cache(pdf_hash)
    
    if cached_chunks is not None and cached_embeddings is not None:
        logging.info("Using cached embeddings!")
        corpus_embeddings = cached_embeddings
        doc_chunks = cached_chunks
        
        # Handle enhanced chunk data
        if cached_chunks_data is not None:
            chunks_data = cached_chunks_data
            logging.info("Using cached enhanced chunk metadata")
        else:
            # Fallback: create basic chunk data from cached chunks
            chunks_data = [{"id": i, "text": chunk, "page": "unknown", "section": "unknown", "chunk_type": "legacy"} 
                          for i, chunk in enumerate(cached_chunks)]
            logging.info("Created basic chunk metadata for legacy cache")
    else:
        logging.info("Generating new embeddings for document chunks with parallel processing...")
        # Use parallel embedding generation for better performance
        corpus_embeddings = await generate_embeddings_parallel(doc_chunks)
        # Save with PDF URL for reference and enhanced chunk data
        await save_embeddings_to_cache(pdf_hash, doc_chunks, corpus_embeddings, str(submission.documents), chunks_data)
        logging.info("Embeddings generated and cached successfully.")
    
    # 4. & 5. Process Questions with Enhanced Parallel Processing and Caching
    answer_results = await process_questions_parallel_enhanced(
        questions=submission.questions,
        chunks_data=chunks_data,
        corpus_embeddings=corpus_embeddings,
        doc_hash=pdf_hash,
        client=app.state.mistral_client
    )
    
    # Extract simple answers for backward compatibility
    final_answers = [result["answer"] for result in answer_results]
    
    # Log each question-answer pair to CSV for enhanced endpoint
    for i, result in enumerate(answer_results):
        question_time = time.time()
        response_time_ms = (question_time - start_time) * 1000
        
        # Extract enhanced data - handle different data structures
        confidence_score = result.get("confidence_score")
        if not confidence_score and "evidence" in result:
            evidence = result["evidence"]
            if isinstance(evidence, dict):
                confidence_score = evidence.get("confidence_score")
            elif isinstance(evidence, list) and evidence:
                # If evidence is a list, look for confidence in the first item
                if isinstance(evidence[0], dict):
                    confidence_score = evidence[0].get("confidence_score")
        
        # Log to CSV asynchronously
        asyncio.create_task(csv_logger.log_query_response(
            document_url=str(submission.documents),
            document_hash=pdf_hash,
            question=submission.questions[i],
            answer=result["answer"],
            response_time_ms=response_time_ms,
            endpoint_type="enhanced",
            cache_used=cached_chunks is not None,
            chunk_count=len(chunks_data),
            confidence_score=confidence_score
        ))
    
    # Compile metadata
    response_metadata = {
        "document_info": {
            "pdf_hash": pdf_hash,
            "total_chunks": len(chunks_data),
            "total_characters": sum(len(chunk["text"]) for chunk in chunks_data),
            "processing_method": "enhanced_rag_v2"
        },
        "processing_stats": {
            "questions_processed": len(submission.questions),
            "parallel_processing": len(submission.questions) > 1,
            "cache_used": cached_chunks is not None,
            "enhanced_features": [
                "advanced_chunking",
                "query_understanding", 
                "hybrid_retrieval",
                "intent_detection",
                "evidence_tracking",
                "structure_awareness"
            ]
        }
    }
        
    # 6. Return Enhanced Structured JSON Output
    return EnhancedSubmissionResponse(
        answers=final_answers,
        detailed_answers=answer_results,
        metadata=response_metadata
    )

# --- To run the server ---
if __name__ == "__main__":
    # Use PORT environment variable from Render, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)