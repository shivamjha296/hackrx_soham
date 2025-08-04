# Migration Guide: From SentenceTransformers to Mistral Embeddings

## Key Changes Made:

### 1. **Imports Updated**
```python
# OLD:
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# NEW:
from sklearn.metrics.pairwise import cosine_similarity
# (mistralai import remains the same)
```

### 2. **Model Initialization Removed**
```python
# OLD: Loading local embedding model on startup
app.state.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# NEW: Only Mistral client needed (no local model)
app.state.mistral_client = Mistral(api_key=api_key)
```

### 3. **Embedding Generation**
```python
# OLD: Local model encoding
corpus_embeddings = app.state.embedding_model.encode(doc_chunks)

# NEW: Mistral API call
corpus_embeddings = generate_embeddings(doc_chunks, app.state.mistral_client)
```

### 4. **Semantic Search**
```python
# OLD: SentenceTransformers utility
query_embedding = model.encode([query])
hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

# NEW: Manual cosine similarity calculation
query_embedding = generate_embeddings([query], client)
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
top_indices = np.argsort(similarities)[-top_k:][::-1]
```

## Benefits of the New Approach:

1. **No Local Model Loading**: Reduces memory usage and startup time
2. **API-based**: More scalable and doesn't require model downloads
3. **Consistent Provider**: Both embeddings and LLM from Mistral
4. **Reduced Dependencies**: Fewer heavy packages in requirements.txt
5. **Batch Processing**: Handles API limits gracefully with automatic batching

## Important: Batch Size Limits

Mistral's embedding API has batch size limits. The implementation now includes:
- **Automatic batch processing** for large document sets
- **Conservative batch size** (20 texts per request) to avoid API errors
- **Progress logging** for transparency during processing

```python
# Configuration
MISTRAL_EMBEDDING_BATCH_SIZE = 20  # Conservative batch size
MISTRAL_EMBEDDING_MODEL = "mistral-embed"

# The function automatically splits large text lists into batches
def generate_embeddings(texts: List[str], client: Mistral, batch_size: int = 20):
    # Processes texts in batches to avoid "Batch size too large" errors
```

## Requirements.txt Changes:
```
# REMOVED:
sentence-transformers

# ADDED:
scikit-learn  # For cosine_similarity calculation
```

## Environment Variables Needed:
- `MISTRAL_API_KEY` (same as before)

The functionality remains exactly the same - only the implementation details changed for better scalability and reduced resource usage.
