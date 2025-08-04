#!/usr/bin/env python3
"""
Test script to verify Mistral embeddings functionality.
This script demonstrates how to use Mistral's embedding API as shown in your example.
"""

import os
import numpy as np
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity

def test_mistral_embeddings():
    """Test Mistral embeddings API with batch processing."""
    
    # Initialize Mistral client
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("MISTRAL_API_KEY environment variable not set!")
        return
    
    client = Mistral(api_key=api_key)
    model = "mistral-embed"
    
    # Test texts (small batch)
    test_texts = [
        "Embed this sentence.", 
        "As well as this one.",
        "This is a different sentence about embeddings.",
        "Machine learning is fascinating."
    ]
    
    print("Testing Mistral embeddings with small batch...")
    
    try:
        # Generate embeddings (small batch - should work)
        embeddings_response = client.embeddings.create(
            model=model,
            inputs=test_texts
        )
        
        print(f"✓ Successfully generated embeddings for {len(test_texts)} texts")
        print(f"✓ Response contains {len(embeddings_response.data)} embeddings")
        
        # Extract embeddings
        embeddings = [item.embedding for item in embeddings_response.data]
        embeddings_array = np.array(embeddings)
        
        print(f"✓ Embeddings shape: {embeddings_array.shape}")
        print(f"✓ Each embedding has {len(embeddings[0])} dimensions")
        
        # Test similarity calculation
        query_embedding = np.array([embeddings[0]])  # First text as query
        corpus_embeddings = embeddings_array[1:]     # Rest as corpus
        
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        print(f"✓ Cosine similarities: {similarities}")
        
        # Find most similar
        most_similar_idx = np.argmax(similarities)
        print(f"✓ Most similar to '{test_texts[0]}' is: '{test_texts[most_similar_idx + 1]}'")
        
        print("\n✅ Small batch test passed!")
        
        # Test batch size limits
        print("\nTesting batch size limits...")
        large_batch = [f"Test sentence number {i}" for i in range(100)]
        
        try:
            large_response = client.embeddings.create(
                model=model,
                inputs=large_batch
            )
            print(f"✓ Large batch of {len(large_batch)} succeeded")
        except Exception as e:
            print(f"❌ Large batch failed (expected): {e}")
            print("This confirms we need batch processing for large document chunks.")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mistral_embeddings()
