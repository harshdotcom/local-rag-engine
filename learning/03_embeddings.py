# 03_embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the embedding model (downloads once, then cached)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # Small, fast, 384 dimensions - perfect for learning
)

# Test it
sentence1 = "The cat sat on the mat"
sentence2 = "A feline rested on a rug"
sentence3 = "Stock market crashed today"

vec1 = embedding_model.embed_query(sentence1)
vec2 = embedding_model.embed_query(sentence2)
vec3 = embedding_model.embed_query(sentence3)

print(f"Vector size: {len(vec1)} dimensions")
print(f"First 5 values of vec1: {vec1[:5]}")

# Manual cosine similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nSimilarity (cat vs feline): {cosine_similarity(vec1, vec2):.4f}")  # ~0.85
print(f"Similarity (cat vs stocks): {cosine_similarity(vec1, vec3):.4f}")  # ~0.10