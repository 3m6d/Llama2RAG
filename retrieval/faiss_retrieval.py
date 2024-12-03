import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index
index = faiss.read_index("faiss_index.idx")

# Load the stored document chunks
document_chunks = np.load("document_chunks.npy", allow_pickle=True)

# Load the same embedding model used for indexing
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def retrieve_documents(query, k=5):
    """
    Retrieve the top-k relevant document chunks for a given query.

    Parameters:
    - query (str): The user's query.
    - k (int): Number of top results to return.

    Returns:
    - list: List of top-k document chunks matching the query.
    """
    try:
        # Encode the query using the same embedding model
        query_embedding = embedding_model.encode([query])

        # Search the FAISS index for the top-k nearest neighbors
        distances, indices = index.search(query_embedding, k)

        # Retrieve the matched document chunks using the indices
        retrieved_chunks = [document_chunks[idx] for idx in indices[0]]

        # Debugging output (optional)
        for i, idx in enumerate(indices[0]):
            print(f"Index: {idx}, Distance: {distances[0][i]}, Chunk: {document_chunks[idx][:100]}...")

        return retrieved_chunks

    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return []

