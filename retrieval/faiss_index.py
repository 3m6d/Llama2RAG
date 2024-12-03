import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from database import db
import numpy as np


"""
This script generates a FAISS index and saves it to disk. 
It also retrieves the text chunks from the database and
generates embeddings for them.
"""

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"""
Sentence transformers are built on transformer architectures, 
which excel in capturing contextual relationships in sequential data
"""

# Retrieve the text chunks from the database
rows = db.retrieve_chunks()  
document_chunks = [row[2] for row in rows]  # The chunk text is the 3rd column

# Generate embeddings for the chunks
if document_chunks:
    chunk_embeddings = model.encode(document_chunks, show_progress_bar=True)
    print("Sample Embedding for Chunk 0:", chunk_embeddings[0][:5])  # Print first 5 values
    print("Sample Embedding for Chunk 1:", chunk_embeddings[1][:5])
    

    variances = np.var(chunk_embeddings, axis=0)
    print("Average Variance Across Embedding Dimensions:", np.mean(variances))

    print("Chunk embeddings shape:", chunk_embeddings.shape)

else:
    print("No document chunks found in the database.")
    chunk_embeddings = np.array([])  # Assign an empty array

# Create a FAISS index if embeddings exist
embedding_dimension = chunk_embeddings.shape[1] if chunk_embeddings.size > 0 else 0
if embedding_dimension > 0:
    # Create a FAISS index with IndexFlatL2
    index = faiss.IndexFlatL2(embedding_dimension)
    print(f"FAISS index created with dimension: {embedding_dimension}")
    
    # Add embeddings to the FAISS index
    index.add(chunk_embeddings)
    print(f"Number of embeddings added to the FAISS index: {index.ntotal}")  # Print the total number of embeddings

    # Save the FAISS index
    faiss.write_index(index, "faiss_index.idx")
    print("FAISS index created and saved as faiss_index.idx")

    # Save the document chunks for later retrieval
    np.save("document_chunks.npy", document_chunks)
    print("Document chunks saved as document_chunks.npy")
else:
    print("Failed to create FAISS index due to empty embeddings.")


# retrieval/faiss_index.py

def query_faiss_index(query, index, embedding_model, document_chunks, k=5, threshold=1.0):
    try:
        # Encode the user query
        query_embedding = embedding_model.encode([query])

        # Search the FAISS index
        distances, indices = index.search(query_embedding, k=k)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        # Retrieve chunks within the threshold
        results = []
        for i, idx in enumerate(indices[0]):
            distance = distances[0][i]
            if distance <= threshold:
                results.append(document_chunks[idx])
                print(f"Index: {idx}, Distance: {distance}, Chunk: {document_chunks[idx][:100]}... (Included)")
            else:
                print(f"Index: {idx}, Distance: {distance} exceeds threshold {threshold}. (Skipped)")
        return results

    except Exception as e:
        print(f"An error occurred during querying: {e}")
        return []


    
print(f"Number of unique document chunks: {len(set(document_chunks))}")


"""if __name__ == "__main__":
    query = "what are the study about in this stock market?"
    results = query_faiss_index(query, k=5)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}\n")"""

