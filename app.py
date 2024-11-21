from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from database import db  # Assuming you have this import 
from models.llama_model import generate_response  # Assuming you have this function in llama_model.py
from retrieval.faiss_index import query_faiss_index  # Import the existing function

# Initialize Flask app
app = Flask(__name__)

# Global variables for the index, model, and chunks
index = None
embedding_model = None
document_chunks = []

<<<<<<< HEAD



=======
>>>>>>> 28bcd8ae440328b0690a4c2be9311d4d3004eae0
def initialize_app():
    global index, embedding_model, document_chunks   
    try:
        # Load the FAISS index and embedding model once during initialization
        index = faiss.read_index("faiss_index.idx")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Connect to the database
        db.connect_db()

        # Retrieve document chunks from the database
        rows = db.retrieve_chunks()
        if not rows:
            raise ValueError("No document chunks found in the database.")
        document_chunks = [row[2] for row in rows]  #  'chunk' is the 3rd column in retrieve_chunks() result

    except Exception as e:
        print(f"Error initializing application: {e}")
        index = None
        embedding_model = None
        document_chunks = []
@app.route('/query', methods=['POST'])
def query():
    # Check if everything was initialized properly
    if not index or not embedding_model or not document_chunks:
        return jsonify({"error": "The server is not properly initialized."}), 500

    # Validate incoming JSON
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Invalid query provided. Please provide a valid 'query' field."}), 400

    user_query = data["query"].strip()
    if not user_query:
        return jsonify({"error": "Empty query provided."}), 400

    try:
        # Use the imported FAISS query function to find relevant chunks
        matched_chunks = query_faiss_index(user_query, index, embedding_model, document_chunks)
        if not matched_chunks:
            combined_text = ""
        else:
            combined_text = "\n".join(matched_chunks)  
            # Print matched chunks for debugging (optional)
            print(f"Matched chunks for query '{user_query}': {matched_chunks}")

        # Generate a response using the LLaMA model
        response = generate_response(user_query, combined_text)

        # Return the response in JSON format
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

# Call the initialization function
initialize_app()

if __name__ == '__main__':
    # Consider removing debug=True in production for security
    app.run(debug=True)

