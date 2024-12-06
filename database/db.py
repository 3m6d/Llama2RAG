import sqlite3
import numpy as np

### abrstrction 

# Connect to the SQLite database (create it if it doesn't exist)
def connect_db(db_path=r"C:\sqlite3\stock_data.db"):
    return sqlite3.connect(db_path)
 
def convert_embedding_to_blob(embedding):
    # Convert NumPy array (embedding) to binary data (BLOB)
    return embedding.tobytes()

# Insert the text chunks into the documents table and corresponding embeddings into embeddings table
def insert_chunks_and_embeddings(filename, chunks):
    conn = connect_db()
    try:
        cursor = conn.cursor()

        for index, chunk in enumerate(chunks):
            # Insert the chunk into the documents table first without embedding_id
            cursor.execute(
                "INSERT INTO documents (filename, chunk, chunk_index) VALUES (?, ?, ?);", 
                (filename, chunk, index)
            )

            # Get the ID of the inserted document row
            document_id = cursor.lastrowid

            # Generate an embedding for the chunk (simulated here as random for example)
            embedding = np.random.rand(512).astype(np.float32)  # Example embedding

            # Insert the embedding into the embeddings table and link it to the document
            cursor.execute(
                "INSERT INTO embeddings (document_id, embedding) VALUES (?, ?);", 
                (document_id, convert_embedding_to_blob(embedding))
            )

            # Get the ID of the newly inserted embedding
            embedding_id = cursor.lastrowid

            # Update the documents table with the embedding ID for this chunk
            cursor.execute(
                "UPDATE documents SET embedding_id = ? WHERE id = ?;", 
                (embedding_id, document_id)
            )

        # Commit the transaction
        conn.commit()

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()  # Rollback any changes if an error occurs

    finally:
        conn.close()

# Retrieve all chunks from the database
def retrieve_chunks():
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        return rows

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []

    finally:
        conn.close()
