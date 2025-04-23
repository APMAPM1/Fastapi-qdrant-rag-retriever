# Simple RAG Retrieval API with FastAPI and Qdrant Cloud

This project implements a basic Retrieval API component, often used in Retrieval-Augmented Generation (RAG) systems. It uses FastAPI to create a web server, Sentence Transformers to generate text embeddings, and Qdrant Cloud as a vector database for efficient similarity search.

The API allows you to index text documents (currently chunked by sentence) and then perform semantic searches to find the most relevant document chunks based on a query.

## Features

*   **FastAPI Backend:** Provides a robust and fast web API framework.
*   **Semantic Search:** Uses `sentence-transformers` (`all-MiniLM-L6-v2` by default) to understand the meaning of text and find relevant results even if keywords don't match exactly.
*   **Qdrant Cloud Integration:** Leverages Qdrant Cloud for storing and searching vector embeddings efficiently.
*   **Sentence Chunking:** Uses NLTK to split documents into sentences for indexing.
*   **Environment Variable Configuration:** Easy configuration via a `.env` file for Qdrant credentials.
*   **Basic Health Check:** Includes a `/health` endpoint.
*   **Automatic Documentation:** FastAPI provides interactive API docs at `/docs`.

## Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Git
*   A [Qdrant Cloud](https://cloud.qdrant.io/) account (or a local Qdrant instance). You will need:
    *   Your Qdrant Cluster URL.
    *   A Qdrant API Key (if required by your cluster security settings).

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/APMAPM1/Fastapi-qdrant-rag-retriever.git
    cd Fastapi-qdrant-rag-retriever
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the application, NLTK might download the 'punkt' tokenizer data if it's not already present.*

4.  **Configure Environment Variables:**
    Create a file named `.env` in the project's root directory. **Do not commit this file to Git.** Add your Qdrant Cloud credentials:
    ```dotenv
    # .env file
    QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io:6333
    QDRANT_API_KEY=your_qdrant_api_key_here # Optional if your cluster has no auth

    # Optional overrides
    # COLLECTION_NAME=my_custom_collection
    # PORT=8080
    ```
    Replace the placeholder URL and API Key with your actual Qdrant Cloud details.

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --port 8000
