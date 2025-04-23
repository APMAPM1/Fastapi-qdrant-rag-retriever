import os
import logging
import nltk
from typing import List, Dict, Any

# Use dotenv to load environment variables from .env file for local development
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Qdrant Cloud Configuration (Read from Environment Variables)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Default collection name, can be overridden by environment variable
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_cloud_documents")

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Global Variables ---
model: SentenceTransformer | None = None
qdrant_client: QdrantClient | None = None
vector_size: int | None = None

# --- Dummy Data (Same as before) ---
DUMMY_DOCUMENTS = [
    "Solar power harnesses the energy of the sun using photovoltaic (PV) panels to generate electricity. These panels contain semiconductor materials that convert sunlight directly into DC power.",
    "Wind energy is captured using wind turbines. The wind turns the large blades of the turbine, which spins a generator to produce electricity. Wind farms can be located onshore or offshore.",
    "Hydroelectric power, or hydropower, is generated from the energy of moving water. Dams are often built to control water flow and create reservoirs, driving turbines connected to generators as water is released.",
    "Geothermal energy taps into the heat from within the Earth. Wells are drilled to access underground reservoirs of steam and hot water, which drive turbines linked to electricity generators.",
    "Renewable energy sources are crucial for combating climate change as they produce little to no greenhouse gas emissions during operation. Transitioning to renewables improves energy security and can create economic opportunities.",
    "Photovoltaic cells work based on the photoelectric effect, where light energy dislodges electrons in a material, creating an electric current. Silicon is the most common material used in solar panels.",
    "Modern wind turbines are highly efficient machines, capable of generating significant amounts of power. Their height and blade design are optimized to capture the maximum amount of wind energy available at a site."
]

# --- Helper Functions (Mostly same, adapted logging/error handling slightly) ---

def chunk_text_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    try:
        # Ensure punkt is downloaded (moved check to startup)
        return nltk.sent_tokenize(text)
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}")
        # Fallback
        return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

def setup_qdrant_and_index(client: QdrantClient, model_instance: SentenceTransformer, docs: List[str]):
    """Initializes Qdrant collection and indexes the documents."""
    global vector_size
    vector_size = model_instance.get_sentence_embedding_dimension()
    logging.info(f"Embedding model vector size: {vector_size}")

    try:
        # Check if collection exists
        collection_exists = False
        try:
            client.get_collection(collection_name=COLLECTION_NAME)
            collection_exists = True
            logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
            # Decide if you want to recreate or just use existing
            # For this example, we'll recreate it to ensure clean state on startup
            logging.info(f"Recreating collection '{COLLECTION_NAME}'.")
            client.delete_collection(collection_name=COLLECTION_NAME)
            collection_exists = False # Mark as non-existent after deletion
        except Exception as e:
             # A specific check for "Not Found" or similar error is safer
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower() or '404' in str(e):
                 logging.info(f"Collection '{COLLECTION_NAME}' not found. Creating.")
            else:
                 # Log unexpected error during check, but proceed to create attempt
                 logging.warning(f"Unexpected error checking collection: {e}. Attempting creation.")


        if not collection_exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                # Add other configurations like HNSW params if needed for performance
            )
            logging.info(f"Collection '{COLLECTION_NAME}' created successfully.")

        # --- Indexing Steps (Same as before) ---
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_text_into_sentences(doc))
        logging.info(f"Generated {len(all_chunks)} chunks for indexing.")
        if not all_chunks:
            logging.warning("No chunks generated. Indexing skipped.")
            return

        logging.info("Generating embeddings...")
        embeddings = model_instance.encode(all_chunks, show_progress_bar=True)
        logging.info("Embeddings generated.")

        points_to_upsert = [
            PointStruct(id=i, vector=vector.tolist(), payload={"text": chunk})
            for i, (chunk, vector) in enumerate(zip(all_chunks, embeddings))
        ]

        logging.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{COLLECTION_NAME}'...")
        # Consider using batching for very large datasets (client.upsert handles batching internally to some extent)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True
        )
        logging.info("Upsert completed.")

    except Exception as e:
        logging.error(f"Error during Qdrant setup or indexing: {e}", exc_info=True)
        # Raising HTTPException might prevent server startup, consider if that's desired
        # For now, log error and potentially let app start in degraded state
        # raise HTTPException(status_code=500, detail="Failed to initialize or index data in Qdrant.")
        logging.critical("CRITICAL: Failed Qdrant setup/indexing. API might not function correctly.")


# --- FastAPI Application ---
app = FastAPI(
    title="Simple RAG Retrieval API (Qdrant Cloud)",
    description="API to perform similarity search on documents stored in Qdrant Cloud.",
    version="0.1.1"
)

# --- Pydantic Models (Same as before) ---
class SearchResult(BaseModel):
    id: int | str
    score: float
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup."""
    global model, qdrant_client
    logging.info("Application startup: Initializing resources...")

    # 1. Validate Environment Variables
    if not QDRANT_URL:
        logging.error("FATAL: QDRANT_URL environment variable is not set.")
        raise RuntimeError("QDRANT_URL environment variable is required.")
    if not QDRANT_API_KEY:
        # Note: API key might be optional for some local/unsecured Qdrant setups,
        # but is typically required for Cloud.
        logging.warning("QDRANT_API_KEY environment variable is not set. Required for authenticated Qdrant Cloud.")
        # Depending on your cloud setup, you might want to raise RuntimeError here too:
        # raise RuntimeError("QDRANT_API_KEY environment variable is required for Qdrant Cloud.")


        # 2. Download NLTK data (if needed)
    try:
        # Check if 'punkt' is already available
        nltk.data.find('tokenizers/punkt')
        logging.info("NLTK 'punkt' tokenizer found.")
    except LookupError: # <-- Corrected line: Only catch LookupError here
        logging.info("NLTK 'punkt' tokenizer not found. Attempting download...")
        try:
            nltk.download('punkt')
            logging.info("'punkt' downloaded successfully.")
        except Exception as download_exc:
             # Catch any exception during the download process itself
             logging.error(f"Failed to download NLTK 'punkt': {download_exc}. Sentence tokenization might be impaired.")
             
    # 3. Initialize Sentence Transformer Model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded.")
    except Exception as e:
        logging.error(f"Failed to load Sentence Transformer model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load embedding model {EMBEDDING_MODEL_NAME}") from e

    # 4. Initialize Qdrant Client for Cloud
    logging.info(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,  # API key is now passed here
            # Add timeouts if needed, e.g., timeout=60
        )
        # Quick check to confirm connection
        qdrant_client.get_collections() # Raises exception if connection fails
        logging.info("Qdrant client initialized and connected to Cloud.")
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant Cloud: {e}", exc_info=True)
        raise RuntimeError("Could not connect to Qdrant Cloud database.") from e

    # 5. Setup Collection and Index Data (only if client and model loaded successfully)
    if qdrant_client and model:
        logging.info("Setting up Qdrant collection and indexing data...")
        setup_qdrant_and_index(qdrant_client, model, DUMMY_DOCUMENTS)
    else:
        logging.error("Qdrant client or embedding model not available. Skipping indexing.")
        # Consider raising error if indexing is critical for app function

    logging.info("Application startup complete.")


# --- API Endpoints (Same as before) ---
@app.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="The search query string"),
    limit: int = Query(3, ge=1, le=20, description="Number of results to return")
):
    """Performs similarity search in the Qdrant collection based on the query."""
    global model, qdrant_client
    if not model or not qdrant_client:
        raise HTTPException(status_code=503, detail="Service not ready. Model or DB client not initialized.")
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")

    logging.info(f"Received search query: '{q}', limit: {limit}")
    try:
        query_vector = model.encode(q).tolist()
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
        )
        logging.info(f"Found {len(search_result)} results from Qdrant.")
        results_list = [
            SearchResult(id=hit.id, score=hit.score, text=hit.payload.get("text", "N/A"))
            for hit in search_result
        ]
        return SearchResponse(results=results_list)
    except Exception as e:
        logging.error(f"Error during search for query '{q}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during the search.") # Avoid leaking detailed errors

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    if not model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    if not qdrant_client:
         raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    try:
        # Check Qdrant connection by trying a light operation
        qdrant_client.get_collections()
        return {"status": "ok", "message": "Service is running and connected to Qdrant"}
    except Exception as e:
        logging.error(f"Health check failed: Qdrant connection issue: {e}")
        raise HTTPException(status_code=503, detail="Qdrant connection error")


# --- Main Execution ---
if __name__ == "__main__":
    # Use uvicorn to run the app
    import uvicorn
    logging.info("Starting Uvicorn server...")
    # Use port 8000 by default, can be overridden by PORT env var if needed
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) # Use reload for development