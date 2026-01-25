import os

class Settings:
    PROJECT_NAME = "ShopSmart AI"
    VERSION = "1.0.0"
    
    # Infrastructure
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    CACHE_EXPIRE_SECONDS = int(os.getenv("CACHE_EXPIRE_SECONDS", 3600))
    
    # Models
    DENSE_MODEL_ID = "all-MiniLM-L6-v2"
    SPARSE_MODEL_ID = "Qdrant/bm25"
    
    # Switching from MiniLM (6 layers) to TinyBERT (2 layers)
    RERANKER_ID = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    
    # Paths
    DATA_PATH = os.path.join("data", "amazon.csv")
    COLLECTION_NAME = "shopsmart_products"

settings = Settings()
