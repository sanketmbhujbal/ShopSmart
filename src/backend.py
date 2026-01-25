import time
import json
import redis
import requests
import logging
import numpy as np
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from src.config import settings

# --- 1. Setup Structured Logging ---
logging.basicConfig(
    filename='search_logs.jsonl',
    level=logging.INFO,
    format='%(message)s'
)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# --- Database & Models ---
try:
    r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)
except:
    r = None

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
dense_model = SentenceTransformer(settings.DENSE_MODEL_ID)
sparse_model = HashingVectorizer(n_features=30000, norm='l2', alternate_sign=False)
reranker = CrossEncoder(settings.RERANKER_ID)

# --- Helper Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_sparse_vector(text):
    matrix = sparse_model.transform([text])
    matrix.sort_indices()
    return {"indices": matrix.indices.tolist(), "values": matrix.data.tolist()}

def search_via_rest(vector_name, vector_data, top_k):
    url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/collections/{settings.COLLECTION_NAME}/points/search"
    payload = {"vector": {"name": vector_name, "vector": vector_data}, "limit": top_k, "with_payload": True}
    try:
        response = requests.post(url, json=payload, timeout=0.5)
        response.raise_for_status()
        result = response.json()
        hits = []
        for item in result.get("result", []):
            hits.append(qmodels.ScoredPoint(
                id=item['id'], version=item['version'], score=item['score'], 
                payload=item['payload'], vector=None
            ))
        return hits
    except Exception:
        return []

def log_search_event(query, latency, num_results, source):
    """Saves the search event to a local file for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "latency_ms": latency,
        "results_count": num_results,
        "source": source
    }
    logging.info(json.dumps(log_entry))

# --- API ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search")
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    # 1. Cache Check
    cache_key = f"search:{request.query.lower().strip()}:{request.top_k}:v5"
    if r:
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            # Log the cache hit silently
            background_tasks.add_task(log_search_event, request.query, 0, len(data['results']), "Cache")
            return data

    try:
        # 2. Vectorization
        dense_vec = dense_model.encode(request.query).tolist()
        sparse_vec = get_sparse_vector(request.query)
        
        # 3. Retrieval
        fetch_k = 20
        dense_hits = search_via_rest("dense", dense_vec, fetch_k)
        sparse_hits = search_via_rest("sparse", sparse_vec, fetch_k)
        
        seen_ids = set()
        candidates = []
        for hit in dense_hits + sparse_hits:
            if hit.id not in seen_ids:
                seen_ids.add(hit.id)
                candidates.append(hit)
        
        if not candidates:
            # Return empty structure WITH latency/source keys to prevent crashes
            return {"results": [], "latency_ms": 0, "source": "Empty"}

        # 4. Rerank
        rerank_pool = candidates[:12]
        cross_inputs = [[request.query, hit.payload['title']] for hit in rerank_pool]
        logits = reranker.predict(cross_inputs)
        
        for idx, hit in enumerate(rerank_pool):
            hit.score = float(sigmoid(logits[idx]))

        ranked_results = sorted(rerank_pool, key=lambda x: x.score, reverse=True)[:request.top_k]
        
        results = [{
            "id": h.payload.get('product_id'),
            "title": h.payload.get('title'),
            "price": h.payload.get('price'),
            "rating": h.payload.get('rating'),
            "image": h.payload.get('img_link'),
            "category": h.payload.get('category'),
            "relevance_score": round(h.score, 3)
        } for h in ranked_results]
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        # Log event
        background_tasks.add_task(log_search_event, request.query, latency, len(results), "Hybrid+TinyBERT")
        
        response = {
            "results": results,
            "latency_ms": latency,
            "source": "Hybrid+TinyBERT"
        }
        
        if r:
            r.setex(cache_key, settings.CACHE_EXPIRE_SECONDS, json.dumps(response))
            
        return response

    except Exception as e:
        print(f"Error: {e}")
        # Return error structure WITH keys to prevent crashes
        return {"results": [], "latency_ms": 0, "source": "Error"}