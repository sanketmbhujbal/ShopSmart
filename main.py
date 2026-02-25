import time
import os
import json
import traceback
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from openai import OpenAI
from dotenv import load_dotenv
import redis

load_dotenv()

# --- CONFIGURATION ---
COLLECTION_NAME = "walmart_products_hybrid"
DENSE_MODEL_ID = "TaylorAI/bge-micro-v2"

# --- REDIS INITIALIZATION ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Test the connection on startup
try:
    redis_client.ping()
    print("🟢 Redis Cache Connected Successfully!")
except Exception as e:
    # Catching broad Exceptions here because local port 6379 might be 
    # occupied by non-Redis HTTP services, throwing InvalidResponse!
    print(f"🟡 Redis connection failed: {e}")
    print("🟡 Running without cache.")
    redis_client = None

# --- APP INITIALIZATION ---
app = FastAPI(title="Neutral Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Loading Vector Retrieval Engine...")

# Initialize Vector DB & Embedders
qdrant = QdrantClient("localhost", port=6333)

# Dense Embedder (For Meaning)
dense_embedder = SentenceTransformer(DENSE_MODEL_ID)

# Sparse Embedder (For Exact Keywords)
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

llm_client = OpenAI()
print("✅ Systems Online!")

# --- DATA MODELS ---
class ResolutionRequest(BaseModel):
    product_title: str

class ProductMatch(BaseModel):
    product_title: str
    match_score: float
    retailer: str
    latency_ms: float
    price: str  
    url: str    

# --- ASYNC TELEMETRY WORKER (OBSERVABILITY) ---
def log_trace(query: str, search_results: list, decision: dict, final_latency: float, error_state: str = None):
    print("\n📝 [Background] Saving pipeline trace...")
    try:
        trace = {
            "timestamp": time.time(),
            "query": query,
            "retrieval": {
                "top_candidates": [
                    {
                        "title": hit.payload.get('name', hit.payload.get('original_string', 'Unknown')),
                        "score": getattr(hit, 'score', 0.0)
                    } for hit in (search_results or [])
                ]
            },
            "llm_judge": {
                "decision": decision.get("match_found") if decision else False,
                "reasoning": decision.get("reasoning") if decision else "No reasoning provided.",
                "candidate_id": decision.get("candidate_id") if decision else None
            },
            "outcome": {
                "latency_ms": final_latency,
                "status": "success" if decision and decision.get("match_found") else "not_found",
                "error": error_state
            }
        }
        
        # Append to our local JSONL file
        with open("pipeline_traces.jsonl", "a") as f:
            f.write(json.dumps(trace) + "\n")
            
        print("✅ [Background] Trace saved to pipeline_traces.jsonl\n")
    except Exception as e:
        print(f"❌ [Background] Failed to save trace: {e}")

# --- API ENDPOINT ---
@app.post("/resolve", response_model=ProductMatch)
def resolve_entity(request: ResolutionRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    query = request.product_title
    normalized_query = " ".join(query.lower().split())
    
    # 0. CHECK REDIS CACHE (Production Cost Optimization)
    if redis_client:
        cached_result = redis_client.get(normalized_query)
        if cached_result:
            print(f"⚡ REDIS CACHE HIT: 0 API tokens used for '{query}'")
            return ProductMatch(**json.loads(cached_result))

    print(f"\n" + "="*50)
    print(f"🔍 INCOMING QUERY: \n'{query}'")
    
    # 1. RETRIEVAL (Hybrid Search)
    query_dense = dense_embedder.encode(query).tolist()
    query_sparse = list(sparse_embedder.embed([query]))[0]
    
    qdrant_sparse = models.SparseVector(
        indices=query_sparse.indices.tolist(),
        values=query_sparse.values.tolist()
    )

    try:
        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=10),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit=10)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5
        ).points
    except Exception as e:
        print("\n🛑 FATAL QDRANT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    if not search_results:
        raise HTTPException(status_code=404, detail="No products found in vector search.")
    
    candidates_text = ""
    candidate_map = {}
    for i, hit in enumerate(search_results):
        # Dynamically hunt for the title key based on Walmart's actual JSON structure
        title = hit.payload.get('name', hit.payload.get('title', hit.payload.get('original_string', 'Unknown Title')))
        candidates_text += f"[{i}] {title}\n"
        hit.payload['extracted_title'] = title 
        candidate_map[str(i)] = hit.payload

    # 2. FAST LLM RESOLUTION (Precision)
    fast_prompt = f"""
    You are a strictly impartial e-commerce Entity Resolution engine. 
    Compare the QUERY PRODUCT to the CANDIDATE PRODUCTS and determine if there is an EXACT SKU match.
    
    CRITICAL RULES:
    1. BRAND MISMATCH = Reject (e.g., Apple vs Sony).
    2. CONDITION MISMATCH = Reject. A query for a "New" item MUST NOT match a "Refurbished", "Renewed", or "Restored" candidate.
    3. DEVICE VS ACCESSORY = Reject. A case, skin, or cover is NEVER a match for the main device.
    4. STRICT GENERATION MATCH = Reject if the generation numbers differ (XM4 != XM5).
    5. MISSING MINOR WORDS = Accept only if the core model matches perfectly AND it is not an accessory or refurbished.

    --- EXAMPLES ---
    Input Query: "Sony WH-1000XM4 Wireless Noise Canceling - Black"
    Input Candidates: "[0] Restored Sony WH-1000XM4 Wireless Headphones (Refurbished)"
    Output:
    {{
        "reasoning": "Condition mismatch. The query implies a New item, but the candidate is Refurbished/Restored.",
        "match_found": false,
        "candidate_id": null
    }}

    Input Query: "Sony WH-1000XM5 Silicone Case Cover Only"
    Input Candidates: "[0] Sony WH-1000XM5 Wireless Headphones"
    Output:
    {{
        "reasoning": "The query is for an accessory (silicone case), but the candidate is the main device (headphones).",
        "match_found": false,
        "candidate_id": null
    }}

    --- YOUR TASK ---
    Input Query: "{query}"
    Input Candidates:
    {candidates_text}
    
    Output strictly in the JSON format shown above:
    """
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": fast_prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    total_latency = (time.time() - start_time) * 1000
    
    # Graceful Error Handling + Tracing
    try:
        decision = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print(f"🛑 CRITICAL: LLM failed to return JSON in {total_latency:.0f}ms")
        background_tasks.add_task(log_trace, query, search_results, {}, total_latency, "JSON Decode Error")
        raise HTTPException(status_code=404, detail="AI reasoning failure. Safely rejecting.")

    print(f"\n🧠 AI REASONING: {decision.get('reasoning')}")
    print(f"🎯 FINAL DECISION: {decision.get('match_found')}\n")
    
    if not decision.get("match_found") or decision.get("candidate_id") is None or str(decision.get("candidate_id")).lower() in ["none", "null", ""]:
        print(f"🛑 LLM REJECT in {total_latency:.0f}ms")
        background_tasks.add_task(log_trace, query, search_results, decision, total_latency, "LLM Rejected")
        raise HTTPException(status_code=404, detail="LLM rejected all candidates.")

    # 3. WE HAVE A VALID MATCH
    try:
        matched_idx = str(decision["candidate_id"])
        matched_product = candidate_map[matched_idx]
    except (KeyError, ValueError, TypeError):
        print(f"🛑 CRITICAL: LLM returned invalid ID '{decision.get('candidate_id')}'")
        background_tasks.add_task(log_trace, query, search_results, decision, total_latency, "Invalid Candidate ID")
        raise HTTPException(status_code=500, detail="LLM returned an unparseable candidate ID.")

    matched_title = matched_product.get('extracted_title', 'Unknown Title')
    print(f"⚡ FAST ACCEPT in {total_latency:.0f}ms: Matched '{matched_title}'")
    print("="*50)
    
    # Extract price safely from Walmart's nested structure
    price_val = matched_product.get('price')
    if not price_val and 'priceInfo' in matched_product:
        price_val = matched_product['priceInfo'].get('currentPrice', {}).get('priceString', 'N/A')
        
    # Extract URL safely and format canonical paths
    url_val = matched_product.get('url', matched_product.get('canonicalUrl', '#'))
    if url_val.startswith('/'):
        url_val = "https://www.walmart.com" + url_val
    
    # 4. Format the final response
    final_response = {
        "product_title": matched_title,
        "match_score": 1.0, 
        "retailer": matched_product.get('retailer', 'Walmart'),
        "latency_ms": total_latency,
        "price": str(price_val),
        "url": url_val
    }
    
    # Save to Redis with a 24-hour expiration (86400 seconds)
    if redis_client:
        redis_client.setex(
            normalized_query, 
            86400, 
            json.dumps(final_response)
        )
    
    # Log the successful trace!
    background_tasks.add_task(log_trace, query, search_results, decision, total_latency, None)
    
    return ProductMatch(**final_response)