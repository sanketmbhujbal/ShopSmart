import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# 1. Initialize DB and Embedders
print("⏳ Loading embedding models... (This might take a second)")
qdrant = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "walmart_products_hybrid"

dense_embedder = SentenceTransformer('TaylorAI/bge-micro-v2')
sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

# 2. Collection for Hybrid Search
print(f"🗑️ Creating/Resetting collection: {COLLECTION_NAME}")
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384, # bge-micro-v2 size
            distance=models.Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    }
)

# 3. Load raw e-commerce data
with open('walmart_raw_dump.json', 'r') as file:
    raw_data = json.load(file)

# --- THE RECURSIVE HUNTER ---
# This function digs through deeply nested Next.js/React web dumps 
# to find actual product dictionaries.
def extract_products(node):
    found_products = []
    
    if isinstance(node, dict):
        # Walmart typically stores the product title under the key "name"
        # We also check for "price" or "imageInfo" to avoid grabbing random metadata
        if "name" in node and ("price" in node or "priceInfo" in node or "imageInfo" in node):
            found_products.append(node)
        elif "original_string" in node or "title" in node and "price" in node:
            found_products.append(node)
        
        # If this dict isn't a product, recursively check all its values
        for key, value in node.items():
            # Sometimes Walmart puts the main list under 'itemResults'
            if key == "itemResults" and isinstance(value, list):
                found_products.extend(value)
            else:
                found_products.extend(extract_products(value))
                
    elif isinstance(node, list):
        # Recursively check every item in the list
        for item in node:
            found_products.extend(extract_products(item))
            
    return found_products

products = extract_products(raw_data)

# Deduplicate products just in case the web dump listed them twice
unique_products = {p.get('name', p.get('title', str(i))): p for i, p in enumerate(products)}.values()
products = list(unique_products)

points = []
print(f"🔄 Found and embedding {len(products)} products...")

for i, product in enumerate(products):
    # Extract the title
    title = product.get('name', product.get('title', product.get('original_string', '')))
    
    # Skip empty entries
    if not title or not isinstance(title, str):
        continue

    # Generate both vectors
    dense_vec = dense_embedder.encode(title).tolist()
    sparse_vec = list(sparse_embedder.embed([title]))[0]
    
    # Package them into a Qdrant Point
    points.append(
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_vec,
                "sparse": models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            },
            payload=product # Save the properly formatted object
        )
    )

# 5. Push to Database
if not points:
    print("🛑 ERROR: Still couldn't find any products! Check your JSON file.")
else:
    print("🚀 Uploading to Qdrant...")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Hybrid Database Ingestion Complete!")