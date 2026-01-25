import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from src.config import settings

def clean_currency(x):
    """Converts '₹1,299' -> 1299.0"""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        clean_str = x.replace("₹", "").replace("$", "").replace(",", "").strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    return 0.0

def clean_rating(x):
    """Converts '4.2' -> 4.2"""
    try:
        return float(pd.to_numeric(x, errors='coerce'))
    except:
        return 0.0

def ingest_data():
    print(f"🚀 Starting Ingestion for {settings.PROJECT_NAME}...")
    
    # 1. Load Data
    if not os.path.exists(settings.DATA_PATH):
        print(f"❌ Error: File not found at {settings.DATA_PATH}")
        return

    df = pd.read_csv(settings.DATA_PATH)
    
    # 2. Data Cleaning
    print("🧹 Cleaning raw data...")
    df['product_name'] = df['product_name'].fillna("Unknown Product")
    df['category'] = df['category'].fillna("General")
    df['about_product'] = df['about_product'].fillna("")
    df['review_content'] = df['review_content'].fillna("")
    df['img_link'] = df['img_link'].fillna("")
    df['product_link'] = df['product_link'].fillna("")
    
    df['price'] = df['discounted_price'].apply(clean_currency)
    df['rating'] = df['rating'].apply(clean_rating).fillna(0.0)
    
    df['search_text'] = (
        df['product_name'] + ". " + 
        df['category'].str.replace("|", " ") + ". " + 
        df['about_product'] + ". " + 
        df['review_content']
    )
    
    print(f"✅ Data Cleaned. {len(df)} records ready.")

    # 3. Initialize Models
    print("🧠 Loading AI Models...")
    # Dense: Standard Bi-Encoder
    dense_model = SentenceTransformer(settings.DENSE_MODEL_ID)
    
    # Sparse: Scikit-Learn HashingVectorizer (Stateless & Robust)
    # n_features=30000 ensures collisions are rare for this dataset size
    sparse_model = HashingVectorizer(n_features=30000, norm='l2', alternate_sign=False)
    
    # 4. Setup Vector DB
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    client.recreate_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=384, 
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    print(f"📦 Collection '{settings.COLLECTION_NAME}' created.")

    # 5. Batch Processing
    batch_size = 64
    total_batches = (len(df) // batch_size) + 1
    
    print(f"🏭 Ingesting in {total_batches} batches...")
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        texts = batch['search_text'].tolist()
        
        # Generate Dense Vectors
        dense_vectors = dense_model.encode(texts)
        
        points = []
        # Zip loop to process dense vectors and generate sparse vectors on the fly
        for idx, (d_vec, text, row) in enumerate(zip(dense_vectors, texts, batch.itertuples(index=False))):
            
            # Generate Sparse Vector (Stateless)
            sparse_matrix = sparse_model.transform([text])
            sparse_matrix.sort_indices()
            sparse_indices = sparse_matrix.indices.tolist()
            sparse_values = sparse_matrix.data.tolist()

            # Smart Category Badge
            cat_path = str(row.category)
            smart_badge = cat_path.split("|")[-1] if "|" in cat_path else cat_path

            payload = {
                "product_id": str(row.product_id),
                "title": str(row.product_name),
                "category": smart_badge,
                "price": float(row.price),
                "rating": float(row.rating),
                "description": str(row.about_product)[:500],
                "img_link": str(row.img_link),
                "product_link": str(row.product_link)
            }

            points.append(models.PointStruct(
                id=i + idx,
                vector={
                    "dense": d_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload=payload
            ))

        # Upload Batch
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        print(f"   🔹 Batch {i // batch_size + 1}/{total_batches} done.")

    print("🎉 Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()