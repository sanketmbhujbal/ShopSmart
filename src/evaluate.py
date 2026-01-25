import requests
import numpy as np

# --- 1. The "Golden Set" ---
# A list of queries and the EXACT keywords that must appear in a good result
# This simulates User Intent
TEST_SUITE = [
    {
        "query": "braided usb c cable fast charging",
        "required_terms": ["braided", "usb"],
        "one_of": ["cable", "cord"],  # Must have one of these
        "min_price": 0  
    },
    {
        "query": "wireless ergonomic mouse",
        "required_terms": ["wireless", "mouse"],
        "one_of": ["ergonomic", "comfort", "vertical"],
    },
    {
        "query": "steam iron auto shut off",
        "required_terms": ["iron", "steam"],
        "one_of": ["shut", "safety", "auto"],
    },
    {
        "query": "4k monitor",
        "required_terms": ["4k"],
        "one_of": ["monitor", "display", "tv", "screen"], 
    }
]

def calculate_ndcg(relevance_scores, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at K.
    Standard industry metric for ranking quality.
    """
    # DCG: Discounted Cumulative Gain
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k])])
    
    # IDCG: Ideal DCG (Best possible ordering)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores[:k])])
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_system():
    print(f"🧪 Starting Evaluation on {len(TEST_SUITE)} test cases...\n")
    
    total_ndcg = []
    total_latency = []
    
    for case in TEST_SUITE:
        query = case['query']
        print(f"🔎 Testing: '{query}'")
        
        # 1. Call API
        try:
            response = requests.post(
                "http://127.0.0.1:8000/search", 
                json={"query": query, "top_k": 10}
            )
            data = response.json()
            results = data['results']
            total_latency.append(data['latency_ms'])
        except Exception as e:
            print(f"   ❌ API Failed: {e}")
            continue

        # 2. auto-grade the results 
        relevance_scores = []
        for item in results:
            title_lower = item['title'].lower()
            
            # Check Requirements
            has_required = all(term in title_lower for term in case['required_terms'])
            has_variant = any(term in title_lower for term in case['one_of'])
            
            # Scoring Logic:
            # 2 points = Perfect Match (Required + Variant)
            # 1 point = Partial Match (Required only)
            # 0 points = Irrelevant
            if has_required and has_variant:
                score = 2
            elif has_required:
                score = 1
            else:
                score = 0
                
            relevance_scores.append(score)

        # 3. Calculate NDCG
        score = calculate_ndcg(relevance_scores)
        total_ndcg.append(score)
        
        print(f"   Score: {score:.2f} | Latency: {data['latency_ms']}ms")
        print(f"   Top Result: {results[0]['title'][:50]}...")
        print("-" * 40)

    # 4. Final Report
    avg_ndcg = np.mean(total_ndcg)
    avg_lat = np.mean(total_latency)
    
    print("\n" + "="*30)
    print("📊 SYSTEM REPORT CARD")
    print("="*30)
    print(f"✅ Average NDCG Score: {avg_ndcg:.3f} / 1.0")
    print(f"⚡ Average Latency:    {avg_lat:.1f} ms")
    print("="*30)
    
    if avg_ndcg > 0.8:
        print("🏆 Result: EXCELLENT (Production Ready)")
    elif avg_ndcg > 0.6:
        print("👍 Result: GOOD (Solid Baseline)")
    else:
        print("⚠️ Result: NEEDS IMPROVEMENT")

if __name__ == "__main__":
    evaluate_system()