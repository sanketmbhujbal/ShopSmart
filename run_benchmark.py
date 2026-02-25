import time
import requests

# Test suite with Ground Truth (The "expected_match" boolean)
TEST_QUERIES = [
    {"query": "Sony WH-1000XM5 The Best Wireless Noise Canceling Headphones, Silver", "expected_match": True},
    {"query": "Sony WH-CH720N Wireless Noise Canceling Headphones Black", "expected_match": True},
    {"query": "Sony WH-1000XM4 Wireless Noise Canceling - Black", "expected_match": False},
    {"query": "Apple AirPods Pro 2nd Generation", "expected_match": False}, 
    {"query": "Restored Sony WH-1000XM5 Wireless Headphones (Refurbished)", "expected_match": False}, 
    {"query": "Sony WH-1000XM5 Silicone Case Cover Only", "expected_match": False}, 
]

def calculate_p95(data):
    if not data: return 0
    sorted_data = sorted(data)
    index = int(0.95 * len(sorted_data))
    return sorted_data[min(index, len(sorted_data) - 1)]

def run_evaluations():
    print("🚀 Starting Production Evaluation Suite...\n")
    
    latencies = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected_match"]
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:8000/resolve",
                json={"product_title": query},
                timeout=15
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # 200 means the AI found a match. 404 means it rejected the candidates.
            actual_match = response.status_code == 200
            
            # Grade the Prediction
            if actual_match and expected:
                true_positives += 1
                status = "✅ TP (Correct Match)"
            elif not actual_match and not expected:
                true_negatives += 1
                status = "✅ TN (Correct Reject)"
            elif actual_match and not expected:
                false_positives += 1
                status = "❌ FP (Hallucination!)"
            else:
                false_negatives += 1
                status = "❌ FN (Missed Match)"
                
            print(f"{status} | Latency: {latency:.2f}s | Query: {query[:40]}...")
            
        except Exception as e:
            print(f"⚠️ ERROR  | Query: {query[:40]}... | {e}")

    # Calculate Production Metrics
    total_queries = len(TEST_QUERIES)
    accuracy = ((true_positives + true_negatives) / total_queries) * 100
    
    actual_negatives = false_positives + true_negatives
    fpr = (false_positives / actual_negatives * 100) if actual_negatives > 0 else 0
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = calculate_p95(latencies)
    
    print("\n" + "="*50)
    print("📊 PRODUCTION EVALUATION REPORT")
    print("="*50)
    print(f"Total Queries:       {total_queries}")
    print(f"System Accuracy:     {accuracy:.1f}%")
    print(f"False Positive Rate: {fpr:.1f}%  <-- (Target: 0.0%)")
    print(f"Average Latency:     {avg_latency:.2f}s")
    print(f"p95 Latency (SLA):   {p95_latency:.2f}s")
    print("="*50)

if __name__ == "__main__":
    run_evaluations()