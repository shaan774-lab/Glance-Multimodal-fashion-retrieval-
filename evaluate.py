import json
import os
import sys
from retriever import FashionRetriever

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Starting automated query evaluation...")
    
    retriever = FashionRetriever()
    
    # 5 evaluation queries from the assignment
    eval_queries = [
        {"id": 0, "type": "Attribute Specific", "query": "A person in a bright yellow raincoat.", "target_id": 0},
        {"id": 1, "type": "Contextual/Place", "query": "Professional business attire inside a modern office.", "target_id": 1},
        {"id": 2, "type": "Complex Semantic", "query": "Someone wearing a blue shirt sitting on a park bench.", "target_id": 2},
        {"id": 3, "type": "Style Inference", "query": "Casual weekend outfit for a city walk.", "target_id": 3},
        {"id": 4, "type": "Compositional", "query": "A red tie and a white shirt in a formal setting.", "target_id": 4}
    ]
    
    results_summary = []
    
    print("\nEvaluating queries:")
    print("=" * 60)
    
    for q_item in eval_queries:
        query_text = q_item["query"]
        target_id = q_item["target_id"]
        q_type = q_item["type"]
        
        print(f"\nQuery Type: {q_type}")
        print(f"Query: \"{query_text}\"")
        
        # Run search
        results = retriever.search(query_text, top_k=5)
        
        # Find target rank
        target_rank = -1
        for rank, res in enumerate(results):
            if res["id"] == target_id:
                target_rank = rank + 1
                break
                
        top_res = results[0] if results else None
        success = (target_rank == 1)
        
        print(f"Top 1 Retrieved: ID={top_res['id']} (Score: {top_res['scores']['final']:.4f})")
        print(f"  Caption: {top_res['caption']}")
        print(f"  Category: {top_res['category']} | Color: {top_res['color']} | Env: {top_res['environment']}")
        print(f"  Target Rank: {target_rank} (Success: {success})")
        print("-" * 50)
        
        # Store for JSON
        res_list = []
        for rank, res in enumerate(results[:3]):
            res_list.append({
                "rank": rank + 1,
                "id": res["id"],
                "image_path": res["image_path"],
                "caption": res["caption"],
                "category": res["category"],
                "color": res["color"],
                "environment": res["environment"],
                "clothing_type": res["clothing_type"],
                "visual_score": res["scores"]["visual"],
                "textual_score": res["scores"]["textual"],
                "attribute_score": res["scores"]["attribute"],
                "final_score": res["scores"]["final"]
            })
            
        results_summary.append({
            "query_id": q_item["id"],
            "query_type": q_type,
            "query": query_text,
            "target_id": target_id,
            "target_rank": target_rank,
            "success": success,
            "results": res_list
        })
        
    # Write to JSON
    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    # Print final summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Query Type':<20} | {'Top 1 ID':<8} | {'Top 1 Score':<12} | {'Success':<8}")
    print("-" * 60)
    for r in results_summary:
        top1_id = r["results"][0]["id"]
        top1_score = r["results"][0]["final_score"]
        success_str = "PASS" if r["success"] else "FAIL"
        print(f"{r['query_type']:<20} | {top1_id:<8} | {top1_score:<12.4f} | {success_str:<8}")
    print("=" * 60)
    print("Evaluation results saved to data/eval_results.json")

if __name__ == "__main__":
    main()
