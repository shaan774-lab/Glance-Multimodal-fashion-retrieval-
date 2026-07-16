import sys
import sqlite3
import numpy as np
import torch
import re
from transformers import CLIPProcessor, CLIPModel

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

class FashionRetriever:
    def __init__(self, db_path="data/fashion_index.db", model_name="openai/clip-vit-base-patch32"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _lazy_load_model(self):
        if self.model is None:
            print("Loading CLIP model for query embedding...", file=sys.stderr)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
    def parse_query(self, query_str):
        q = query_str.lower()
        
        # Color vocabulary
        colors = ["red", "blue", "yellow", "green", "white", "black", "brown", "beige", "gray", "pink", "orange", "purple"]
        
        # Category vocabulary
        categories = ["shirt", "t-shirt", "hoodie", "jeans", "jacket", "coat", "trench coat", "dress", "skirt", "pants", "tie", "sweater", "cardigan", "sandals", "shoes", "boots", "bag", "handbag", "raincoat", "suit", "blazer", "blouse", "socks"]
        
        # Split by conjunctions to bind colors to categories
        clauses = re.split(r'\band\b|,|\bwith\b|\bin\b|\bat\b|\bon\b', q)
        
        bindings = []
        for clause in clauses:
            clause_colors = [c for c in colors if c in clause]
            clause_cats = [cat for cat in categories if cat in clause]
            for c in clause_colors:
                for cat in clause_cats:
                    bindings.append((c, cat))
                    
        # General extraction
        gen_colors = [c for c in colors if c in q]
        gen_cats = [cat for cat in categories if cat in q]
        
        # Environment extraction
        gen_envs = []
        if "office" in q or "business" in q or "indoor" in q or "desk" in q:
            gen_envs.append("office")
        if "park" in q or "bench" in q or "lawn" in q or "trees" in q:
            gen_envs.append("park")
        if "street" in q or "city" in q or "walk" in q or "outdoor" in q or "road" in q:
            gen_envs.append("street")
        if "home" in q or "living" in q or "bedroom" in q or "house" in q or "couch" in q:
            gen_envs.append("home")
            
        # Vibe/Style extraction
        styles = []
        if "formal" in q or "business" in q or "suit" in q or "office" in q:
            styles.append("formal")
        if "casual" in q or "loungewear" in q or "weekend" in q:
            styles.append("casual")
        if "streetwear" in q or "hoodie" in q or "jacket" in q:
            styles.append("streetwear")
            
        return {
            "bindings": bindings,
            "colors": gen_colors,
            "categories": gen_cats,
            "environments": gen_envs,
            "styles": styles
        }
        
    def get_query_embedding(self, query_str):
        self._lazy_load_model()
        with torch.no_grad():
            inputs = self.processor(text=[query_str], return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
            
    def search(self, query_str, top_k=5, w_visual=0.5, w_textual=0.2, w_attr=0.3):
        parsed = self.parse_query(query_str)
        query_emb = self.get_query_embedding(query_str)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path, category, clothing_type, color, environment, caption, style_tags, visual_embedding, text_embedding FROM fashion_items")
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            idx, img_path, category, clothing_type, color, env, caption, style_tags_str, vis_blob, txt_blob = row
            style_tags = style_tags_str.split(",") if style_tags_str else []
            
            # Deserialize embeddings
            vis_emb = np.frombuffer(vis_blob, dtype=np.float32)
            txt_emb = np.frombuffer(txt_blob, dtype=np.float32)
            
            # Cosine similarities
            vis_score = float(np.dot(query_emb, vis_emb))
            txt_score = float(np.dot(query_emb, txt_emb))
            
            # Attribute Match Score
            attr_score = 0.0
            
            # 1. Environment Match
            if parsed["environments"]:
                if env in parsed["environments"]:
                    attr_score += 1.0
                else:
                    attr_score -= 0.5  # Soft penalty for environment mismatch
                    
            # 2. Category & Style Match
            if parsed["categories"]:
                if category in parsed["categories"]:
                    attr_score += 0.5
                else:
                    attr_score -= 0.2
                    
            if parsed["styles"]:
                # Check style similarity (formal/casual)
                style_matches = sum(1 for s in parsed["styles"] if s == clothing_type.lower() or s in style_tags)
                attr_score += 0.3 * style_matches
                
            # 3. Compositional Bindings Match (CRITICAL GUARD)
            if parsed["bindings"]:
                for bound_color, bound_cat in parsed["bindings"]:
                    # Check if this item is the target category
                    if category == bound_cat or bound_cat in category:
                        # Check if color binds correctly
                        if color == bound_color:
                            attr_score += 1.5  # Strong reward for correct binding
                        else:
                            attr_score -= 1.5  # Strong penalty for incorrect color binding!
            else:
                # Fallback to general colors if no specific bindings
                if parsed["colors"]:
                    if color in parsed["colors"]:
                        attr_score += 0.8
                    else:
                        attr_score -= 0.3
                        
            # Normalize attribute score between -1.0 and 1.0 (approximate scaling)
            attr_score = max(-1.0, min(attr_score / 3.0, 1.0))
            
            # Combine scores
            final_score = (w_visual * vis_score) + (w_textual * txt_score) + (w_attr * attr_score)
            
            results.append({
                "id": idx,
                "image_path": img_path,
                "category": category,
                "clothing_type": clothing_type,
                "color": color,
                "environment": env,
                "caption": caption,
                "style_tags": style_tags,
                "scores": {
                    "visual": vis_score,
                    "textual": txt_score,
                    "attribute": attr_score,
                    "final": final_score
                }
            })
            
        conn.close()
        
        # Sort by final score descending
        results.sort(key=lambda x: x["scores"]["final"], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query the fashion retrieval engine.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query string")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    retriever = FashionRetriever()
    print(f"Searching for query: '{args.query}'")
    results = retriever.search(args.query, top_k=args.k)
    
    print("\nSearch Results:")
    for rank, res in enumerate(results):
        print(f"Rank {rank+1}: ID={res['id']} (Score: {res['scores']['final']:.4f})")
        print(f"  Path: {res['image_path']}")
        print(f"  Caption: {res['caption']}")
        print(f"  Category: {res['category']} | Color: {res['color']} | Environment: {res['environment']}")
        print(f"  Score Breakdown: Visual={res['scores']['visual']:.4f}, Textual={res['scores']['textual']:.4f}, Attr={res['scores']['attribute']:.4f}")
        print("-" * 40)
