import os
import json
import sqlite3
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def main():
    print("Starting indexing pipeline...")
    
    # Load metadata
    metadata_path = "data/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file {metadata_path} not found.")
        return
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} images.")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model 'openai/clip-vit-base-patch32'...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Connect to SQLite
    db_path = "data/fashion_index.db"
    if os.path.exists(db_path):
        os.remove(db_path) # Start fresh
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fashion_items (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            category TEXT,
            clothing_type TEXT,
            color TEXT,
            environment TEXT,
            caption TEXT,
            style_tags TEXT,
            visual_embedding BLOB,
            text_embedding BLOB
        )
    """)
    conn.commit()
    
    batch_size = 32
    print("Extracting features and indexing...")
    
    for i in range(0, len(metadata), batch_size):
        batch = metadata[i:i+batch_size]
        images = []
        captions = []
        valid_indices = []
        
        for item in batch:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
                images.append(img)
                captions.append(item["caption"])
                valid_indices.append(item)
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")
                
        if not images:
            continue
            
        with torch.no_grad():
            # Extract visual embeddings
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_np = image_features.cpu().numpy()
            
            # Extract text embeddings of captions
            text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**text_inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_np = text_features.cpu().numpy()
            
        # Insert into SQLite
        for idx, item in enumerate(valid_indices):
            style_tags_str = ",".join(item["style_tags"])
            visual_blob = image_features_np[idx].astype(np.float32).tobytes()
            text_blob = text_features_np[idx].astype(np.float32).tobytes()
            
            cursor.execute("""
                INSERT INTO fashion_items (
                    id, image_path, category, clothing_type, color, environment, caption, style_tags, visual_embedding, text_embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["id"],
                item["image_path"],
                item["category"],
                item["clothing_type"],
                item["color"],
                item["environment"],
                item["caption"],
                style_tags_str,
                visual_blob,
                text_blob
            ))
            
        conn.commit()
        print(f"Indexed {i + len(batch)}/{len(metadata)} images...")
        
    conn.close()
    print("Indexing pipeline successfully completed!")

if __name__ == "__main__":
    main()
