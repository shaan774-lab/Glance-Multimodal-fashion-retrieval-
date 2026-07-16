import os
import json
import io
import requests
import pandas as pd
import numpy as np
from PIL import Image

def main():
    print("Starting dataset download and generation...")
    os.makedirs("data/images", exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    
    # 1. Define evaluation target images to inject first
    eval_targets = [
        {
            "url": "https://images.unsplash.com/photo-1604176354204-9268737828e4?auto=format&fit=crop&w=400&h=400&q=80",
            "category": "raincoat",
            "clothing_type": "Casual",
            "color": "yellow",
            "environment": "park",
            "caption": "A person in a bright yellow raincoat walking outside.",
            "style_tags": ["casual", "rainy", "outdoor"]
        },
        {
            "url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=400&h=400&q=80",
            "category": "suit",
            "clothing_type": "Formal",
            "color": "black",
            "environment": "office",
            "caption": "Professional business attire inside a modern office.",
            "style_tags": ["formal", "business", "office"]
        },
        {
            "url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?auto=format&fit=crop&w=400&h=400&q=80",
            "category": "shirt",
            "clothing_type": "Casual",
            "color": "blue",
            "environment": "park",
            "caption": "Someone wearing a blue shirt sitting on a park bench.",
            "style_tags": ["casual", "park", "bench"]
        },
        {
            "url": "https://images.unsplash.com/photo-1485968579580-b6d095142e6e?auto=format&fit=crop&w=400&h=400&q=80",
            "category": "jacket",
            "clothing_type": "Casual",
            "color": "gray",
            "environment": "street",
            "caption": "Casual weekend outfit for a city walk.",
            "style_tags": ["casual", "weekend", "city"]
        },
        {
            "url": "https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?auto=format&fit=crop&w=400&h=400&q=80",
            "category": "tie",
            "clothing_type": "Formal",
            "color": "red",
            "environment": "office",
            "caption": "A red tie and a white shirt in a formal setting.",
            "style_tags": ["formal", "tie", "shirt", "office"]
        }
    ]
    
    metadata = []
    saved_count = 0
    
    print("Downloading and injecting evaluation target images...")
    for idx, target in enumerate(eval_targets):
        try:
            res = requests.get(target["url"], headers=headers, timeout=10)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content)).convert("RGB")
                img_filename = f"image_{saved_count:04d}.jpg"
                img_path = os.path.join("data/images", img_filename)
                img.save(img_path, "JPEG", quality=85)
                
                metadata.append({
                    "id": saved_count,
                    "image_path": img_path,
                    "category": target["category"],
                    "clothing_type": target["clothing_type"],
                    "color": target["color"],
                    "environment": target["environment"],
                    "caption": target["caption"],
                    "style_tags": target["style_tags"]
                })
                saved_count += 1
                print(f"Injected target {idx+1}: {target['caption']}")
            else:
                print(f"Warning: Failed to download target {idx+1}, status code: {res.status_code}")
        except Exception as e:
            print(f"Warning: Exception downloading target {idx+1}: {e}")
            
    # 2. Download look-bench dataset for the remaining images
    url = "https://huggingface.co/datasets/srpone/look-bench/resolve/main/v20251201/real_streetlook/gallery.parquet"
    print("Loading parquet from Hugging Face for remaining images...")
    df = pd.read_parquet(url)
    print(f"Loaded {len(df)} candidate rows.")
    
    colors_rgb = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "green": (0, 128, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "brown": (139, 69, 19),
        "beige": (245, 245, 220),
        "gray": (128, 128, 128),
        "pink": (255, 192, 203),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128)
    }
    
    def get_closest_color(img, bbox):
        try:
            if bbox is not None and len(bbox) == 4:
                w, h = img.size
                ymin, xmin, ymax, xmax = bbox
                if max(bbox) <= 1.0:
                    ymin, xmin, ymax, xmax = int(ymin*h), int(xmin*w), int(ymax*h), int(xmax*w)
                ymin, ymax = max(0, min(ymin, h)), max(0, min(ymax, h))
                xmin, xmax = max(0, min(xmin, w)), max(0, min(xmax, w))
                
                if (xmax - xmin) > 5 and (ymax - ymin) > 5:
                    crop = img.crop((xmin, ymin, xmax, ymax))
                else:
                    crop = img.crop((w//4, h//4, 3*w//4, 3*h//4))
            else:
                w, h = img.size
                crop = img.crop((w//4, h//4, 3*w//4, 3*h//4))
                
            crop_small = crop.resize((16, 16))
            pixels = np.array(crop_small).reshape(-1, 3)
            if pixels.shape[1] == 4:
                pixels = pixels[:, :3]
            avg_rgb = pixels.mean(axis=0)
            
            min_dist = float('inf')
            closest = "black"
            for color_name, rgb in colors_rgb.items():
                dist = np.linalg.norm(avg_rgb - rgb)
                if dist < min_dist:
                    min_dist = dist
                    closest = color_name
            return closest
        except Exception as e:
            return "black"
            
    category_to_clothing = {
        "shirt": "Formal", "blouse": "Formal", "pants": "Formal", "tie": "Formal", "vest": "Formal", "belt": "Formal",
        "t-shirt": "Casual", "white t-shirt": "Casual", "hoodie": "Casual", "jeans": "Casual", "shorts": "Casual",
        "sweatshirt": "Casual", "sweater": "Casual", "dress": "Casual", "skirt": "Casual", "socks": "Casual", "sock": "Casual",
        "jacket": "Outerwear", "coat": "Outerwear", "trench coat": "Outerwear", "cardigan": "Outerwear", "cape": "Outerwear"
    }
    
    environments = ["office", "park", "street", "home"]
    # Adjust targets to account for injected images
    env_counts = {"office": 0, "park": 0, "street": 0, "home": 0}
    for m in metadata:
        env_counts[m["environment"]] += 1
        
    env_targets = {"office": 150, "park": 150, "street": 150, "home": 150}
    
    category_to_env = {
        "pants": "office", "shirt": "office", "blouse": "office", "tie": "office", "vest": "office",
        "dress": "park", "skirt": "park", "sandals": "park",
        "hoodie": "street", "t-shirt": "street", "jeans": "street", "jacket": "street", "coat": "street", "trench coat": "street",
        "sweater": "home", "cardigan": "home", "sweatshirt": "home", "sock": "home", "tights": "home"
    }
    
    total_images_needed = 600
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("Downloading remaining images from look-bench...")
    for idx, row in df.iterrows():
        if saved_count >= total_images_needed:
            break
            
        try:
            img_dict = row["image"]
            if img_dict is None or "bytes" not in img_dict:
                continue
                
            img_bytes = img_dict["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            cat = str(row["category"]).lower()
            clothing_type = category_to_clothing.get(cat, "Casual")
            color = get_closest_color(img, row["bbox"])
            
            suggested_env = category_to_env.get(cat, None)
            if suggested_env is None or env_counts[suggested_env] >= env_targets[suggested_env]:
                open_envs = [env for env, count in env_counts.items() if count < env_targets[env]]
                if not open_envs:
                    break
                suggested_env = min(open_envs, key=lambda e: env_counts[e])
                
            env = suggested_env
            env_counts[env] += 1
            
            img_filename = f"image_{saved_count:04d}.jpg"
            img_path = os.path.join("data/images", img_filename)
            
            img.thumbnail((400, 400))
            img.save(img_path, "JPEG", quality=85)
            
            env_names = {
                "office": "office interior",
                "park": "park green setting",
                "street": "urban street",
                "home": "home setting"
            }
            
            style_tags = [str(x) for x in row["other_attributes"]] if row["other_attributes"] is not None else []
            style_str = ", ".join(style_tags[:2])
            style_desc = f" in {style_str} style" if style_str else ""
            
            caption = f"A person wearing a {color} {cat}{style_desc} seen in a {env_names[env]}."
            
            metadata.append({
                "id": saved_count,
                "image_path": img_path,
                "category": cat,
                "clothing_type": clothing_type,
                "color": color,
                "environment": env,
                "caption": caption,
                "style_tags": style_tags
            })
            
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"Downloaded and processed {saved_count}/{total_images_needed} images...")
                
        except Exception as e:
            continue
            
    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Dataset generation complete! Generated {saved_count} images.")
    print("Final Environment counts:", env_counts)

if __name__ == "__main__":
    main()
