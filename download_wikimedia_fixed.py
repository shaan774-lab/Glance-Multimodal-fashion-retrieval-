import requests
from PIL import Image
import io
import os
import sys

# Reconfigure stdout to handle unicode correctly on Windows
sys.stdout.reconfigure(encoding='utf-8')

queries = {
    "yellow_raincoat": "yellow raincoat",
    "office_suit": "business suit office",
    "park_bench": "blue shirt bench",
    "city_walk": "street fashion walk",
    "red_tie": "red tie white shirt"
}

headers = {
    "User-Agent": "FashionRetrievalBot/1.0 (shaan@example.com) requests/2.33.1"
}

os.makedirs("test_inject_wiki", exist_ok=True)

def download_wikimedia_image(query, filename):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,  # File namespace
        "srlimit": 20,
        "format": "json"
    }
    
    try:
        res = requests.get(url, params=params, headers=headers).json()
        search_results = res.get("query", {}).get("search", [])
    except Exception as e:
        print(f"  Failed to query search for '{query}': {e}")
        return False
        
    for result in search_results:
        title = result["title"]
        # Skip if not an image
        title_lower = title.lower()
        if not (title_lower.endswith(".jpg") or title_lower.endswith(".jpeg") or title_lower.endswith(".png")):
            continue
            
        print(f"  Found file title: {title}")
        
        try:
            info_params = {
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json"
            }
            info_res = requests.get(url, params=info_params, headers=headers).json()
            pages = info_res.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                imageinfo = page_data.get("imageinfo", [])
                if imageinfo:
                    img_url = imageinfo[0]["url"]
                    print(f"    Attempting download from: {img_url}")
                    img_res = requests.get(img_url, headers=headers, timeout=10)
                    if img_res.status_code == 200:
                        img = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                        img_path = f"test_inject_wiki/{filename}.jpg"
                        img.thumbnail((400, 400))
                        img.save(img_path, "JPEG", quality=85)
                        print(f"    Successfully saved {img_path}")
                        return True
        except Exception as e:
            print(f"    Skipping {title} due to error: {e}")
            continue
            
    return False

for name, q in queries.items():
    print(f"Searching Wikimedia for '{q}'...")
    success = download_wikimedia_image(q, name)
    if not success:
        print(f"Failed to download image for '{q}'")
