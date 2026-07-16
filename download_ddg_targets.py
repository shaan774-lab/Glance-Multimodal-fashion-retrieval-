from duckduckgo_search import DDGS
import requests
from PIL import Image
import io
import os

queries = {
    "yellow_raincoat": "person in bright yellow raincoat outside",
    "office_suit": "professional business attire inside modern office",
    "park_bench": "person in blue shirt sitting on park bench",
    "city_walk": "casual weekend outfit for a city walk fashion",
    "red_tie": "red tie and white shirt in formal suit setting"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

os.makedirs("test_inject", exist_ok=True)

with DDGS() as ddgs:
    for name, q in queries.items():
        print(f"Searching for query '{q}'...")
        try:
            results = ddgs.images(q, max_results=10)
            success = False
            for r in results:
                img_url = r["image"]
                print(f"  Trying URL: {img_url}...")
                try:
                    res = requests.get(img_url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        img = Image.open(io.BytesIO(res.content)).convert("RGB")
                        img_path = f"test_inject/{name}.jpg"
                        img.save(img_path, "JPEG", quality=85)
                        print(f"  Successfully saved {img_path}, size={img.size}")
                        success = True
                        break
                except Exception as e:
                    print(f"  Failed: {e}")
            if not success:
                print(f"Error: Could not download any images for query '{q}'")
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
