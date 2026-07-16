import requests
from PIL import Image
import io
import os

ids = [
    "photo-1544816155-12df9643f363", # yellow raincoat
    "photo-1508873696983-2df519f0397e", # yellow raincoat
    "photo-1578587018452-892bacefd3f2"  # yellow raincoat
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

os.makedirs("test_yellow", exist_ok=True)
for i in ids:
    clean_id = i.replace('photo-', '')
    url = f"https://images.unsplash.com/photo-{clean_id}?auto=format&fit=crop&w=400&h=400&q=80"
    print("Downloading", url)
    res = requests.get(url, headers=headers)
    print("Status:", res.status_code)
    if res.status_code == 200:
        try:
            img = Image.open(io.BytesIO(res.content))
            img.save(f"test_yellow/{i}.jpg")
            print("Saved", i)
        except Exception as e:
            print("Error:", e)
