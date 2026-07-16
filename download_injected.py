import requests
import os
from PIL import Image
import io

urls = {
    "yellow_raincoat": "https://images.unsplash.com/photo-1604176354204-9268737828e4?auto=format&fit=crop&w=400&h=400&q=80"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

for name, url in urls.items():
    print(f"Downloading {name}...")
    res = requests.get(url, headers=headers)
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        try:
            img = Image.open(io.BytesIO(res.content))
            img.save(f"test_inject/{name}.jpg")
            print(f"Successfully saved test_inject/{name}.jpg, size={img.size}")
        except Exception as e:
            print(f"Error opening image: {e}")
