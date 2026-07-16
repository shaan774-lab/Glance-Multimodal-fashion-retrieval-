import requests
from PIL import Image
import io

url = "https://upload.wikimedia.org/wikipedia/commons/2/29/Yellow_Raincoat.jpg"
headers = {
    "User-Agent": "FashionRetrievalBot/1.0 (shaan@example.com) requests/2.33.1"
}

res = requests.get(url, headers=headers)
print("Status:", res.status_code)
if res.status_code == 200:
    try:
        img = Image.open(io.BytesIO(res.content))
        img.save("test_inject/yellow_raincoat.jpg")
        print("Success! Saved test_inject/yellow_raincoat.jpg, size=", img.size)
    except Exception as e:
        print("Error:", e)
