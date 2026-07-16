import pandas as pd
import io
import os
from PIL import Image

# Use the Hugging Face parquet URL which is highly reliable
url0 = "https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/main/data/train-00000-of-00002-6cff4c59f91661c3.parquet"
url1 = "https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/main/data/train-00001-of-00002-bb459e5ac5f01e71.parquet"

print("Loading dataset part 0...")
df = pd.read_parquet(url0)
print(f"Loaded {len(df)} rows.")

os.makedirs("test_inject", exist_ok=True)

# 1. Find Yellow Jacket
yellow_jackets = df[
    (df["baseColour"].str.lower() == "yellow") & 
    (df["articleType"].str.lower().isin(["jackets", "sweatshirts", "coats", "jackets"]))
]
print("Yellow jackets found in Part 0:", len(yellow_jackets))
if not yellow_jackets.empty:
    row = yellow_jackets.iloc[0]
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    img.save("test_inject/yellow_raincoat.jpg")
    print("Saved yellow_raincoat.jpg from Part 0, ID:", row["id"])
else:
    print("Searching Part 1 for yellow jackets...")
    df1 = pd.read_parquet(url1)
    yellow_jackets = df1[
        (df1["baseColour"].str.lower() == "yellow") & 
        (df1["articleType"].str.lower().isin(["jackets", "sweatshirts", "coats", "jackets"]))
    ]
    if not yellow_jackets.empty:
        row = yellow_jackets.iloc[0]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img.save("test_inject/yellow_raincoat.jpg")
        print("Saved yellow_raincoat.jpg from Part 1, ID:", row["id"])

# 2. Find Blue Shirt
blue_shirts = df[
    (df["baseColour"].str.lower() == "blue") & 
    (df["articleType"].str.lower() == "shirts")
]
print("Blue shirts found in Part 0:", len(blue_shirts))
if not blue_shirts.empty:
    row = blue_shirts.iloc[0]
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    img.save("test_inject/park_bench.jpg")
    print("Saved park_bench.jpg (blue shirt) from Part 0, ID:", row["id"])

# 3. Find Red Tie
red_ties = df[
    (df["baseColour"].str.lower() == "red") & 
    (df["articleType"].str.lower() == "ties")
]
print("Red ties found in Part 0:", len(red_ties))
if not red_ties.empty:
    row = red_ties.iloc[0]
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    img.save("test_inject/red_tie.jpg")
    print("Saved red_tie.jpg from Part 0, ID:", row["id"])
