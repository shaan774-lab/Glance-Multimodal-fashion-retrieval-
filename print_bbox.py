import pandas as pd
url = "https://huggingface.co/datasets/srpone/look-bench/resolve/main/v20251201/real_streetlook/gallery.parquet"
df = pd.read_parquet(url)
for i, row in df.head(5).iterrows():
    print("Category:", row["category"])
    print("Bbox:", row["bbox"])
    print("Main attribute:", row["main_attribute"])
    print("Other attributes:", row["other_attributes"])
    print("-" * 20)
