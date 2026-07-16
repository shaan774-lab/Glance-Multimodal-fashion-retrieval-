import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://huggingface.co/datasets/srpone/look-bench/resolve/main/v20251201/real_streetlook/gallery.parquet"
df = pd.read_parquet(url)
print("Unique categories:", [str(x) for x in df["category"].unique()])
print("\nUnique main_attribute:", [str(x) for x in df["main_attribute"].unique()[:20]])
print("\nSample other_attributes:")
for x in df["other_attributes"].head(10):
    print([str(i) for i in x] if x is not None else None)
