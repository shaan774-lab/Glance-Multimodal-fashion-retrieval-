import pandas as pd
url = "https://huggingface.co/datasets/orianrivlin/fashion-lookmatch-dataset/resolve/main/data/train-00000-of-00003.parquet"
df = pd.read_parquet(url)
print("Unique categories:", df["category"].unique())
print("Unique colors:", df["color"].unique()[:10])
print("\nSample style_tags:")
print(df["style_tags"].head(10))
print("\nSample captions:")
for i, row in df.head(5).iterrows():
    print(f"- Caption: {row['caption']}")
    print(f"  Prompt: {row['prompt']}")
    print("  Style tags:", row['style_tags'])
    print("  Color:", row['color'])
