import pandas as pd

df = pd.read_parquet("nba_injuries_2024_25.parquet")

# Voir ce qui génère des Unknown
unknown = df[df["injury_category"] == "Unknown"]
print(unknown["reason"].value_counts().head(20))