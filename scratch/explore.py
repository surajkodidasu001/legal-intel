from datasets import load_dataset

ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
print(ds)
print("classes:", len(ds.features["label"].names))
print(ds.features["label"].names[:10])

for r in ds.select(range(3)):
    print("\n---", ds.features["label"].names[r["label"]])
    print(repr(r["text"][:300]))

df = ds.to_pandas()
lens = df["text"].str.split().str.len()
print("\nword counts:\n", lens.describe())
print("\ntop classes:\n", df["label"].value_counts(normalize=True).head(10))

print("\ntail classes:\n", df["label"].value_counts(normalize=True).tail(10))
print("\nrarest class count:", df["label"].value_counts().min())
print("\ndocs under 10 words:", (lens < 10).sum())
print(df.loc[lens < 10, "text"].head(5).tolist())
