"""
ingest.py
---------
Step 1 of the RAG pipeline: turn a CSV of Indian food nutrition data
into a searchable vector database.

What this script does, in order:
1. Reads the CSV into a table (pandas)
2. Converts each row into one plain-English sentence ("chunking")
3. Runs each sentence through an embedding model (turns text -> numbers)
4. Saves everything into a local Chroma vector database on disk

Run this ONCE (or again whenever your data changes) before running query.py
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "indian_food_sample.csv"
DB_FOLDER = "chroma_db"          # Chroma will create this folder on disk
COLLECTION_NAME = "indian_food_nutrition"


def row_to_text(row) -> str:
    """
    Turn one dataset row into a natural-language sentence.
    This is the 'chunk' that actually gets embedded and searched.
    Keeping it descriptive (not just raw numbers) helps the embedding
    model understand what it's looking at.
    """
    return (
        f"{row['dish_name']} is a {row['category']} dish with "
        f"{row['calories_kcal_per_100g']} kcal, "
        f"{row['protein_g']}g protein, "
        f"{row['carbs_g']}g carbohydrates, and "
        f"{row['fat_g']}g fat per 100g serving."
    )


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} dishes.")

    print("Converting rows into text chunks...")
    documents = [row_to_text(row) for _, row in df.iterrows()]
    ids = [f"dish-{i}" for i in range(len(df))]
    # metadata lets us pull back the raw numbers later without re-parsing text
    metadatas = df.to_dict(orient="records")

    print("Loading local embedding model (first run will download it, "
          "needs internet once)...")
    # all-MiniLM-L6-v2 is small, free, and runs on CPU -- good for learning
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    print("Connecting to local Chroma database...")
    client = chromadb.PersistentClient(path=DB_FOLDER)

    # If you re-run ingest.py after changing the CSV, start fresh:
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    print("Embedding and storing all dishes...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"Done. Stored {collection.count()} dishes in '{DB_FOLDER}/'.")
    print("You can now run query.py")


if __name__ == "__main__":
    main()
