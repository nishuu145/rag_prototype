"""
query.py
--------
Step 2 of the RAG pipeline: given something a user typed (e.g. "I had
2 rotis and dal for lunch"), find the most relevant dishes from the
vector database, then (optionally) send them to an LLM to get a final
nutrition estimate.

Run ingest.py first -- this script just reads the database it created.
"""

import chromadb
from chromadb.utils import embedding_functions

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "indian_food_nutrition"
TOP_K = 3  # how many matching dishes to retrieve


def retrieve(user_query: str):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    results = collection.query(query_texts=[user_query], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas


def build_prompt(user_query: str, docs: list[str]) -> str:
    """
    This is the 'augmented' part of RAG: we hand the model real data
    instead of letting it guess from memory.
    """
    context = "\n".join(f"- {d}" for d in docs)
    return (
        "You are a nutrition assistant. Use ONLY the reference data below "
        "to answer -- do not use outside knowledge of nutrition values.\n\n"
        f"Reference data:\n{context}\n\n"
        f"User's meal: \"{user_query}\"\n\n"
        "Estimate the total calories, protein, carbs, and fat for what the "
        "user described. Show your reasoning briefly, then give a final total."
    )


def call_llm(prompt: str) -> str:
    """
    OPTIONAL step -- sends the prompt to an LLM API so it can reason over
    the retrieved data and produce a final answer.

    Uses the Anthropic API here as an example. Swap this out for whichever
    API your team's FastAPI backend already uses (OpenAI, etc.) -- the
    pattern (retrieve -> build prompt -> call LLM) stays the same either way.

    You'll need to: pip install anthropic
    and set an API key: export ANTHROPIC_API_KEY="your-key-here"
    """
    import os
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("[Skipped LLM call -- no ANTHROPIC_API_KEY set. "
                "The retrieved context above is what would have been sent.]")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    print("Type a meal description (or 'quit' to exit).\n")
    while True:
        user_query = input("Meal> ").strip()
        if user_query.lower() in ("quit", "exit"):
            break
        if not user_query:
            continue

        docs, metas = retrieve(user_query)

        print("\n--- Retrieved matches (nearest by meaning) ---")
        for d in docs:
            print(f"  {d}")

        prompt = build_prompt(user_query, docs)
        print("\n--- Prompt that would be sent to the LLM ---")
        print(prompt)

        print("\n--- LLM answer ---")
        print(call_llm(prompt))
        print()


if __name__ == "__main__":
    main()
