import numpy as np
from search.loader import loader


def search_text(query, top_k=20):
    query_embedding = loader.text_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = loader.text_index.search(query_embedding, top_k)

    results = loader.df_text.iloc[indices[0]].copy()
    results["score"] = scores[0]

    return results