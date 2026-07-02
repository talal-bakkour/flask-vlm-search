import torch
from search.loader import loader


def search_clip(query, top_k=20):
    tokens = loader.clip_tokenizer([query]).to(loader.device)

    with torch.no_grad():
        query_embedding = loader.clip_model.encode_text(tokens)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    query_embedding = query_embedding.cpu().numpy().astype("float32")

    scores, indices = loader.clip_index.search(query_embedding, top_k)

    results = loader.df_clip.iloc[indices[0]].copy()
    results["clip_score"] = scores[0]

    return results