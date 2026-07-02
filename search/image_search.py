import torch
from PIL import Image

from search.loader import loader


def search_by_image(image_path, top_k=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = loader.clip_preprocess(image).unsqueeze(0).to(loader.device)

    with torch.no_grad():
        image_embedding = loader.clip_model.encode_image(image_tensor)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    image_embedding = image_embedding.cpu().numpy().astype("float32")

    scores, indices = loader.clip_index.search(image_embedding, top_k)

    results = loader.df_clip.iloc[indices[0]].copy()
    results["image_score"] = scores[0]

    return results