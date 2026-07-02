import torch
import faiss
import pandas as pd
import open_clip

from sentence_transformers import SentenceTransformer

from config import (
    TEXT_METADATA,
    CLIP_METADATA,
    TEXT_FAISS,
    CLIP_FAISS,
    TEXT_MODEL,
    CLIP_MODEL,
    CLIP_PRETRAINED,
)


class ModelLoader:
    def __init__(self):
        print("Loading metadata...")
        self.df_text = pd.read_csv(TEXT_METADATA)
        self.df_clip = pd.read_csv(CLIP_METADATA)

        print("Loading FAISS indexes...")
        self.text_index = faiss.read_index(str(TEXT_FAISS))
        self.clip_index = faiss.read_index(str(CLIP_FAISS))

        print("Loading text model...")
        self.text_model = SentenceTransformer(TEXT_MODEL)

        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL,
            pretrained=CLIP_PRETRAINED
        )

        self.clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        print("All models loaded successfully.")
        print("Device:", self.device)


loader = ModelLoader()