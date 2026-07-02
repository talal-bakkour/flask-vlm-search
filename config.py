from pathlib import Path

# ==========================================
# Project Directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "images"

# ==========================================
# Metadata
# ==========================================

TEXT_METADATA = MODELS_DIR / "df_text_metadata.csv"
CLIP_METADATA = MODELS_DIR / "df_clip_metadata.csv"

# ==========================================
# FAISS
# ==========================================

TEXT_FAISS = MODELS_DIR / "text_faiss.index"
CLIP_FAISS = MODELS_DIR / "clip_faiss.index"

# ==========================================
# Embeddings
# ==========================================

CLIP_EMBEDDINGS = MODELS_DIR / "clip_embeddings.npy"

# ==========================================
# Models
# ==========================================

TEXT_MODEL = "BAAI/bge-large-en-v1.5"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ==========================================
# Retrieval
# ==========================================

TOP_K = 20

TEXT_WEIGHT = 0.7
CLIP_WEIGHT = 0.3