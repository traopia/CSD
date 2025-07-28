import os
import json
import random
from typing import List, Tuple, Dict
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import sys
#sys.path.append('./CSD/CSD')  # Path to the CSD model code
from CSD.model import CSD_CLIP

# Paths
JSON_PATH = "fashion_directory.json"
IMAGES_DIR = "/Users/traopia/Library/CloudStorage/OneDrive-UvA/fashion_images/images"
SAMPLE_SIZE = 1000
CSD_WEIGHTS = "pretrainedmodels/checkpoint.pth"  # Update if your file is named differently

# Load model
model = CSD_CLIP(name='vit_large')
ckpt = torch.load(CSD_WEIGHTS, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# Preprocessing for CLIP ViT-L/14
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

from rembg import remove
from PIL import Image
import io

def remove_bg_pil(image: Image.Image) -> Image.Image:
    out = remove(image)
    return Image.open(io.BytesIO(out)) if isinstance(out, bytes) else out

def sample_images(json_path: str, sample_size: int) -> List[Dict]:
    with open(json_path, 'r') as f:
        first_lines = [next(f) for _ in range(10)]
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            sample_source = data
        elif isinstance(data, dict):
            sample_source = None
            for v in data.values():
                if isinstance(v, list):
                    sample_source = v
                    break
            if sample_source is None:
                raise ValueError("Could not find a list of entries in the JSON file.")
        else:
            raise ValueError("JSON root is neither a list nor a dict.")
        return random.sample(sample_source, min(sample_size, len(sample_source)))
    except Exception as e:
        print(f"Error loading JSON as list: {e}. Trying line-by-line sampling.")
        sample = []
        with open(json_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if i < sample_size:
                    sample.append(entry)
                else:
                    j = random.randint(0, i)
                    if j < sample_size:
                        sample[j] = entry
        return sample

def get_image_path(image_url: str) -> str:
    filename = image_url.replace("://", "-").replace("/", "_")
    return os.path.join(IMAGES_DIR, filename)

def extract_features(image_paths: List[str]) -> np.ndarray:
    features = []
    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(path).convert('RGB')
            #image = remove_bg_pil(image)
            img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                _, _, style_output = model(img_tensor)
            feat = style_output.squeeze().cpu().numpy()
            features.append(feat)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            features.append(np.zeros(1024))  # fallback for ViT-L
    return np.stack(features)

def split_query_database(entries: List[Dict], query_frac: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    random.shuffle(entries)
    n_query = int(len(entries) * query_frac)
    return entries[:n_query], entries[n_query:]

def retrieval(query_feats: np.ndarray, db_feats: np.ndarray, top_k: int = 5) -> np.ndarray:
    sims = cosine_similarity(query_feats, db_feats)
    return np.argsort(-sims, axis=1)[:, :top_k]

def evaluate_retrieval(query_entries: List[Dict], db_entries: List[Dict], retrieved_indices: np.ndarray) -> float:
    db_houses = [e['fashion_house'] for e in db_entries]
    correct = 0
    for i, q in enumerate(query_entries):
        top_k_indices = retrieved_indices[i, :]  # all top-k indices
        # Check if any of the top-k retrieved images have the same fashion house
        if any(db_houses[idx] == q['fashion_house'] for idx in top_k_indices):
            correct += 1
    return correct / len(query_entries)

import matplotlib.pyplot as plt

def show_retrieval(query_entry, db_entries, retrieved_indices, num_show=10):
    fig, axes = plt.subplots(1, num_show + 1, figsize=(2.5 * (num_show + 1), 3))
    # Show query image
    query_img = Image.open(query_entry['image_path']).convert('RGB')
    #query_img = remove_bg_pil(query_img)
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n{query_entry['fashion_house']}", fontsize=10)
    axes[0].axis('off')
    # Show retrieved images
    for i in range(num_show):
        idx = retrieved_indices[i]
        db_img = Image.open(db_entries[idx]['image_path']).convert('RGB')
        db_house = db_entries[idx]['fashion_house']
        axes[i + 1].imshow(db_img)
        axes[i + 1].set_title(f"Top {i+1}\n{db_house}", fontsize=10)
        axes[i + 1].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    entries = sample_images(JSON_PATH, SAMPLE_SIZE)
    for e in entries:
        e['image_path'] = get_image_path(e['image_url'])
    entries = [e for e in entries if os.path.exists(e['image_path'])]
    print(f"Found {len(entries)} images with files present.")
    query_entries, db_entries = split_query_database(entries)
    query_feats = extract_features([e['image_path'] for e in query_entries])
    db_feats = extract_features([e['image_path'] for e in db_entries])
    retrieved = retrieval(query_feats, db_feats, top_k=10)
    # Visualize retrievals for the first 5 queries
    for i in range(5):
        print(f"Query {i+1}: {query_entries[i]['fashion_house']}")
        show_retrieval(query_entries[i], db_entries, retrieved[i], num_show=10)
    acc = evaluate_retrieval(query_entries, db_entries, retrieved)
    print(f"Top-10 retrieval accuracy by fashion house: {acc:.2%}")

if __name__ == "__main__":
    main()