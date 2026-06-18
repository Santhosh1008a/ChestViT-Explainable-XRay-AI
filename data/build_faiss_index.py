"""
data/build_faiss_index.py
-------------------------
Builds a FAISS vector database from disease_descriptions.json
using sentence-transformers, laying groundwork for retrieval-augmented explanations.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_index(json_path: str, output_index_path: str):
    print("Loading disease descriptions...")
    with open(json_path, "r") as f:
        descriptions = json.load(f)

    diseases = list(descriptions.keys())
    texts = [descriptions[d] for d in diseases]

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to index
    index.add(embeddings)

    print(f"Index built with {index.ntotal} vectors of dimension {dimension}.")

    # Save to disk
    faiss.write_index(index, output_index_path)
    print(f"Saved FAISS index to {output_index_path}")

    # Save metadata mapping
    metadata_path = Path(output_index_path).with_suffix(".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(diseases, f)
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    build_index("data/disease_descriptions.json", "data/disease_knowledge.index")
