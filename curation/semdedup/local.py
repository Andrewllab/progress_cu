"""Local cosine SemDeDup runner, writing the released HDF5 writer's DataFrame.

No SLURM required. FAISS clustering runs on CPU; pairwise similarities use
PyTorch on CUDA when available. Run separately per task / compatible action space.
"""
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import torch
import tyro

from curation.semdedup.loader import EmbeddingLoader
from curation.semdedup.semdedup import SemDeDupJob


def main(embedding_dir: str, output_dir: str, ncentroids: int = 50,
         niter: int = 100, seed: int = 1234, image_weight: float = 0.8,
         action_key: str = "action", which_to_keep: str = "random"):
    if not 0 <= image_weight <= 1:
        raise ValueError("image_weight must be in [0, 1]")
    if which_to_keep not in ("random", "hard", "easy"):
        raise ValueError("which_to_keep must be random, hard or easy")
    modalities = {"image_embeds": image_weight, action_key: 1 - image_weight}
    loader = EmbeddingLoader(embedding_dir, modalities)
    data, metadata = loader.load_embeddings(normalization=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if not np.isfinite(data).all() or not 1 <= ncentroids <= len(data):
        raise ValueError("Embeddings must be finite and 1 <= ncentroids <= number of chunks")
    ids = metadata['id']['all'].astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs; encode tasks into separate directories")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(
        embedding_dir=embedding_dir, ncentroids=ncentroids, niter=niter,
        seed=seed, modalities=modalities, which_to_keep=which_to_keep,
    ), indent=2))
    kmeans = faiss.Kmeans(data.shape[1], ncentroids, niter=niter,
                          spherical=True, seed=seed, gpu=False)
    kmeans.train(data)
    similarity, assignment = kmeans.index.search(data, 1)
    np.save(out / "centroids.npy", kmeans.centroids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    job = SemDeDupJob(SimpleNamespace(seed=seed, sim_metric="cosine",
                                      Kmeans_with_cos_dist=True), 0)
    rng = random.Random(seed)
    frames = []
    for cluster_id in range(ncentroids):
        indices = np.flatnonzero(assignment[:, 0] == cluster_id)
        if not len(indices):
            continue
        indices = indices[np.argsort(similarity[indices, 0])].tolist()
        if which_to_keep == "random":
            rng.shuffle(indices)
        elif which_to_keep == "easy":
            indices.reverse()
        cluster = np.column_stack([ids[indices], indices])
        maximum, nearest = job.semdedup(cluster, torch.from_numpy(data[indices]), device)
        frames.append(pd.DataFrame(dict(
            max_sim=maximum.numpy(), cluster=cluster_id,
            most_similar_idx=nearest.numpy(), id_in_dataset=indices,
        ), index=ids[indices]))
    result = pd.concat(frames)
    with (out / "dedup.pkl").open("wb") as handle:
        pickle.dump(result, handle)
    print(f"Saved {len(result)} chunks to {out / 'dedup.pkl'}")


if __name__ == "__main__":
    tyro.cli(main)
