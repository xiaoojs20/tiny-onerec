import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path




def load_one(path: Path) -> np.ndarray:
    """Load one .npy/.npz embedding file -> (N, D) float32, and clean NaN/Inf."""
    obj = np.load(path, allow_pickle=False)  # safer loading :contentReference[oaicite:3]{index=3}

    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.files)
        if len(keys) == 1:
            arr = obj[keys[0]]
        elif "arr_0" in obj.files:
            arr = obj["arr_0"]
        else:
            raise ValueError(f"{path} is .npz with multiple arrays: {keys}")
    else:
        arr = obj

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 1D/2D array, got shape={arr.shape}")

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # in-place clean NaN/Inf :contentReference[oaicite:4]{index=4}
    if np.isnan(arr).any() or np.isinf(arr).any():
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def load_embeddings(data_path: str) -> np.ndarray:
    """If path is dir -> load all .npy/.npz and concat on axis0 (print row ranges)."""
    p = Path(data_path)

    if p.is_dir():
        files = sorted([*p.glob("*.npy"), *p.glob("*.npz")])
        if not files:
            raise FileNotFoundError(f"No .npy/.npz files in: {p}")

        arrays, start, dim = [], 0, None
        print(f"[EmbDataset] Loading directory: {p} (files={len(files)})")

        for f in files:
            a = load_one(f)
            dim = a.shape[1] if dim is None else dim
            if a.shape[1] != dim:
                raise ValueError(f"Dim mismatch: expected {dim}, got {a.shape[1]} in {f.name}")

            end = start + a.shape[0] - 1
            print(f"[EmbDataset] rows {start} ~ {end}: {f.name} (n={a.shape[0]}, dim={a.shape[1]})")
            start = end + 1
            arrays.append(a)

        return np.concatenate(arrays, axis=0)

    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {p}")

    return load_one(p)



class EmbDataset(data.Dataset):

    def __init__(self, data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        # self.embeddings = np.load(data_path)
        self.embeddings = load_embeddings(data_path)
        
        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0
        
        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
