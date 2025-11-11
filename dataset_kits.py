import os
import glob
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def _require_medimg():
    try:
        import nibabel as nib  # noqa: F401
        from scipy.ndimage import zoom  # noqa: F401
    except Exception as e:
        raise RuntimeError("nibabel and scipy are required for preprocessing KiTS. Install with: pip install nibabel scipy") from e


def _read_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    import nibabel as nib
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return np.asarray(data, dtype=np.float32), spacing


def _resample(image: np.ndarray, label: Optional[np.ndarray], in_spacing: Tuple[float, float, float], out_spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    from scipy.ndimage import zoom as ndzoom
    factors = (
        in_spacing[0] / out_spacing[0],
        in_spacing[1] / out_spacing[1],
        in_spacing[2] / out_spacing[2],
    )
    img_rs = ndzoom(image, zoom=factors, order=1)
    lbl_rs = None
    if label is not None:
        lbl_rs = ndzoom(label, zoom=factors, order=0)
    return img_rs.astype(np.float32), (lbl_rs.astype(np.int16) if lbl_rs is not None else None)


def _window_and_normalize(image: np.ndarray, clip: Tuple[float, float]) -> np.ndarray:
    lo, hi = clip
    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)
    return image.astype(np.float32)


def _case_paths(raw_root: str) -> List[Tuple[str, Optional[str], str]]:
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(raw_root, 'case_*'))):
        img_path = os.path.join(case_dir, 'imaging.nii.gz')
        seg_path = os.path.join(case_dir, 'segmentation.nii.gz')
        case_id = os.path.basename(case_dir)
        if os.path.isfile(img_path):
            cases.append((img_path, seg_path if os.path.isfile(seg_path) else None, case_id))
    return cases


def preprocess_kits(raw_root: str, out_root: str, target_spacing: Tuple[float, float, float] = (1.5, 1.5, 3.0), intensity_clip: Tuple[float, float] = (-200.0, 300.0), seed: int = 13) -> Dict:
    _require_medimg()
    os.makedirs(out_root, exist_ok=True)
    index_path = os.path.join(out_root, 'index.json')
    index = {"cases": []}
    cases = _case_paths(raw_root)
    rng = random.Random(seed)
    rng.shuffle(cases)
    for img_path, seg_path, case_id in cases:
        img, in_sp = _read_nifti(img_path)
        lbl = None
        if seg_path is not None and os.path.isfile(seg_path):
            lbl, _ = _read_nifti(seg_path)
        img, lbl = _resample(img, lbl, in_sp, target_spacing)
        img = _window_and_normalize(img, intensity_clip)
        save_path = os.path.join(out_root, f'{case_id}.npz')
        np.savez_compressed(save_path, image=img, label=(lbl if lbl is not None else np.array([], dtype=np.int16)), spacing=np.asarray(target_spacing, dtype=np.float32))
        index["cases"].append({"case_id": case_id, "file": os.path.basename(save_path), "has_label": bool(lbl is not None)})
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    return index


def make_federated_splits(preprocessed_root: str, num_clients: int, train_frac: float = 0.8, seed: int = 13) -> Dict:
    index_path = os.path.join(preprocessed_root, 'index.json')
    if not os.path.isfile(index_path):
        raise FileNotFoundError(index_path)
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    cases = [c for c in index.get("cases", []) if os.path.isfile(os.path.join(preprocessed_root, c["file"]))]
    rng = random.Random(seed)
    rng.shuffle(cases)
    clients = {str(i): {"train": [], "val": []} for i in range(num_clients)}
    for i, c in enumerate(cases):
        cid = str(i % num_clients)
        clients[cid]["train"].append(c)
    for cid, split in clients.items():
        n = len(split["train"]) 
        k = int(n * train_frac)
        split["val"] = split["train"][k:]
        split["train"] = split["train"][:k]
    out = {"num_clients": num_clients, "clients": clients}
    with open(os.path.join(preprocessed_root, 'splits.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def _load_npz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d = np.load(path, allow_pickle=False)
    img = d["image"].astype(np.float32)
    lbl = None
    if "label" in d and d["label"].size > 0:
        lbl = d["label"].astype(np.int16)
    return img, lbl


def _to_slices3d(arr: np.ndarray, axis: int = 2) -> List[np.ndarray]:
    return [np.take(arr, i, axis=axis) for i in range(arr.shape[axis])]


def load_client_dataset(preprocessed_root: str, client_id: str, mode: str = "train", slice_axis: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
    splits_path = os.path.join(preprocessed_root, 'splits.json')
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(splits_path)
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    if client_id not in splits.get("clients", {}):
        raise KeyError(client_id)
    items = []
    for c in splits["clients"][client_id].get(mode, []):
        case_file = os.path.join(preprocessed_root, c["file"]) 
        img, lbl = _load_npz(case_file)
        if slice_axis is None:
            x = {"image": img}
            if lbl is not None:
                x["label"] = lbl
            items.append(x)
        else:
            img_s = _to_slices3d(img, axis=slice_axis)
            lbl_s = _to_slices3d(lbl, axis=slice_axis) if lbl is not None else [None] * len(img_s)
            for a, b in zip(img_s, lbl_s):
                x = {"image": a}
                if b is not None:
                    x["label"] = b
                items.append(x)
    return items


def example_usage_preprocess_and_split(raw_root: str, out_root: str, num_clients: int = 5) -> Dict:
    _require_medimg()
    preprocess_kits(raw_root, out_root)
    return make_federated_splits(out_root, num_clients)
