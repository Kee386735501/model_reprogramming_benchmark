
"""
dataset_distance.py

Compute dataset distances between two image folders using five metrics:
1) KL Divergence (Gaussian approximation on features)
2) Wasserstein Distance (Sliced Wasserstein-1 on features)
3) MMD (RBF-kernel, unbiased estimator) on features
4) FID (Fréchet Inception Distance) on features
5) KID (Kernel Inception Distance, unbiased) on features

Features are 2048-d from Inception v3 (pool3). Requires:
  - torch, torchvision, pillow

USAGE (single pair of folders, including "flat" folders like ...\\clipart\\cat):
  python dataset_distance.py --dataA <folder_a> --dataB <folder_b> [--device cuda|cpu]
                             [--batch 64] [--max-batches N]
                             [--kid-subsets 100] [--kid-size 100]
                             [--mmd-sigma S] [--sw-proj 128]
                             [--kl-symmetric] [--kl-diag] [--pca-d 256]

USAGE (per-class comparison, e.g., compare clipart\\<cls> vs real\\<cls> for all classes):
  python dataset_distance.py --dataA <rootA> --dataB <rootB> --per-class
                             [--classes cat,dog,car] [--out-csv per_class_metrics.csv]
                             [other metric args...]
"""

from typing import Optional, Tuple, Dict, List
import os
import glob
import csv
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------
# TorchVision (lazy import)
# --------------------
def _lazy_import_torchvision():
    from torchvision import datasets, transforms, models
    return datasets, transforms, models

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _list_images_flat(root: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, f"*{ext}")))
    if not files:
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    files = sorted(list(dict.fromkeys(files)))
    return files

class FlatImageDataset(Dataset):
    def __init__(self, root: str, transform):
        self.root = root
        self.transform = transform
        self.files = _list_images_flat(root)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under: {root}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        y = 0
        return x, y

def _has_class_subdirs(root: str) -> bool:
    try:
        for entry in os.scandir(root):
            if entry.is_dir():
                return True
        return False
    except FileNotFoundError:
        raise

def _list_immediate_subdirs(root: str) -> List[str]:
    try:
        return sorted([d.name for d in os.scandir(root) if d.is_dir()])
    except FileNotFoundError:
        raise

# -----------------------------
# InceptionV3 feature extractor
# -----------------------------
class InceptionV3Features(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        datasets, transforms, models = _lazy_import_torchvision()

        # Build inception_v3 in a version-robust way (avoid forcing aux_logits kwarg)
        try:
            w = models.Inception_V3_Weights.IMAGENET1K_V1
            self.backbone = models.inception_v3(weights=w)  # don't pass aux_logits explicitly
            self.preprocess = w.transforms()
        except Exception:
            # Fallback for very old torchvision versions
            self.backbone = models.inception_v3(pretrained=True)
            # Best-effort default transforms (Inception expects ~299 input)
            self.preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Disable auxiliary classifier post-hoc for feature extraction
        if hasattr(self.backbone, "aux_logits"):
            self.backbone.aux_logits = False
        if hasattr(self.backbone, "AuxLogits"):
            self.backbone.AuxLogits = None

        self.backbone.fc = nn.Identity()
        self.backbone.dropout = nn.Identity()
        self.device = torch.device(device)
        self.to(self.device).eval()
        for p in self.parameters():
            p.requires_grad = False
        self._datasets = datasets

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def dataloader(self, root: str, batch_size: int = 64, num_workers: int = 2, shuffle: bool = False) -> DataLoader:
        if _has_class_subdirs(root):
            ds = self._datasets.ImageFolder(root=root, transform=self.preprocess)
        else:
            ds = FlatImageDataset(root=root, transform=self.preprocess)
        
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    @torch.no_grad()
    def extract(self, loader: DataLoader, max_batches: Optional[int] = None) -> torch.Tensor:
        feats = []
        for i, (x, _) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            f = self.forward(x)
            feats.append(f.detach().cpu())
            if (max_batches is not None) and (i + 1 >= max_batches):
                break
        return torch.cat(feats, dim=0)

# -----------------------------
# Math helpers
# -----------------------------
def _stable_eigh_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = (mat + mat.T) * 0.5
    vals, vecs = np.linalg.eigh(mat + eps * np.eye(mat.shape[0], dtype=mat.dtype))
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def _cov(feats: np.ndarray) -> np.ndarray:
    return np.cov(feats, rowvar=False)

def _spd_regularize(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = (cov + cov.T) * 0.5
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, a_min=eps, a_max=None)
    return (vecs * vals) @ vecs.T

def _kl_gaussians(mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray, eps: float = 1e-6) -> float:
    """
    Numerically-stable KL( N_p || N_q ) for multivariate Gaussians.
    Uses SPD regularization and slogdet.
    """
    d = mu_p.shape[0]
    cov_p = _spd_regularize(cov_p, eps)
    cov_q = _spd_regularize(cov_q, eps)
    inv_q = np.linalg.inv(cov_q)
    term_trace = float(np.trace(inv_q @ cov_p))
    diff = (mu_q - mu_p).reshape(d, 1)
    term_quad = float((diff.T @ inv_q @ diff).squeeze())
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    if sign_q <= 0 or sign_p <= 0:
        cov_p = _spd_regularize(cov_p, eps*10)
        cov_q = _spd_regularize(cov_q, eps*10)
        sign_q, logdet_q = np.linalg.slogdet(cov_q)
        sign_p, logdet_p = np.linalg.slogdet(cov_p)
    term_logdet = float(logdet_q - logdet_p)
    return 0.5 * (term_trace + term_quad - d + term_logdet)

def _mmd_unbiased(X: torch.Tensor, Y: torch.Tensor, kernel: str = "rbf", **kwargs) -> float:
    assert X.dim() == 2 and Y.dim() == 2
    n = X.shape[0]
    m = Y.shape[0]
    if kernel == "rbf":
        sigma = kwargs.get("sigma", None)
        def pdist2(A, B):
            A2 = (A**2).sum(1, keepdim=True)
            B2 = (B**2).sum(1, keepdim=True).t()
            return A2 - 2*A @ B.t() + B2
        if sigma is None:
            with torch.no_grad():
                Z = torch.cat([X, Y], dim=0)
                D = pdist2(Z, Z)
                med = torch.median(D[D>0])
                sigma = torch.sqrt(med * 0.5 + 1e-8)
        gamma = 1.0 / (2.0 * sigma**2 + 1e-12)
        def rbf(A, B):
            return torch.exp(-gamma * pdist2(A, B))
        Kxx = rbf(X, X)
        Kyy = rbf(Y, Y)
        Kxy = rbf(X, Y)
        sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1))
        sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1))
        sum_xy = Kxy.mean()
        mmd2 = sum_xx + sum_yy - 2*sum_xy
        return float(mmd2.item())
    else:
        raise NotImplementedError("Only RBF kernel MMD is implemented.")

def _polynomial_kernel(X: torch.Tensor, Y: Optional[torch.Tensor] = None, degree: int = 3, coef0: float = 1.0) -> torch.Tensor:
    if Y is None:
        Y = X
    return (X @ Y.t() / X.shape[1] + coef0) ** degree

def _kid_unbiased(X: torch.Tensor, Y: torch.Tensor, degree: int = 3, coef0: float = 1.0, num_subsets: int = 100, subset_size: int = 100):
    import numpy as np
    n = X.shape[0]
    m = Y.shape[0]
    rng = np.random.default_rng(123)
    scores = []
    for _ in range(num_subsets):
        idx_x = torch.from_numpy(rng.integers(0, n, size=subset_size))
        idx_y = torch.from_numpy(rng.integers(0, m, size=subset_size))
        x = X[idx_x]
        y = Y[idx_y]
        Kxx = _polynomial_kernel(x, x, degree=degree, coef0=coef0)
        Kyy = _polynomial_kernel(y, y, degree=degree, coef0=coef0)
        Kxy = _polynomial_kernel(x, y, degree=degree, coef0=coef0)
        mmd = (Kxx.sum() - Kxx.trace()) / (subset_size*(subset_size-1)) \
              + (Kyy.sum() - Kyy.trace()) / (subset_size*(subset_size-1)) \
              - 2 * Kxy.mean()
        scores.append(float(mmd.item()))
    return float(np.mean(scores)), float(np.std(scores))

def _sliced_wasserstein(X: np.ndarray, Y: np.ndarray, num_projections: int = 128, seed: int = 42) -> float:
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(num_projections):
        v = rng.normal(size=(d, 1))
        v /= np.linalg.norm(v) + 1e-12
        x_proj = (X @ v).squeeze(1)
        y_proj = (Y @ v).squeeze(1)
        x_sorted = np.sort(x_proj)
        y_sorted = np.sort(y_proj)
        n = min(x_sorted.shape[0], y_sorted.shape[0])
        total += np.mean(np.abs(x_sorted[:n] - y_sorted[:n]))
    return total / num_projections

# -----------------------------
# Public metrics
# -----------------------------
def _apply_pca(fa: torch.Tensor, fb: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project features onto top-d principal components (fit on concatenated A+B)."""
    if d is None:
        return fa, fb
    A = fa.numpy()
    B = fb.numpy()
    X = np.concatenate([A, B], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d].T  # (feat_dim x d)
    A_proj = (A - X.mean(axis=0)) @ W
    B_proj = (B - X.mean(axis=0)) @ W
    return torch.from_numpy(A_proj), torch.from_numpy(B_proj)

def _gaussian_kl_from_feats(fa: torch.Tensor, fb: torch.Tensor, symmetric: bool, diag: bool) -> float:
    A = fa.numpy()
    B = fb.numpy()
    mu1, mu2 = A.mean(axis=0), B.mean(axis=0)
    if diag:
        var1 = A.var(axis=0) + 1e-6
        var2 = B.var(axis=0) + 1e-6
        # KL of diagonal Gaussians
        kl_pq = 0.5 * (np.sum(var1/var2) + np.sum((mu2-mu1)**2/var2) - A.shape[1] + np.sum(np.log(var2)) - np.sum(np.log(var1)))
        if symmetric:
            kl_qp = 0.5 * (np.sum(var2/var1) + np.sum((mu1-mu2)**2/var1) - A.shape[1] + np.sum(np.log(var1)) - np.sum(np.log(var2)))
            return float(0.5*(kl_pq + kl_qp))
        return float(kl_pq)
    else:
        cov1, cov2 = _cov(A), _cov(B)
        kl_pq = _kl_gaussians(mu1, cov1, mu2, cov2)
        if symmetric:
            kl_qp = _kl_gaussians(mu2, cov2, mu1, cov1)
            return float(0.5*(kl_pq + kl_qp))
        return float(kl_pq)

def compute_features_from_folders(folder_a: str, folder_b: str, batch_size: int = 64, device: str = None, max_batches: Optional[int] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = InceptionV3Features(device=device)
    loader_a = fe.dataloader(folder_a, batch_size=batch_size, shuffle=False)
    loader_b = fe.dataloader(folder_b, batch_size=batch_size, shuffle=False)
    fa = fe.extract(loader_a, max_batches=max_batches)
    fb = fe.extract(loader_b, max_batches=max_batches)
    return fa, fb

def metric_FID(feats_a: torch.Tensor, feats_b: torch.Tensor) -> float:
    A = feats_a.numpy()
    B = feats_b.numpy()
    mu1, mu2 = A.mean(axis=0), B.mean(axis=0)
    cov1, cov2 = _cov(A), _cov(B)
    diff = mu1 - mu2
    c1_half = _stable_eigh_sqrt(cov1)
    inner = c1_half @ cov2 @ c1_half
    trace_term = np.trace(cov1 + cov2 - 2.0 * _stable_eigh_sqrt(inner))
    fid = float(diff @ diff + trace_term)
    return fid

def metric_KID(feats_a: torch.Tensor, feats_b: torch.Tensor, degree: int = 3, coef0: float = 1.0, num_subsets: int = 100, subset_size: int = 100):
    return _kid_unbiased(feats_a, feats_b, degree=degree, coef0=coef0, num_subsets=num_subsets, subset_size=subset_size)

def metric_MMD(feats_a: torch.Tensor, feats_b: torch.Tensor, kernel: str = "rbf", **kwargs) -> float:
    return _mmd_unbiased(feats_a, feats_b, kernel=kernel, **kwargs)

def metric_Wasserstein(feats_a: torch.Tensor, feats_b: torch.Tensor, num_projections: int = 128) -> float:
    A = feats_a.numpy()
    B = feats_b.numpy()
    return float(_sliced_wasserstein(A, B, num_projections=num_projections))

def metric_KL(feats_a: torch.Tensor, feats_b: torch.Tensor, symmetric: bool = False, diag: bool = False, pca_d: Optional[int] = None) -> float:
    if pca_d is not None:
        feats_a, feats_b = _apply_pca(feats_a, feats_b, pca_d)
    return _gaussian_kl_from_feats(feats_a, feats_b, symmetric=symmetric, diag=diag)

def _compute_all_metrics(fa: torch.Tensor, fb: torch.Tensor, sw_proj: int, kid_subsets: int, kid_size: int, mmd_sigma: Optional[float], kl_symmetric: bool, kl_diag: bool, pca_d: Optional[int]):
    res = {}
    res["FID"] = metric_FID(fa, fb)
    km, ks = metric_KID(fa, fb, num_subsets=kid_subsets, subset_size=kid_size)
    res["KID_mean"] = km
    res["KID_std"] = ks
    sigma = None if mmd_sigma is None else torch.tensor(mmd_sigma)
    res["MMD_RBF"] = metric_MMD(fa, fb, sigma=sigma)
    res["Sliced_Wasserstein"] = metric_Wasserstein(fa, fb, num_projections=sw_proj)
    res["KL_Gauss"] = metric_KL(fa, fb, symmetric=kl_symmetric, diag=kl_diag, pca_d=pca_d)
    return res

# -----------------------------
# CLI
# -----------------------------
def _run_cli():
    import argparse, json
    p = argparse.ArgumentParser(description="Compute dataset distances (FID, KID, MMD, KL, Sliced Wasserstein). Supports ImageFolder or flat folders, plus per-class comparison.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataA", required=True, help="Path to dataset A (root or single-class folder)")
    p.add_argument("--dataB", required=True, help="Path to dataset B (root or single-class folder)")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if None)")
    p.add_argument("--batch", type=int, default=64, help="Batch size for feature extraction")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--max-batches", type=int, default=None, help="Optional cap on batches per dataset")
    p.add_argument("--kid-subsets", type=int, default=100, help="KID: number of random subsets")
    p.add_argument("--kid-size", type=int, default=100, help="KID: subset size")
    p.add_argument("--mmd-sigma", type=float, default=None, help="MMD RBF bandwidth (if None, median heuristic)")
    p.add_argument("--sw-proj", type=int, default=128, help="Number of random projections for sliced Wasserstein")
    p.add_argument("--kl-symmetric", action="store_true", help="Use symmetric KL = 0.5*(KL(P||Q)+KL(Q||P))")
    p.add_argument("--kl-diag", action="store_true", help="Use diagonal-covariance KL (very stable).")
    p.add_argument("--pca-d", type=int, default=None, help="Apply PCA to this dimensionality before KL (e.g., 256).")

    # per-class controls
    p.add_argument("--per-class", action="store_true", help="If set, iterate over class subfolders and compare per class.")
    p.add_argument("--classes", type=str, default=None, help="Comma-separated class names to compare (default: intersection of subfolders under A and B).")
    p.add_argument("--out-csv", type=str, default=f"per_class_metrics_{int(time.time())}.csv", help="Output CSV file for per-class mode.")
    p.add_argument("--summary", action="store_true", help="In per-class mode, also compute and append a MEAN row and print averages.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = InceptionV3Features(device=device)

    if not args.per_class:
        loader_a = fe.dataloader(args.dataA, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)
        loader_b = fe.dataloader(args.dataB, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)
        feats_a = fe.extract(loader_a, max_batches=args.max_batches)
        feats_b = fe.extract(loader_b, max_batches=args.max_batches)
        print(f"[Info] A feats: {tuple(feats_a.shape)}, B feats: {tuple(feats_b.shape)}")
        results = _compute_all_metrics(
            feats_a, feats_b,
            sw_proj=args.sw_proj,
            kid_subsets=args.kid_subsets,
            kid_size=args.kid_size,
            mmd_sigma=args.mmd_sigma,
            kl_symmetric=args.kl_symmetric,
            kl_diag=args.kl_diag,
            pca_d=args.pca_d
        )
        print(json.dumps(results, indent=2))
        return

    # ---- per-class mode ----
    # Determine classes to iterate
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        subA = set(_list_immediate_subdirs(args.dataA))
        subB = set(_list_immediate_subdirs(args.dataB))
        classes = sorted(list(subA & subB))
        if not classes:
            raise RuntimeError("No common class subfolders found between A and B. Provide --classes explicitly or check your roots.")

    rows = []
    for cls in classes:
        pathA = os.path.join(args.dataA, cls)
        pathB = os.path.join(args.dataB, cls)
        if not (os.path.isdir(pathA) and os.path.isdir(pathB)):
            print(f"[Skip] Missing folder for class '{cls}'. A: {os.path.isdir(pathA)}  B: {os.path.isdir(pathB)}")
            continue
        print(f"[Class] {cls} -> A: {pathA}  |  B: {pathB}")

        loader_a = fe.dataloader(pathA, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)
        loader_b = fe.dataloader(pathB, batch_size=args.batch, num_workers=args.num_workers, shuffle=False)
        feats_a = fe.extract(loader_a, max_batches=args.max_batches)
        feats_b = fe.extract(loader_b, max_batches=args.max_batches)
        print(f"[Info] {cls}: A feats {tuple(feats_a.shape)}, B feats {tuple(feats_b.shape)}")
        res = _compute_all_metrics(
            feats_a, feats_b,
            sw_proj=args.sw_proj,
            kid_subsets=args.kid_subsets,
            kid_size=args.kid_size,
            mmd_sigma=args.mmd_sigma,
            kl_symmetric=args.kl_symmetric,
            kl_diag=args.kl_diag,
            pca_d=args.pca_d
        )
        row = {"class": cls}
        row.update(res)
        rows.append(row)

    fieldnames = ["class", "FID", "KID_mean", "KID_std", "MMD_RBF", "Sliced_Wasserstein", "KL_Gauss"]
    # Optional: compute per-metric mean across classes
    mean_row = None
    if args.summary and len(rows) > 0:
        keys = [k for k in fieldnames if k != "class"]
        mean_vals = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        mean_row = {"class": "MEAN", **mean_vals}
        print({"MEAN": mean_vals, "num_classes": len(rows)})

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        if mean_row is not None:
            w.writerow(mean_row)

    print(f"[Saved] Per-class metrics -> {args.out_csv}")
    trailer = {"num_classes": len(rows), "csv": args.out_csv}
    if mean_row is not None:
        trailer.update({"mean_row": "MEAN"})
    print(trailer)

if __name__ == "__main__":
    _run_cli()
