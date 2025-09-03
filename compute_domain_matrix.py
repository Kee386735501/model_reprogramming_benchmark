"""
compute_domain_matrix.py

Build pairwise domain-distance matrices over DomainNet-style roots by
calling dataset_distance's feature extractor and metrics. For each
domain pair (A,B), it computes per-class metrics and then averages
across all common classes, yielding one scalar per metric. Finally it
writes one CSV per metric with a symmetric matrix and an optional JSON
summary.

Usage example:
  python compute_domain_matrix.py \
    --root /data/gpfs/projects/punim1943/domainnet \
    --device cuda --per-class --summary \
    --out-prefix domainnet

You can pass through the same metric arguments as dataset_distance.py:
  --batch, --num-workers, --max-batches, --kid-subsets, --kid-size,
  --mmd-sigma, --sw-proj, --kl-symmetric, --kl-diag, --pca-d

Notes:
  - This script caches features per (domain, class) so that each
    domain/class features are extracted only once across all pairs.
  - The output matrices include: FID, KID_mean, KID_std, MMD_RBF,
    Sliced_Wasserstein, KL_Gauss.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

import dataset_distance as dd


DOMAINS_DEFAULT = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
]


def _classes_intersection(path_a: str, path_b: str) -> List[str]:
    subA = set(dd._list_immediate_subdirs(path_a))
    subB = set(dd._list_immediate_subdirs(path_b))
    classes = sorted(list(subA & subB))
    if not classes:
        raise RuntimeError(
            f"No common class subfolders found between:\n A={path_a}\n B={path_b}"
        )
    return classes


def compute_pair_mean(
    fe: dd.InceptionV3Features,
    root: str,
    domA: str,
    domB: str,
    *,
    batch: int = 64,
    num_workers: int = 2,
    max_batches: int | None = None,
    kid_subsets: int = 100,
    kid_size: int = 100,
    mmd_sigma: float | None = None,
    sw_proj: int = 128,
    kl_symmetric: bool = False,
    kl_diag: bool = False,
    pca_d: int | None = None,
    feature_cache: Dict[Tuple[str, str], torch.Tensor] | None = None,
) -> Dict[str, float]:
    """Compute per-class metrics for (domA, domB) and return mean of each metric.

    Features for each (domain, class) are cached in feature_cache if provided.
    """
    pathA = os.path.join(root, domA)
    pathB = os.path.join(root, domB)
    classes = _classes_intersection(pathA, pathB)

    # accumulate rows per class
    rows = []
    for cls in classes:
        keyA = (domA, cls)
        keyB = (domB, cls)
        if feature_cache is not None and keyA in feature_cache:
            fa = feature_cache[keyA]
        else:
            la = fe.dataloader(os.path.join(pathA, cls), batch_size=batch, num_workers=num_workers, shuffle=False)
            fa = fe.extract(la, max_batches=max_batches)
            if feature_cache is not None:
                feature_cache[keyA] = fa

        if feature_cache is not None and keyB in feature_cache:
            fb = feature_cache[keyB]
        else:
            lb = fe.dataloader(os.path.join(pathB, cls), batch_size=batch, num_workers=num_workers, shuffle=False)
            fb = fe.extract(lb, max_batches=max_batches)
            if feature_cache is not None:
                feature_cache[keyB] = fb

        res = dd._compute_all_metrics(
            fa, fb,
            sw_proj=sw_proj,
            kid_subsets=kid_subsets,
            kid_size=kid_size,
            mmd_sigma=mmd_sigma,
            kl_symmetric=kl_symmetric,
            kl_diag=kl_diag,
            pca_d=pca_d,
        )
        rows.append(res)

    # mean across classes
    keys = ["FID", "KID_mean", "KID_std", "MMD_RBF", "Sliced_Wasserstein", "KL_Gauss"]
    mean_vals = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return mean_vals


def write_matrix_csv(path: str, labels: List[str], mat: List[List[float | str]]):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row in enumerate(mat):
            w.writerow([labels[i]] + row)


def main(args):
    p = argparse.ArgumentParser(description="Compute pairwise domain distance matrices using dataset_distance metrics.")
    p.add_argument("--root",default="./data", help="Root path containing domain folders (e.g., clipart, painting, ...)")
    p.add_argument("--domains", type=str, default=",".join(DOMAINS_DEFAULT), help="Comma-separated domain folder names to include")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if None)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--kid-size", type=int, default=100)
    p.add_argument("--mmd-sigma", type=float, default=None)
    p.add_argument("--sw-proj", type=int, default=128)
    p.add_argument("--kl-symmetric", action="store_true")
    p.add_argument("--kl-diag", action="store_true")
    p.add_argument("--pca-d", type=int, default=None)
    p.add_argument("--out-prefix", type=str, default="domainnet")
    p.add_argument("--save-json", action="store_true")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = dd.InceptionV3Features(device=device)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    n = len(domains)

    metrics_keys = ["FID", "KID_mean", "KID_std", "MMD_RBF", "Sliced_Wasserstein", "KL_Gauss"]
    matrices: Dict[str, List[List[float | str]]] = {k: [["-" for _ in range(n)] for __ in range(n)] for k in metrics_keys}

    cache: Dict[Tuple[str, str], torch.Tensor] = {}

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal stays as '-'
                continue
            domA, domB = domains[i], domains[j]
            print(f"[Pair] {domA} vs {domB}")
            mean_vals = compute_pair_mean(
                fe, args.root, domA, domB,
                batch=args.batch,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
                kid_subsets=args.kid_subsets,
                kid_size=args.kid_size,
                mmd_sigma=args.mmd_sigma,
                sw_proj=args.sw_proj,
                kl_symmetric=args.kl_symmetric,
                kl_diag=args.kl_diag,
                pca_d=args.pca_d,
                feature_cache=cache,
            )
            for k in metrics_keys:
                val = mean_vals[k]
                matrices[k][i][j] = val
                matrices[k][j][i] = val  # symmetry

    # write CSVs
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for k in metrics_keys:
        out_csv = f"{args.out_prefix}_matrix_{k}.csv"
        write_matrix_csv(out_csv, domains, matrices[k])
        print(f"[Saved] {out_csv}")

    if args.save_json:
        out_json = f"{args.out_prefix}_matrices.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({k: matrices[k] for k in metrics_keys}, f, indent=2)
        print(f"[Saved] {out_json}")


if __name__ == "__main__":
    class Args: pass
    args = Args()
    args.root = "./data"
    args.domains = "clipart,infograph,painting,quickdraw,real,sketch"
    args.device = "cuda"
    args.batch = 64
    args.num_workers = 4
    args.kid_subsets = 100
    args.kid_size = 100
    args.sw_proj = 128
    args.kl_symmetric = True
    args.kl_diag = True
    args.pca_d = 256
    args.out_prefix = "outputs/domainnet"
    args.save_json = True

    main(args)

