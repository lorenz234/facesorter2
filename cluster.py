#!/usr/bin/env python3
"""
cluster.py — automatic face clustering.

Scans an input folder of photos, detects every face with InsightFace
(antelopev2), extracts a 512-dim embedding per face, then groups the faces
into people with HDBSCAN. No reference photos / calibration needed.

Outputs:
  out/embeddings.npz   all face embeddings + metadata (for resume / re-cluster)
  out/clusters.csv     file, face_index, cluster, score

Run:
  python cluster.py --data all_pictures --out out
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Optional HEIC/HEIF support (iPhone photos). Registered lazily in load_image().
try:
    import pillow_heif  # noqa: F401

    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False

# HEIF-family formats (iPhone / mirrorless cameras) decoded via pillow-heif.
# ".hif" is the extension some cameras (e.g. Canon) use for HEIF stills.
HEIF_EXTS = {".heic", ".heif", ".hif"}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
} | HEIF_EXTS


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_image(path: Path):
    """Return a BGR uint8 numpy array (OpenCV layout) or None on failure."""
    import cv2

    ext = path.suffix.lower()
    if ext in HEIF_EXTS:
        if not _HEIF_OK:
            return None
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def find_images(data_dir: Path) -> list[Path]:
    """All supported images under data_dir, sorted for stable ordering."""
    files = [
        p for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files


# --------------------------------------------------------------------------- #
# Face detection / embedding
# --------------------------------------------------------------------------- #
def _flatten_nested_model(model: str, root: Path) -> None:
    """Work around the antelopev2 release zip, which extracts its .onnx files
    into a doubly-nested  models/<model>/<model>/  folder. Move them up one
    level so InsightFace can find the detection/recognition models."""
    base = root / "models" / model
    nested = base / model
    if (nested.is_dir() and any(nested.glob("*.onnx"))
            and not any(base.glob("*.onnx"))):
        for f in nested.iterdir():
            shutil.move(str(f), str(base / f.name))
        try:
            nested.rmdir()
        except OSError:
            pass


def build_analyzer(model: str, det_size: int):
    """Construct and prepare an InsightFace FaceAnalysis app (CPU).

    The first call downloads the model pack (~350 MB) into ~/.insightface.
    """
    from insightface.app import FaceAnalysis

    root = Path.home() / ".insightface"
    kwargs = dict(
        name=model,
        root=str(root),
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    try:
        app = FaceAnalysis(**kwargs)
    except AssertionError:
        # Detection model missing -> almost always the nested-folder bug.
        _flatten_nested_model(model, root)
        app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def detect_and_embed(app, data_dir: Path, files: list[Path]):
    """Run detection on every file. Returns lists of per-face records plus the
    list of all images we successfully processed (even those with no face, so
    --resume doesn't keep re-scanning face-less photos)."""
    from tqdm import tqdm

    embeddings: list[np.ndarray] = []
    rel_files: list[str] = []
    face_indices: list[int] = []
    bboxes: list[np.ndarray] = []
    det_scores: list[float] = []
    processed: list[str] = []

    n_faces = 0
    n_skipped = 0
    for path in tqdm(files, desc="Detecting faces", unit="img"):
        img = load_image(path)
        if img is None:
            n_skipped += 1
            tqdm.write(f"  ! could not read {path}")
            continue
        try:
            faces = app.get(img)
        except Exception as exc:  # noqa: BLE001
            n_skipped += 1
            tqdm.write(f"  ! detection failed for {path}: {exc}")
            continue

        rel = str(path.relative_to(data_dir))
        processed.append(rel)
        for idx, face in enumerate(faces):
            emb = face.normed_embedding  # already L2-normalized, 512-dim
            if emb is None:
                continue
            embeddings.append(emb.astype(np.float32))
            rel_files.append(rel)
            face_indices.append(idx)
            bboxes.append(face.bbox.astype(np.float32))
            det_scores.append(float(face.det_score))
            n_faces += 1

    print(f"\nDetected {n_faces} faces in {len(processed)} images "
          f"({n_skipped} skipped).")
    return embeddings, rel_files, face_indices, bboxes, det_scores, processed


# --------------------------------------------------------------------------- #
# Persistence (embeddings.npz)
# --------------------------------------------------------------------------- #
def save_embeddings(path: Path, embeddings, rel_files, face_indices,
                    bboxes, det_scores, processed):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32).reshape(-1, 512),
        files=np.asarray(rel_files, dtype=np.str_),
        face_indices=np.asarray(face_indices, dtype=np.int32),
        bboxes=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
        det_scores=np.asarray(det_scores, dtype=np.float32),
        processed_files=np.asarray(processed, dtype=np.str_),
    )


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    files = [str(f) for f in data["files"]]
    # Backward compat: older caches lack processed_files -> fall back to the
    # images that had faces (face-less photos will be re-scanned once).
    if "processed_files" in data.files:
        processed = [str(f) for f in data["processed_files"]]
    else:
        processed = list(dict.fromkeys(files))
    return (
        list(data["embeddings"]),
        files,
        list(int(i) for i in data["face_indices"]),
        list(data["bboxes"]),
        list(float(s) for s in data["det_scores"]),
        processed,
    )


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def cluster_faces(embeddings: np.ndarray, min_cluster_size: int, algo: str):
    """
    Cluster L2-normalized embeddings.

    Euclidean distance on unit vectors is monotonic with cosine distance, so
    plain euclidean HDBSCAN groups faces by appearance similarity.

    Returns (labels, scores) where score is cluster-membership strength in
    [0, 1] (1.0 for DBSCAN core points).
    """
    from sklearn.preprocessing import normalize

    X = normalize(embeddings.astype(np.float64))  # defensive re-normalize

    if algo == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            print("  ! hdbscan not installed — falling back to DBSCAN.")
            algo = "dbscan"

    if algo == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=1,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X)
        scores = clusterer.probabilities_.astype(np.float32)
    else:
        from sklearn.cluster import DBSCAN

        # eps in euclidean space on unit vectors; ~0.9 ≈ cosine sim > ~0.6.
        clusterer = DBSCAN(
            eps=0.9, min_samples=max(2, min_cluster_size), metric="euclidean"
        )
        labels = clusterer.fit_predict(X)
        scores = np.where(labels >= 0, 1.0, 0.0).astype(np.float32)

    return labels.astype(np.int32), scores


def write_clusters_csv(path: Path, rel_files, face_indices, labels, scores,
                       bboxes):
    """Columns: file, face_index, cluster, score, x1, y1, x2, y2.
    The bbox (top-left x1,y1 / bottom-right x2,y2, in pixels) lets you draw a
    box around the face straight from this CSV, no embeddings.npz needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "face_index", "cluster", "score",
                         "x1", "y1", "x2", "y2"])
        for f, fi, lab, sc, bb in zip(rel_files, face_indices, labels,
                                      scores, bboxes):
            x1, y1, x2, y2 = (int(round(float(v))) for v in bb)
            writer.writerow([f, int(fi), int(lab), f"{float(sc):.4f}",
                             x1, y1, x2, y2])


def summarize(labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    n_noise = int((labels == -1).sum())
    clusters = sorted(set(int(x) for x in labels) - {-1})
    print(f"\nFound {len(clusters)} people across {len(labels)} faces "
          f"({n_noise} unclustered).")
    for c in clusters:
        print(f"  person {c:>2}: {int((labels == c).sum())} faces")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detect, embed and cluster faces into people.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=Path("data"),
                   help="Input folder of images (searched recursively).")
    p.add_argument("--out", type=Path, default=Path("out"),
                   help="Output folder for embeddings.npz and clusters.csv.")
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="HDBSCAN sensitivity: smallest group counted as a person.")
    p.add_argument("--min-det-score", type=float, default=0.0,
                   help="Drop weak detections (backs of heads / blurry / "
                        "profile) below this confidence before clustering. "
                        "Try 0.65. Such faces are moved to noise, not deleted.")
    p.add_argument("--model", default="antelopev2",
                   help="InsightFace model pack name.")
    p.add_argument("--det-size", type=int, default=640,
                   help="Detector input size (px). Larger finds smaller faces.")
    p.add_argument("--algo", choices=["hdbscan", "dbscan"], default="hdbscan",
                   help="Clustering algorithm.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse embeddings.npz and only embed new images.")
    p.add_argument("--recluster", action="store_true",
                   help="Skip detection; re-cluster existing embeddings.npz only.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    emb_path = args.out / "embeddings.npz"
    csv_path = args.out / "clusters.csv"

    if args.recluster:
        if not emb_path.exists():
            print(f"error: {emb_path} not found (nothing to re-cluster).")
            return 1
        embeddings, rel_files, face_indices, bboxes, det_scores, _ = \
            load_embeddings(emb_path)
        print(f"Loaded {len(embeddings)} embeddings from {emb_path}.")
    else:
        if not args.data.is_dir():
            print(f"error: input folder '{args.data}' does not exist.")
            return 1

        files = find_images(args.data)
        if not files:
            print(f"error: no images found under '{args.data}'.")
            return 1
        print(f"Found {len(files)} images under '{args.data}'.")

        done: set[str] = set()
        prev = ([], [], [], [], [], [])
        if args.resume and emb_path.exists():
            prev = load_embeddings(emb_path)
            done = set(prev[5])  # every image processed before, faces or not
            todo = [f for f in files
                    if str(f.relative_to(args.data)) not in done]
            print(f"Resume: {len(done)} images already processed, "
                  f"{len(todo)} new.")
            files = todo

        if not _HEIF_OK and any(f.suffix.lower() in HEIF_EXTS for f in files):
            print("  ! HEIF (.heic/.heif/.hif) files found but pillow-heif is "
                  "not installed — they will be skipped. "
                  "Run: pip install pillow-heif")

        if files:
            app = build_analyzer(args.model, args.det_size)
            new = detect_and_embed(app, args.data, files)
        else:
            new = ([], [], [], [], [], [])

        # Merge previous (resume) + new records.
        embeddings = list(prev[0]) + list(new[0])
        rel_files = list(prev[1]) + list(new[1])
        face_indices = list(prev[2]) + list(new[2])
        bboxes = list(prev[3]) + list(new[3])
        det_scores = list(prev[4]) + list(new[4])
        processed = list(prev[5]) + list(new[5])

        if not embeddings:
            print("No faces detected — nothing to cluster.")
            return 1

        save_embeddings(emb_path, embeddings, rel_files, face_indices,
                        bboxes, det_scores, processed)
        print(f"Saved {len(embeddings)} embeddings -> {emb_path}")

    X = np.asarray(embeddings, dtype=np.float32).reshape(-1, 512)

    # Optionally drop weak detections (e.g. backs of heads) before clustering.
    # They are kept in the output as noise (-1) rather than deleted.
    keep = np.asarray(det_scores, dtype=np.float32) >= args.min_det_score
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Filtering out {n_dropped} faces below det-score "
              f"{args.min_det_score} (-> noise).")

    print(f"\nClustering {int(keep.sum())} faces "
          f"(algo={args.algo}, min_cluster_size={args.min_cluster_size}) ...")
    labels = np.full(len(X), -1, dtype=np.int32)
    scores = np.zeros(len(X), dtype=np.float32)
    if keep.any():
        k_labels, k_scores = cluster_faces(X[keep], args.min_cluster_size,
                                           args.algo)
        labels[keep] = k_labels
        scores[keep] = k_scores

    write_clusters_csv(csv_path, rel_files, face_indices, labels, scores,
                       bboxes)
    summarize(labels)
    print(f"\nWrote {csv_path}")
    print(f"Done in {time.time() - t0:.1f}s. "
          f"Next: python export_clusters.py --data {args.data} "
          f"--clusters {csv_path} --out {args.out} --min-size 3 --clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
