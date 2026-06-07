#!/usr/bin/env python3
"""
visualize.py — see which person_XX folder is who.

Two views, both built from the clusters + bounding boxes in clusters.csv
(produced by cluster.py — no re-detection):

  key       a labeled grid with one clear face per person
              -> out/people_overview.jpg
  annotate  draw labeled person_XX boxes on a photo
              -> out/annotated/<photo>_annotated.jpg

The person numbering matches export_clusters.py exactly (largest group first),
so person_03 here is the same person_03 folder on disk.

Run:
  python visualize.py key      --data all_pictures --out out
  python visualize.py annotate --data all_pictures --out out               # auto-pick fullest group photo
  python visualize.py annotate --data all_pictures --out out photo.jpg     # one specific picture
  python visualize.py annotate --data all_pictures --out out --top 5       # the 5 fullest group photos
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cluster import load_image  # HEIF-aware loader


# --------------------------------------------------------------------------- #
# Shared loading
# --------------------------------------------------------------------------- #
def read_rows(csv_path: Path):
    """Rows of (file, face_index, cluster, score, bbox|None). bbox comes from
    the x1,y1,x2,y2 columns when present (newer clusters.csv)."""
    rows = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        has_xy = {"x1", "y1", "x2", "y2"}.issubset(reader.fieldnames or [])
        for r in reader:
            bbox = (np.array([float(r["x1"]), float(r["y1"]),
                              float(r["x2"]), float(r["y2"])], np.float32)
                    if has_xy else None)
            rows.append((r["file"], int(r["face_index"]),
                         int(r["cluster"]), float(r["score"]), bbox))
    return rows


def resolve_bboxes(rows, emb_npz: Path):
    """Face bboxes keyed by (file, face_index). Prefer the coordinates already
    in clusters.csv; fall back to embeddings.npz for older CSVs."""
    d = {(f, fi): bb for f, fi, _, _, bb in rows if bb is not None}
    if d:
        return d
    if emb_npz.exists():
        data = np.load(emb_npz, allow_pickle=False)
        files = [str(f) for f in data["files"]]
        fidx = [int(i) for i in data["face_indices"]]
        return {(files[k], fidx[k]): data["bboxes"][k]
                for k in range(len(files))}
    return {}


def person_labels(rows):
    """cluster id -> person_XX, matching export_clusters.py ordering."""
    imgs = defaultdict(set)
    for f, _, c, _, _ in rows:
        if c != -1:
            imgs[c].add(f)
    order = sorted(imgs, key=lambda c: (-len(imgs[c]), c))
    width = max(2, len(str(max(len(order) - 1, 0))))
    label = {c: f"person_{i:0{width}d}" for i, c in enumerate(order)}
    return label, order, imgs


def _check_inputs(data: Path, clusters_csv: Path):
    if not clusters_csv.exists():
        print(f"error: {clusters_csv} not found. Run cluster.py first.")
        return False
    if not data.is_dir():
        print(f"error: input folder '{data}' does not exist.")
        return False
    return True


def _to_rel(arg: str, data: Path):
    """Map a user-supplied image path to its path relative to --data (the key
    used in clusters.csv). Accepts a path relative to --data or an absolute
    path inside it."""
    p = Path(arg)
    if (data / p).exists():
        return str(p)
    if p.exists():
        try:
            return str(p.resolve().relative_to(data.resolve()))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# key: labeled face grid
# --------------------------------------------------------------------------- #
def crop_face(img, bbox, pad=0.45):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * (1 + pad) / 2
    a, b = int(max(0, cx - half)), int(min(w, cx + half))
    c, d = int(max(0, cy - half)), int(min(h, cy + half))
    if b <= a or d <= c:
        return None
    return img[c:d, a:b]


def choose_representative(faces, bboxes):
    """faces: list of (file, face_index, score). Prefer a confident AND large
    (clear, frontal) face: top half by membership score, then biggest box."""
    scored = [f for f in faces if (f[0], f[1]) in bboxes]
    if not scored:
        return None
    scored.sort(key=lambda f: f[2], reverse=True)
    top = scored[:max(1, len(scored) // 2)]

    def area(f):
        x1, y1, x2, y2 = bboxes[(f[0], f[1])]
        return (x2 - x1) * (y2 - y1)

    return max(top, key=area)


def _label_cell(thumb, lines, cell_w, thumb_h, footer_h):
    import cv2

    cell = np.full((thumb_h + footer_h, cell_w, 3), 30, np.uint8)
    th, tw = thumb.shape[:2]
    x0 = (cell_w - tw) // 2
    cell[0:th, x0:x0 + tw] = thumb
    y = thumb_h + 22
    for i, text in enumerate(lines):
        scale = 0.6 if i == 0 else 0.42
        color = (255, 255, 255) if i == 0 else (170, 170, 170)
        cv2.putText(cell, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
        y += 20
    return cell


def cmd_key(args) -> int:
    import cv2

    clusters_csv = args.clusters or (args.out / "clusters.csv")
    emb_npz = args.embeddings or (args.out / "embeddings.npz")
    out_file = args.out_file or (args.out / "people_overview.jpg")
    if not _check_inputs(args.data, clusters_csv):
        return 1

    rows = read_rows(clusters_csv)
    bboxes = resolve_bboxes(rows, emb_npz)
    if not bboxes:
        print("error: no face coordinates available. Re-run cluster.py "
              "(adds x/y columns) or pass --embeddings.")
        return 1
    label, order, imgs = person_labels(rows)

    faces_by_cluster: dict[int, list] = defaultdict(list)
    for f, fi, c, sc, _ in rows:
        if c != -1:
            faces_by_cluster[c].append((f, fi, sc))

    people = [c for c in order if len(imgs[c]) >= args.min_size]
    if not people:
        print("No people to show (try a smaller --min-size).")
        return 1

    cells, mapping = [], []
    for c in people:
        rep = choose_representative(faces_by_cluster[c], bboxes)
        n_imgs, n_faces = len(imgs[c]), len(faces_by_cluster[c])
        mapping.append((label[c], c, n_imgs, n_faces, rep[0] if rep else "—"))
        if rep is None:
            continue
        img = load_image(args.data / rep[0])
        crop = crop_face(img, bboxes[(rep[0], rep[1])]) if img is not None else None
        if crop is None or crop.size == 0:
            thumb = np.full((args.thumb, args.thumb, 3), 60, np.uint8)
        else:
            thumb = cv2.resize(crop, (args.thumb, args.thumb))
        cells.append(_label_cell(
            thumb, [label[c], f"{n_imgs} photos, {n_faces} faces"],
            args.thumb, args.thumb, 50))

    cols = args.cols or max(1, round(math.sqrt(len(cells))))
    cols = min(cols, len(cells))
    rows_n = math.ceil(len(cells) / cols)
    gap = 10
    cell_h, cell_w = cells[0].shape[:2]
    canvas = np.full(
        (rows_n * cell_h + (rows_n + 1) * gap,
         cols * cell_w + (cols + 1) * gap, 3), 20, np.uint8)
    for idx, cell in enumerate(cells):
        r, col = divmod(idx, cols)
        y, x = gap + r * (cell_h + gap), gap + col * (cell_w + gap)
        canvas[y:y + cell_h, x:x + cell_w] = cell

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)

    print(f"\nPerson key -> {out_file}\n")
    print(f"{'folder':<11}{'cluster':>8}{'photos':>8}{'faces':>7}  example")
    for lab, c, n_imgs, n_faces, ex in mapping:
        print(f"{lab:<11}{c:>8}{n_imgs:>8}{n_faces:>7}  {ex}")
    return 0


# --------------------------------------------------------------------------- #
# annotate: labeled boxes on a photo
# --------------------------------------------------------------------------- #
def _color_for(cluster: int):
    if cluster < 0:
        return (160, 160, 160)
    import colorsys

    h = (cluster * 0.61803398875) % 1.0           # golden-ratio hue spacing
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _annotate_image(img, faces, label):
    """faces: list of (face_index, bbox, cluster)."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, round(w / 600))
    scale = max(0.6, w / 1600)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, bbox, cluster in faces:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = _color_for(cluster)
        text = label.get(cluster, "?") if cluster != -1 else "?"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        ly = max(0, y1 - 6)
        ty = ly - th - base
        if ty < 0:                                # no room above box
            ly, ty = y2 + th + base + 6, y2 + 6
        cv2.rectangle(out, (x1, ty), (x1 + tw + 8, ly), color, -1)
        cv2.putText(out, text, (x1 + 4, ly - base),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def cmd_annotate(args) -> int:
    import cv2

    clusters_csv = args.clusters or (args.out / "clusters.csv")
    emb_npz = args.embeddings or (args.out / "embeddings.npz")
    if not _check_inputs(args.data, clusters_csv):
        return 1

    rows = read_rows(clusters_csv)
    bboxes = resolve_bboxes(rows, emb_npz)
    if not bboxes:
        print("error: no face coordinates available. Re-run cluster.py "
              "(adds x/y columns) or pass --embeddings.")
        return 1
    label, _, _ = person_labels(rows)

    faces_by_file: dict[str, list] = defaultdict(list)
    for f, fi, c, _, _ in rows:
        if (f, fi) in bboxes:
            faces_by_file[f].append((fi, bboxes[(f, fi)], c))

    if args.image:
        rel = _to_rel(args.image, args.data)
        if rel is None:
            print(f"error: '{args.image}' not found inside --data "
                  f"({args.data}).")
            return 1
        targets = [rel]
    else:
        def rank(f):
            people = {c for _, _, c in faces_by_file[f] if c != -1}
            return (len(people), len(faces_by_file[f]))
        targets = sorted(faces_by_file, key=rank, reverse=True)[:max(1, args.top)]

    out_dir = args.out / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in targets:
        img = load_image(args.data / rel)
        if img is None:
            print(f"  ! could not read {rel}")
            continue
        faces = faces_by_file.get(rel, [])
        names = sorted({label.get(c, "?") if c != -1 else "?"
                        for _, _, c in faces})
        dst = out_dir / (Path(rel).stem + "_annotated.jpg")
        cv2.imwrite(str(dst), _annotate_image(img, faces, label))
        note = ", ".join(names) if faces else "no faces"
        print(f"{rel}: {len(faces)} faces ({note}) -> {dst}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=Path("data"),
                        help="Input photo folder the images were clustered from.")
    common.add_argument("--out", type=Path, default=Path("out"),
                        help="Output folder (also where clusters.csv lives).")
    common.add_argument("--clusters", type=Path, default=None,
                        help="clusters.csv (default: <out>/clusters.csv)")
    common.add_argument("--embeddings", type=Path, default=None,
                        help="embeddings.npz fallback (default: <out>/embeddings.npz)")

    p = argparse.ArgumentParser(
        description="Visualize who each person_XX folder is.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("key", parents=[common],
                       help="Labeled grid, one face per person.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    k.add_argument("--min-size", type=int, default=1,
                   help="Match the value used in export_clusters.py.")
    k.add_argument("--thumb", type=int, default=220, help="Face thumb size (px).")
    k.add_argument("--cols", type=int, default=0,
                   help="Grid columns (0 = auto, roughly square).")
    k.add_argument("--out-file", type=Path, default=None,
                   help="Output image (default: <out>/people_overview.jpg).")
    k.set_defaults(func=cmd_key)

    a = sub.add_parser("annotate", parents=[common],
                       help="Draw labeled boxes on a photo.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("image", nargs="?", default=None,
                   help="A specific photo to annotate (path relative to --data, "
                        "or absolute). Omit to auto-pick the fullest group photo.")
    a.add_argument("--top", type=int, default=1,
                   help="When no image is given, annotate the N fullest photos.")
    a.set_defaults(func=cmd_annotate)

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
