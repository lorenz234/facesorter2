#!/usr/bin/env python3
"""
export_clusters.py — turn clusters.csv into one folder per person.

Reads the clusters.csv produced by cluster.py and copies each source image
into out/clusters/person_00/, person_01/, ... (largest group first). Images
containing several people are copied into each of those people's folders.
Unclustered faces (cluster -1) go to out/clusters/noise/, and images with no
detected face at all go to out/clusters/no_faces/.

Run:
  python export_clusters.py --data all_pictures --clusters out/clusters.csv \
      --out out --min-size 3 --clear
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from cluster import find_images  # same image scan used during clustering


def read_clusters(csv_path: Path):
    """Return {cluster_id: set(relative_file_paths)} from clusters.csv."""
    by_cluster: dict[int, set[str]] = defaultdict(set)
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"file", "cluster"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} missing columns {required - set(reader.fieldnames or [])}"
            )
        for row in reader:
            by_cluster[int(row["cluster"])].add(row["file"])
    return by_cluster


def unique_dest(dest_dir: Path, src: Path) -> Path:
    """Pick a destination path, disambiguating only on real name collisions."""
    candidate = dest_dir / src.name
    if not candidate.exists():
        return candidate
    # Same file already copied here -> reuse it.
    if candidate.stat().st_size == src.stat().st_size:
        return candidate
    stem, suffix = src.stem, src.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_group(data_dir: Path, files: set[str], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in sorted(files):
        src = data_dir / rel
        if not src.exists():
            print(f"  ! missing source: {src}")
            continue
        dst = unique_dest(dest_dir, src)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue  # already there
        shutil.copy2(src, dst)
        copied += 1
    return copied


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Copy images into one folder per person from clusters.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=Path("data"),
                   help="Input folder the images were clustered from.")
    p.add_argument("--clusters", type=Path, default=Path("out/clusters.csv"),
                   help="Path to clusters.csv.")
    p.add_argument("--out", type=Path, default=Path("out"),
                   help="Output folder (clusters/ is created inside it).")
    p.add_argument("--min-size", type=int, default=1,
                   help="Skip clusters with fewer than N distinct images.")
    p.add_argument("--clear", action="store_true",
                   help="Delete out/clusters before exporting.")
    p.add_argument("--no-noise", action="store_true",
                   help="Do not export the unclustered 'noise' faces.")
    p.add_argument("--no-faces-dir", default="no_faces",
                   help="Folder name for images with no detected face "
                        "(e.g. nature/scenery). Set to '' to skip them.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.clusters.exists():
        print(f"error: clusters file '{args.clusters}' not found. "
              f"Run cluster.py first.")
        return 1
    if not args.data.is_dir():
        print(f"error: input folder '{args.data}' does not exist.")
        return 1

    by_cluster = read_clusters(args.clusters)
    clusters_root = args.out / "clusters"

    if args.clear and clusters_root.exists():
        print(f"Clearing {clusters_root} ...")
        shutil.rmtree(clusters_root)

    # People = non-noise clusters, largest first -> person_00 is biggest group.
    people = sorted(
        (c for c in by_cluster if c != -1),
        key=lambda c: (-len(by_cluster[c]), c),
    )

    total_copied = 0
    exported = 0
    skipped = 0
    width = max(2, len(str(max(len(people) - 1, 0))))
    for i, c in enumerate(people):
        files = by_cluster[c]
        if len(files) < args.min_size:
            skipped += 1
            continue
        dest = clusters_root / f"person_{i:0{width}d}"
        n = copy_group(args.data, files, dest)
        total_copied += n
        exported += 1
        print(f"person_{i:0{width}d}: {len(files)} images "
              f"(cluster {c}) -> {n} copied")

    if -1 in by_cluster and not args.no_noise:
        n = copy_group(args.data, by_cluster[-1], clusters_root / "noise")
        total_copied += n
        print(f"noise: {len(by_cluster[-1])} images -> {n} copied")

    # Images with no detected face: every image under --data that never shows
    # up in clusters.csv (no person folder, no noise either).
    if args.no_faces_dir:
        with_faces = set().union(*by_cluster.values()) if by_cluster else set()
        all_images = {str(p.relative_to(args.data))
                      for p in find_images(args.data)}
        face_less = all_images - with_faces
        if face_less:
            dest = clusters_root / args.no_faces_dir
            n = copy_group(args.data, face_less, dest)
            total_copied += n
            print(f"{args.no_faces_dir}: {len(face_less)} images "
                  f"(no faces) -> {n} copied")

    print(f"\nExported {exported} people "
          f"({skipped} clusters skipped for < {args.min_size} images), "
          f"{total_copied} files copied into {clusters_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
