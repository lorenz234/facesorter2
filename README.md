# Face Sorter (clustering version)

Automatically sort a pile of photos into **one folder per person** — no
reference photos, no calibration. It detects every face, computes a face
embedding, groups similar faces together, and copies each photo into the
folder(s) of the people it contains.

```
cluster.py          detect faces → embed → cluster   (writes clusters.csv)
export_clusters.py  clusters.csv → out/clusters/person_00, person_01, ...
visualize.py key       clusters.csv → out/people_overview.jpg (a labeled face key)
visualize.py annotate  draw labeled person_XX boxes on a group photo
```

## Quick start (one command, no conda)

Put your photos in a folder called `all_pictures/` (subfolders are fine), then:

```bash
./run.sh
```

That's it. On the first run it creates a local virtualenv, installs everything,
and downloads the InsightFace **antelopev2** model (~350 MB, online once). Then
it detects → clusters → exports one folder per person → builds the visual keys.

Point it at a different folder or tweak settings with environment variables:

```bash
DATA=my_photos ./run.sh
MIN_CLUSTER_SIZE=5 MIN_DET_SCORE=0.6 ./run.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `DATA` | `all_pictures` | input photo folder |
| `OUT` | `out` | output folder |
| `MIN_CLUSTER_SIZE` | `8` | smallest group counted as a person |
| `MIN_DET_SCORE` | `0.7` | drop weak detections (backs of heads / blurry) |
| `MIN_SIZE` | `3` | skip people seen in fewer than N photos |

## Running the steps yourself

`run.sh` just chains the four scripts; you can run them individually for more
control. Requires Python 3.10–3.12 (`python3.12` on this machine):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 1) detect + cluster
python cluster.py --data all_pictures --out out --min-cluster-size 8 --min-det-score 0.7

# 2) export one folder per person
python export_clusters.py --data all_pictures --clusters out/clusters.csv \
    --out out --min-size 3 --clear
```

Results land in `out/clusters/`:

```
out/clusters/
  person_00/   ← biggest group of the same face
  person_01/
  ...
  noise/       ← photos whose detected face(s) didn't fit any group
  no_faces/    ← photos with no detected face at all (scenery, objects, …)
```

`person_00` is always the largest group. A photo with several people is copied
into each of their folders. `no_faces/` collects every image that had no face
detected — rename it with `--no-faces-dir nature` or skip it with
`--no-faces-dir ''`.

### Which folder is which person?

Generate a labeled "face key" — one representative face per person in a single
image — so you can match folders to faces at a glance:

```bash
python visualize.py key --data all_pictures --out out
```

This writes `out/people_overview.jpg` (a grid of labeled faces) and prints a
`folder → photos → example file` table. The numbering matches the exported
folders exactly. If a person's representative crop is unclear (e.g. they were
looking away in that shot), just open that `person_XX/` folder for more photos.

You can also draw the labels onto a **real group photo** — every face boxed and
named — which is handy for seeing everyone together:

```bash
python visualize.py annotate --data all_pictures --out out                 # auto-picks the photo with the most people
python visualize.py annotate --data all_pictures --out out photo.jpg       # one specific picture
python visualize.py annotate --data all_pictures --out out --top 5         # the 5 fullest group photos
```

Results go to `out/annotated/`. Faces that didn't cluster are boxed in grey as `?`.

## Tuning

If you get **too many tiny groups** (the same person split up), raise the
cluster size. If different people get **merged together**, lower it.

```bash
python cluster.py --data all_pictures --out out --min-cluster-size 8
```

**Phantom "person" of backs-of-heads / blurry faces?** The detector sometimes
fires weakly on the back of a head or a distant/profile face. Those low-quality
detections can clump into a fake person. Drop them with `--min-det-score` (a
detection-confidence floor — try `0.7`); they move to `noise/` instead:

```bash
python cluster.py --out out --recluster --min-cluster-size 8 --min-det-score 0.7
```

Common options:

| Option | Script | Meaning |
|---|---|---|
| `--data` | most | Input photo folder (searched recursively) |
| `--out` | all | Output folder |
| `--min-cluster-size` | cluster.py | Smallest group counted as a person (default 5) |
| `--min-det-score` | cluster.py | Drop weak detections below this confidence (try 0.7) |
| `--det-size` | cluster.py | Detector size in px; raise to catch small/distant faces |
| `--resume` | cluster.py | Re-run after adding photos; only embeds the new ones |
| `--recluster` | cluster.py | Re-cluster existing embeddings without re-detecting |
| `--algo` | cluster.py | `hdbscan` (default) or `dbscan` |
| `--min-size` | export_clusters.py | Skip people seen in fewer than N photos |
| `--clear` | export_clusters.py | Wipe `out/clusters/` before exporting |
| `--no-noise` | export_clusters.py | Don't export the `noise/` folder |
| `--no-faces-dir` | export_clusters.py | Folder for face-less photos (default `no_faces`; `''` to skip) |
| `image` / `--top` | visualize.py annotate | A specific photo to label / auto-pick the N fullest |

### Re-cluster without re-detecting

Detection is the slow part. After the first run the embeddings are cached in
`out/embeddings.npz`, so you can try different settings instantly:

```bash
python cluster.py --out out --recluster --min-cluster-size 10
```

## How it works

1. **Detect** — InsightFace finds every face in every image.
2. **Embed** — each face becomes a 512-dim, L2-normalized vector (antelopev2).
3. **Cluster** — HDBSCAN groups faces by similarity (cosine ≈ euclidean on
   unit vectors). Faces that match nothing become "noise".
4. **Export** — images are copied into per-person folders.

`out/embeddings.npz` holds the raw embeddings + metadata. `out/clusters.csv`
holds one row per face:

```
file, face_index, cluster, score, x1, y1, x2, y2
```

`score` is cluster-membership strength; `x1,y1,x2,y2` is the face's pixel
bounding box (top-left → bottom-right). Those coordinates make the CSV
self-contained — you can draw your own boxes straight from it, and
`visualize.py` uses them directly (no `embeddings.npz` needed).

## Notes

- Supported images: `jpg jpeg png bmp webp tif tiff` plus HEIF stills
  `heic heif hif` (iPhone and mirrorless cameras like Fuji/Canon), decoded via
  `pillow-heif` (included).
- **Videos** (`.mov`, `.mp4`, …) are ignored — only still images are clustered.
- Everything runs on CPU. A large library takes a while on the first pass
  (roughly 1–2 s per photo); use `--resume` when you add more photos later.
- Originals are never modified — images are **copied**, not moved.
