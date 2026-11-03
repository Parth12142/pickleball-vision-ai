"""
build_datasets.py - Build clean, leakage-free YOLO datasets for the base model.

Sources (raw Roboflow exports, see datasets/raw/):
  v13  : pickleball.v13-bickleball_player  classes [Player, ball]  (fully labelled, 512x512)
  v1i  : Pickleball bal detection v1i       8 messy classes, ball-only labels on broadcast frames

Outputs (images are symlinked, labels rewritten):
  datasets/players/{train,val,test}/{images,labels}   class 0 = player   (v13 only)
  datasets/ball/{train,val,test}/{images,labels}      class 0 = ball     (v13 + v1i)
  datasets/<name>/data.yaml, datasets/<name>/split_report.json

Leakage control:
  1. Roboflow augmentation copies share the stem before ".rf." -> same group.
  2. v1i frames from the same source video (prefix before "_mp4-") -> same group.
  3. Perceptual dHash (64-bit) near-duplicates (Hamming <= DHASH_THRESH) -> same group (union-find).
  Groups are split 80/10/10 with a deterministic hash; val/test keep ONE copy per source
  (no augmented duplicates in evaluation).

Usage:
  python -m pbbase.data.build_datasets
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "datasets", "raw")
OUT = os.path.join(ROOT, "datasets")
DHASH_THRESH = 4

# raw class id -> target name, per source
V13_MAP = {0: "player", 1: "ball"}
V1I_MAP = {0: "ball", 1: "ball", 2: "ball", 3: "ball", 6: "ball", 4: "player", 5: "player"}  # 7 'objects' dropped
LIBERIN_MAP = {0: "ball"}


def dhash(path, size=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    return int("".join("1" if v else "0" for v in diff.flatten()), 2)


def read_labels(label_path, cmap):
    """Return list of (name, cx, cy, w, h). Polygon rows are converted to boxes."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path, encoding="utf-8", errors="ignore"):
        t = line.split()
        if len(t) < 5:
            continue
        try:
            c = int(float(t[0]))
            vals = list(map(float, t[1:]))
        except (ValueError, TypeError):
            continue
        if c not in cmap:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:  # polygon -> bbox
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        out.append((cmap[c], cx, cy, w, h))
    return out


def source_key(fname):
    return fname.split(".rf.")[0]


def video_key(fname):
    stem = source_key(fname)
    m = re.split(r"_mp4-\d+", stem)
    return m[0] if len(m) > 1 else stem


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def collect(src_name, cmap, use_video_groups):
    items = []
    for split in ("train", "valid", "test"):
        for img in sorted(glob.glob(os.path.join(RAW, src_name, split, "images", "*"))):
            fname = os.path.basename(img)
            lbl = os.path.join(RAW, src_name, split, "labels", os.path.splitext(fname)[0] + ".txt")
            items.append({
                "src": src_name,
                "orig_split": split,
                "img": img,
                "fname": fname,
                "labels": read_labels(lbl, cmap),
                "source": f"{src_name}:{source_key(fname)}",
                "video": f"{src_name}:{video_key(fname)}" if use_video_groups else None,
            })
    return items


def group_items(items):
    uf = UnionFind()
    for it in items:
        uf.find(it["source"])
        if it["video"]:
            uf.union(it["source"], it["video"])
    # dHash near-duplicate grouping (one hash per source is enough)
    seen = {}
    for it in items:
        if it["source"] in seen:
            continue
        h = dhash(it["img"])
        if h is not None:
            seen[it["source"]] = h
    keys = list(seen.keys())
    hashes = np.array([seen[k] for k in keys], dtype=np.uint64)
    # bucket by 16-bit chunks (pigeonhole: <=4 differing bits -> at least one 16-bit chunk equal)
    for shift in (0, 16, 32, 48):
        buckets = collections.defaultdict(list)
        for k, h in zip(keys, hashes):
            buckets[(int(h) >> shift) & 0xFFFF].append((k, int(h)))
        for members in buckets.values():
            if len(members) < 2 or len(members) > 400:
                continue
            for i in range(len(members)):
                ki, hi = members[i]
                for j in range(i + 1, len(members)):
                    kj, hj = members[j]
                    if bin(hi ^ hj).count("1") <= DHASH_THRESH:
                        uf.union(ki, kj)
    for it in items:
        it["group"] = uf.find(it["source"])
    return items


def assign_split(group, val_pct=10, test_pct=10):
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def write_dataset(name, items, keep_class, min_items_per_img=0):
    base = os.path.join(OUT, name)
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = os.path.join(base, split, sub)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
    stats = collections.defaultdict(lambda: collections.Counter())
    used_eval_sources = set()
    for it in items:
        split = assign_split(it["group"])
        if split != "train":
            if it["source"] in used_eval_sources:
                continue  # only one copy of each source in val/test
            used_eval_sources.add(it["source"])
        labels = [lab for lab in it["labels"] if lab[0] == keep_class]
        if len(labels) < min_items_per_img:
            continue
        uid = hashlib.md5(it["img"].encode()).hexdigest()[:10]
        stem = f"{it['src']}_{uid}"
        ext = os.path.splitext(it["fname"])[1]
        dst_img = os.path.join(base, split, "images", stem + ext)
        os.symlink(os.path.realpath(it["img"]), dst_img)
        with open(os.path.join(base, split, "labels", stem + ".txt"), "w") as f:
            for _, cx, cy, w, h in labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        stats[split]["images"] += 1
        stats[split][f"images_{it['src']}"] += 1
        stats[split]["boxes"] += len(labels)
        stats[split]["empty_images"] += int(len(labels) == 0)
    with open(os.path.join(base, "data.yaml"), "w") as f:
        f.write(f"path: {base}\ntrain: train/images\nval: val/images\ntest: test/images\n\nnc: 1\nnames: ['{keep_class}']\n")
    report = {s: dict(c) for s, c in stats.items()}
    with open(os.path.join(base, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    v13 = collect("v13", V13_MAP, use_video_groups=False)
    v1i = collect("v1i", V1I_MAP, use_video_groups=True)
    liberin = collect("liberin", LIBERIN_MAP, use_video_groups=False)
    print(f"collected v13={len(v13)} v1i={len(v1i)} liberin={len(liberin)}")

    # Players: v13 only (v1i and liberin do not have complete player annotations -> prevents teaching person=background)
    v13 = group_items(v13)
    print("players:", json.dumps(write_dataset("players", v13, "player"), indent=1))

    # Ball: v13 + v1i + liberin grouped jointly with perceptual dHash deduplication
    allb = group_items(v13 + v1i + liberin)
    print("ball:", json.dumps(write_dataset("ball", allb, "ball"), indent=1))
    ngroups = len({it["group"] for it in allb})
    print(f"ball groups: {ngroups}")


if __name__ == "__main__":
    main()
