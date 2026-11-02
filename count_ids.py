import argparse
import collections
import json
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def count_cached_ids(cache_path="cache/players.json"):
    if not os.path.exists(cache_path):
        print(f"[info] Cache file '{cache_path}' does not exist.")
        print("To generate it, run: python detection/players.py")
        return

    with open(cache_path) as f:
        cache = json.load(f)

    frames_seen = collections.Counter()
    for dets in cache:
        for d in dets:
            frames_seen[d["id"]] += 1

    print("total frames in cache:", len(cache))
    for pid, n in sorted(frames_seen.items()):
        print(f"ID {pid}: seen in {n} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count frames per player track ID from cache")
    parser.add_argument("--cache", default="cache/players.json", help="path to players.json cache")
    args = parser.parse_args()
    count_cached_ids(args.cache)
