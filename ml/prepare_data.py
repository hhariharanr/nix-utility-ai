"""Clean a raw image dataset (one folder per class) and make a stratified train/val/test split.
Usage: python prepare_data.py --raw raw --out dataset_split [--map map.json] [--drop Folder1 Folder2]
Class names must end up as: clean, dust, bird_droppings, shading, snow, physical_damage, electrical_damage (use --map to rename or merge)."""
import argparse, hashlib, json, os, random, shutil
from collections import defaultdict
from PIL import Image
ALLOWED = {"clean", "dust", "bird_droppings", "shading", "snow", "physical_damage", "electrical_damage"}
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def dhash(path):  # 64-bit difference hash: catches resized or re-compressed copies of the same photo
    im = Image.open(path).convert("L").resize((9, 8)); px = list(im.getdata())
    return "".join("1" if px[r * 9 + c] > px[r * 9 + c + 1] else "0" for r in range(8) for c in range(8))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--out", default="dataset_split")
    ap.add_argument("--map"); ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.15); a = ap.parse_args()
    mp = json.load(open(a.map)) if a.map else {}; random.seed(a.seed); items, seen, dups = defaultdict(list), set(), 0
    for folder in sorted(os.listdir(a.raw)):
        d = os.path.join(a.raw, folder)
        if not os.path.isdir(d) or folder in a.drop: continue
        cls = mp.get(folder, folder)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(EXT): continue
            p = os.path.join(d, f)
            try: h = (hashlib.md5(open(p, "rb").read()).hexdigest(), dhash(p))
            except Exception: print("skipping unreadable", p); continue
            if h[0] in seen or h[1] in seen: dups += 1; continue  # duplicates would leak between train and test
            seen.update(h); items[cls].append(p)
    bad = set(items) - ALLOWED
    if bad: raise SystemExit(f"Unknown class names {sorted(bad)}. Rename them with --map or drop them. Allowed: {sorted(ALLOWED)}")
    if os.path.exists(a.out): shutil.rmtree(a.out)
    manifest, counts = {}, {}
    for cls, files in items.items():
        random.shuffle(files); n = len(files); nt, nv = int(n * a.test), int(n * a.val)
        parts = {"test": files[:nt], "val": files[nt:nt + nv], "train": files[nt + nv:]}
        for part, fs in parts.items():
            os.makedirs(os.path.join(a.out, part, cls), exist_ok=True)
            for p in fs: shutil.copy(p, os.path.join(a.out, part, cls, os.path.basename(p))); manifest[p] = part
        counts[cls] = {k: len(v) for k, v in parts.items()}
    json.dump(manifest, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"removed {dups} duplicate images\n{'class':<20}{'train':>7}{'val':>6}{'test':>6}")
    for cls, c in sorted(counts.items()): print(f"{cls:<20}{c['train']:>7}{c['val']:>6}{c['test']:>6}")
    tot = [sum(c.values()) for c in counts.values()]
    if min(tot) < 100: print("WARNING: some classes have fewer than 100 images; expect weak results for them")
    if max(tot) > 5 * min(tot): print("WARNING: classes are imbalanced more than 5:1; training uses class weights, but consider merging or dropping tiny classes")
if __name__ == "__main__": main()
