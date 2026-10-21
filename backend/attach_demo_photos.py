"""Attach demo photos to the synthetic (seeded) observations that have an issue, so the map popup shows a picture for them.
Each observation gets a random photo from the matching class folder of a held-out image set (default: the test split in ml/dataset_split/test),
and, if the trained model is available, a Grad-CAM heat map beside it. Classes with no folder (for example shading) are skipped.
Run inside the api container:  python attach_demo_photos.py          (add --include-clean to also give clean panels a photo, --redo to replace existing ones)
The photos are real dataset images attached to synthetic records. Say so in the demo."""
import argparse, os, random
from PIL import Image
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def run(conn, src, dest, include_clean=False, redo=False, seed=1, cam=True):
    rng = random.Random(seed)
    pool = {c: [os.path.join(src, c, f) for f in sorted(os.listdir(os.path.join(src, c))) if f.lower().endswith(EXT)]
            for c in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, c))}
    if not any(pool.values()): raise SystemExit(f"No class folders with images found in {src}")
    cur = conn.cursor()
    cur.execute("SELECT obs_id, issue_class FROM observations WHERE source='seed'" + ("" if redo else " AND image_path IS NULL") + ("" if include_clean else " AND issue_class<>'clean'") + " ORDER BY obs_id")
    rows = cur.fetchall(); os.makedirs(dest, exist_ok=True); made, skipped = [], {}
    for obs_id, cls in rows:
        files = pool.get(cls)
        if not files: skipped[cls] = skipped.get(cls, 0) + 1; continue
        out = os.path.join(dest, f"obs_{obs_id}.jpg")
        Image.open(rng.choice(files)).convert("RGB").save(out, "JPEG", quality=90)  # normalise png/webp to jpg
        cur.execute("UPDATE observations SET image_path=%s WHERE obs_id=%s", (out, obs_id)); made.append((cls, out))
    conn.commit(); heat = 0
    if cam and made:
        try:
            import vision
            if vision.available():
                for _, out in made: vision.predict(out); heat += 1   # writes obs_<id>_cam.jpg next to the photo
            else: print("No trained model found, so no heat maps were made.")
        except Exception as e: print("Heat maps skipped:", e)
    return made, skipped, heat

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/app/ml/dataset_split/test"); ap.add_argument("--dest", default="/data/demo_seed")
    ap.add_argument("--include-clean", action="store_true"); ap.add_argument("--redo", action="store_true"); ap.add_argument("--no-cam", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); import psycopg2
    conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://nix:nix@db:5432/nix"))
    made, skipped, heat = run(conn, a.src, a.dest, a.include_clean, a.redo, a.seed, not a.no_cam)
    by = {}
    for c, _ in made: by[c] = by.get(c, 0) + 1
    print(f"attached {len(made)} photos ({heat} with heat maps):", by)
    if skipped: print("no photos available for these classes, left without a photo:", skipped)
if __name__ == "__main__": main()
