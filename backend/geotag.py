"""Give photos demo GPS tags and a capture date (now, unless --spread-days is used) that fall on the seeded panels, so Photo Intake can be shown with a dataset that has no geotags.
Run inside the api container:  python geotag.py --in /data/demo_photos --out /data/demo_geotagged --n 60
The coordinates are synthetic. Use this for demos and tests only. Training never needs geotags."""
import argparse, csv, math, os, random
from datetime import datetime, timedelta
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
BUFFER_M, M = float(os.getenv("MATCH_BUFFER_M", "3")), 111320.0   # must match the app; M = metres per degree of latitude
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SCEN = ["on_panel", "no_gps", "off_panel", "near_panel", "between_panels"]   # default mix 70 / 10 / 8 / 7 / 5 percent
EXPECT = {"on_panel": "added automatically (or review if the classifier is not confident)", "no_gps": "review: no GPS tag on the photo",
          "off_panel": "review: no panel at this location", "near_panel": "review: location is near a panel, not on it", "between_panels": "review: several panels nearby"}

def dist_m(lon, lat, b):  # metres from a point to a panel (0 if inside)
    return math.hypot(max(b[1] - lon, 0, lon - b[2]) * M * math.cos(math.radians(lat)), max(b[3] - lat, 0, lat - b[4]) * M)

def place(a, allA, scen, rng):
    """a = (asset_id, xmin, xmax, ymin, ymax) in degrees. Returns (lon, lat) for the scenario, or None if it cannot be built for this panel."""
    _, x0, x1, y0, y1 = a; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; others = [b for b in allA if b[0] != a[0]]
    if scen == "on_panel": return rng.uniform(x0 + .2 * w, x1 - .2 * w), rng.uniform(y0 + .2 * h, y1 - .2 * h)
    if scen == "off_panel": return x1 + 0.01, y1 + 0.01
    if scen == "near_panel":                      # just outside this panel: inside its buffer, outside every other panel's
        off = 0.35 * BUFFER_M / M
        for lat in (y1 + off, y0 - off):
            if all(dist_m(cx, lat, b) > BUFFER_M for b in others): return cx, lat
        return None
    if scen == "between_panels":                  # middle of the gap to the right-hand neighbour: inside both buffers
        nb = [b for b in others if abs((b[3] + b[4]) / 2 - cy) < h / 2 and 0 < b[1] - x1 < 3 * w]
        if not nb: return None
        b = min(nb, key=lambda b: b[1]); lon = (x1 + b[1]) / 2
        return (lon, cy) if dist_m(lon, cy, a) <= BUFFER_M and dist_m(lon, cy, b) <= BUFFER_M else None

def dms(v):
    v = abs(v); d = int(v); m = int((v - d) * 60); s = ((v - d) * 60 - m) * 60
    return (IFDRational(d, 1), IFDRational(m, 1), IFDRational(int(round(s * 10000)), 10000))

def save(src, dst, when, pos):
    img = Image.open(src).convert("RGB"); ex = img.getexif()
    ex.get_ifd(0x8769)[36867] = when.strftime("%Y:%m:%d %H:%M:%S")                  # DateTimeOriginal
    if pos:
        lon, lat = pos; g = ex.get_ifd(0x8825)
        g[1], g[2], g[3], g[4] = "N" if lat >= 0 else "S", dms(lat), "E" if lon >= 0 else "W", dms(lon)
    img.save(dst, "JPEG", quality=90, exif=ex)

def run(allA, src, out, n=60, mix=(70, 10, 8, 7, 5), seed=1, spread_days=0.0):
    rng = random.Random(seed); os.makedirs(out, exist_ok=True)
    files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs if f.lower().endswith(EXT)]
    rng.shuffle(files); files = files[:n]
    plan = [s for s, p in zip(SCEN, mix) for _ in range(round(len(files) * p / 100))]
    plan = (plan + ["on_panel"] * len(files))[:len(files)]; rng.shuffle(plan); rows = []
    for k, (f, scen) in enumerate(zip(files, plan)):
        a, pos = rng.choice(allA), None
        for _ in range(20):                                                         # some panels have no right-hand neighbour; try another
            pos = None if scen == "no_gps" else place(a, allA, scen, rng)
            if scen == "no_gps" or pos: break
            a = rng.choice(allA)
        if scen != "no_gps" and not pos: scen, pos = "on_panel", place(a, allA, "on_panel", rng)
        name = f"photo_{k:03d}.jpg"; save(f, os.path.join(out, name), datetime.now() - timedelta(days=rng.uniform(0, spread_days)), pos)
        rows.append([name, scen, "" if scen in ("no_gps", "off_panel") else a[0], pos[1] if pos else "", pos[0] if pos else "", EXPECT[scen]])
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["file", "scenario", "nearest_asset", "lat", "lon", "expected_in_photo_intake"]] + rows)
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spread-days", type=float, default=0.0, help="stamp photos with random dates up to this many days ago (default 0 = now, so they are the latest observation for their panel)")
    ap.add_argument("--mix", default="70,10,8,7,5", help="percent for: on_panel,no_gps,off_panel,near_panel,between_panels"); a = ap.parse_args()
    import psycopg2
    cur = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://nix:nix@db:5432/nix")).cursor()
    cur.execute("SELECT asset_id,ST_XMin(geom),ST_XMax(geom),ST_YMin(geom),ST_YMax(geom) FROM assets ORDER BY asset_id"); allA = cur.fetchall()
    if not allA: raise SystemExit("No panels in the database yet. Run seed.py first.")
    rows = run(allA, a.src, a.out, a.n, tuple(float(x) for x in a.mix.split(",")), a.seed, a.spread_days)
    print(f"wrote {len(rows)} photos and manifest.csv to {a.out}")
    for s in SCEN: print(f"  {s:<16}{sum(1 for r in rows if r[1] == s)}")
if __name__ == "__main__": main()
