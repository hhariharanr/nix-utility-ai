# NiX Utility AI (prototype)
1. Put your model details in `llm.ini` (base_url, token, model; any OpenAI-compatible endpoint), then `docker compose up --build`. Without a configured model, recommendations use rules and chat gives basic data answers.
2. `docker compose exec api python seed.py`  (64 synthetic panels + 250 observations; replace `assets` with your real polygons via ogr2ogr/QGIS)
3. Open http://localhost:8000
- Classifier: runs as a stub until you put a trained Keras model at `data/model.keras` (edit CLASSES env to match its output order).
- LLM: all settings live in `llm.ini` (edits apply on the next request). The client code is `llm_chat()` in backend/main.py.
- Risk weights, loss table and defaults are assumptions at the top of backend/risk.py.

## Photo intake
Use the Photo Intake tab (or `curl -F files=@a.jpg -F files=@b.jpg http://localhost:8000/ingest`). A photo is added automatically only if it has GPS, falls inside exactly one panel and the classifier is confident; otherwise it waits in the review queue.

## Login
Set the shared username and password in `auth.ini` (the app refuses logins until the password is changed). Set `cookie_secure = true` once served over HTTPS.

## Image analysis (EfficientNetB0)
Train on your machine or Colab (a GPU helps), then the app picks the model up automatically.
```
cd ml && pip install -r requirements.txt
python prepare_data.py --raw raw --out dataset_split --map map.json   # dedupe, rename classes, split 70/15/15 (see map.example.json)
python train.py --data dataset_split --out ../data                    # writes data/model.keras + classes.json + report + confusion matrix
python evaluate.py --data realworld --model ../data                   # your own photos, same folder-per-class layout
docker compose up --build                                             # without data/model.keras the app uses a stub classifier
```
Severity comes from the Grad-CAM coverage (share of the photo the model relied on): under 25% low, under 55% medium, otherwise high. These cut-offs are assumptions to tune.

## Demo photos without geotags
Training does not need geotags. Photo Intake does, so for demos give a folder of photos synthetic GPS tags that land on the seeded panels (use held-out test images, not training ones):
```
docker compose up --build                 # once, so the script is inside the image
docker compose exec api python geotag.py --in /data/demo_photos --out /data/demo_geotagged --n 60
```
Put the source photos in `data/demo_photos` first. Upload the results from `data/demo_geotagged` in Photo Intake; `manifest.csv` lists what each photo should do (added automatically, or which review reason). The coordinates are synthetic, so say so in the demo.

## Panel polygons and matching tolerance
`seed.py` now draws 3 m x 3 m panel polygons with 5 m gaps (set `SEED_SIZE_M` / `SEED_GAP_M` to change). A photo matches a panel if its GPS point is inside it, or within `MATCH_BUFFER_M` (default 3 m, set in docker-compose.yml) of exactly one panel. Consumer GPS is often 3-5 m off, so real photos will often land in the review queue; raise `MATCH_BUFFER_M` if that is too strict.
After changing the geometry: `docker compose down -v`, `docker compose up --build`, re-run `seed.py`, and re-run `geotag.py` (older geotagged photos point at the old polygons).

## Photos for the synthetic records (demo)
After `seed.py`, give the seeded observations that have an issue a matching photo from the held-out test images (needs `ml/dataset_split/test/<class>/`):
```
docker compose up -d --build
docker compose exec api python attach_demo_photos.py
```
If a trained model is present, each photo also gets a Grad-CAM heat map (about a minute). Panels whose class has no photo folder, such as shading, stay without one. Re-run with `--redo` to replace, or `--include-clean` to add photos to clean panels.
