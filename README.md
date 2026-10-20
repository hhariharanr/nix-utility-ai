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
