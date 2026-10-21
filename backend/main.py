import os, re, uuid, json, hashlib, configparser, hmac, base64, time, secrets
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import psycopg2, psycopg2.extras
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from PIL.ExifTags import GPSTAGS
from apscheduler.schedulers.background import BackgroundScheduler
from risk import risk_score, PSH

DB = os.getenv("DATABASE_URL", "postgresql://nix:nix@db:5432/nix")
DATA, MODEL_PATH = os.getenv("DATA_DIR", "/data"), os.getenv("MODEL_PATH", "/data/model.keras")
CLASSES = os.getenv("CLASSES", "clean,dust,bird_droppings,shading,snow,physical_damage,electrical_damage").split(",")
ACTIONS = ["monitor", "schedule_cleaning", "schedule_inspection", "replace_module", "escalate_to_engineer"]
STATUSES = ["open", "scheduled", "in_progress", "resolved"]
BUFFER_M = float(os.getenv("MATCH_BUFFER_M", "3"))  # metres of GPS tolerance when a photo falls just outside a panel
from decimal import Decimal
J = lambda o: psycopg2.extras.Json(o, dumps=lambda x: json.dumps(x, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))  # never fail on Decimal or dates

def q(sql, a=None, one=False):
    c = psycopg2.connect(DB)
    try:
        with c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, a)
            if cur.description: return cur.fetchone() if one else cur.fetchall()
    finally:
        c.close()

def notify():  # scheduler job: turn ticket status changes into notifications
    for t in q("SELECT ticket_id,status FROM tickets WHERE notified_status IS DISTINCT FROM status"):
        q("INSERT INTO notifications(ticket_id,message) VALUES(%s,%s)", (t["ticket_id"], f"Ticket #{t['ticket_id']} is now {t['status']}"))
        q("UPDATE tickets SET notified_status=%s WHERE ticket_id=%s", (t["status"], t["ticket_id"]))

@asynccontextmanager
async def life(app):
    import time
    for _ in range(30):  # wait for the database to accept TCP connections
        try: q(open("schema.sql").read()); break
        except psycopg2.OperationalError: time.sleep(2)
    else: raise RuntimeError("database not reachable")
    sch = BackgroundScheduler(); sch.add_job(notify, "interval", seconds=10); sch.start()
    yield
    sch.shutdown()

app = FastAPI(title="NiX Utility AI", lifespan=life)
os.makedirs(DATA, exist_ok=True)
app.mount("/files", StaticFiles(directory=DATA), name="files")

# ---------- GIS ----------
def exif_info(path):
    ex = Image.open(path).getexif()
    g = {GPSTAGS.get(k, k): v for k, v in ex.get_ifd(0x8825).items()}
    lat = lon = None
    if "GPSLatitude" in g and "GPSLongitude" in g:
        d = lambda v, r: (-1 if r in ("S", "W") else 1) * (float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600)
        lat, lon = d(g["GPSLatitude"], g["GPSLatitudeRef"]), d(g["GPSLongitude"], g["GPSLongitudeRef"])
    try: t = datetime.strptime(ex.get_ifd(0x8769).get(36867), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception: t = datetime.now(timezone.utc)
    return lat, lon, t

def match(lat, lon):
    p = "ST_SetSRID(ST_MakePoint(%s,%s),4326)"
    r = q(f"SELECT asset_id, 0.0::float8 AS distance_m FROM assets WHERE ST_Intersects(geom,{p})", (lon, lat))
    if r: return "intersect", r
    r = q(f"SELECT asset_id, ST_Distance(geom::geography,{p}::geography) AS distance_m FROM assets "
          f"WHERE ST_DWithin(geom::geography,{p}::geography,%s) ORDER BY 2 LIMIT 3", (lon, lat, lon, lat, BUFFER_M))
    return ("buffer", r) if r else ("none", [])

# ---------- Classifier (CNN if model present, else deterministic stub) ----------
def classify(path, cam_path=None):
    """Trained EfficientNetB0 + Grad-CAM when data/model.keras exists, otherwise a deterministic stub so the rest of the app still runs."""
    try:
        import vision
        if vision.available(): return vision.predict(path, cam_path)
    except Exception as e:
        print("vision model failed, using stub:", e)
    h = int(hashlib.md5(open(path, "rb").read()).hexdigest(), 16); conf = 0.5 + (h % 45) / 100
    return dict(issue_class=CLASSES[h % len(CLASSES)], severity="medium", confidence=round(conf, 2), coverage_pct=None, needs_review=conf < .6,
                source="stub", severity_source="placeholder", gradcam_path=None)

# ---------- Universal LLM client: OpenAI-compatible chat completions, configured in llm.ini ----------
CFG_PATH = os.getenv("LLM_CONFIG", "llm.ini")
def llm_cfg():
    cp = configparser.ConfigParser(interpolation=None); cp.read(CFG_PATH)
    return cp["llm"] if cp.has_section("llm") else {}

def llm_ready(c=None):
    c = c if c is not None else llm_cfg()
    return bool(c.get("base_url")) and "your-" not in c.get("base_url", "") and bool(c.get("model")) and "your-" not in c.get("model", "")

def llm_chat(messages, tools=None, max_tokens=None):
    c = llm_cfg()
    if not llm_ready(c): raise RuntimeError("no model configured in llm.ini")
    h = {"Content-Type": "application/json", **json.loads(c.get("extra_headers", "{}") or "{}")}
    if c.get("token"):
        sch = c.get("auth_scheme", "Bearer").strip()
        h[c.get("auth_header", "Authorization")] = f"{sch} {c['token']}" if sch else c["token"]
    body = {"model": c["model"], "messages": messages, c.get("max_tokens_param", "max_tokens"): int(max_tokens or c.get("max_tokens", 900)), **json.loads(c.get("extra_body", "{}") or "{}")}
    if c.get("temperature", "").strip(): body["temperature"] = float(c["temperature"])
    if tools: body["tools"] = tools
    v = c.get("verify_ssl", "true").strip(); verify = v.lower() == "true" if v.lower() in ("true", "false") else v
    r = httpx.post(c["base_url"].rstrip("/") + c.get("chat_path", "/chat/completions"), headers=h, json=body, timeout=float(c.get("timeout_seconds", 60)), verify=verify)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

# ---------- Recommendation (LLM with rules fallback) ----------
SYS = ("You are a solar asset maintenance planner. Input is JSON with asset, observation, history and a precomputed risk. "
       "Do not recompute the score. Reply with JSON only: {action (one of %s), priority (low|medium|high|critical), timeframe, "
       "rationale (cite specific numbers), estimated_impact, needs_human_review (true if risk.completeness<0.6 or confidence<0.6)}." % ACTIONS)

def rules(obs, rk):
    c, lv = obs["issue_class"], rk["level"]
    a = ("monitor" if c == "clean" else "escalate_to_engineer" if c == "electrical_damage" else
         ("replace_module" if lv in ("high", "critical") else "schedule_inspection") if c == "physical_damage" else
         "schedule_cleaning" if lv != "low" else "monitor")
    tf = {"critical": "within 48 hours", "high": "within 7 days", "medium": "within 30 days", "low": "next routine visit"}[lv]
    return dict(action=a, priority=lv, timeframe=tf, source="rules",
                rationale=f"{c} ({obs['severity']}) with risk {rk['score']}/100 ({lv}).",
                estimated_impact=f"{rk['energy_loss_kwh_day']} kWh/day" if rk["energy_loss_kwh_day"] is not None else "unknown (capacity missing)",
                needs_human_review=obs["needs_review"] or rk["completeness"] < .6)

def recommend(ctx):
    try:
        m = llm_chat([dict(role="system", content=SYS), dict(role="user", content=json.dumps(ctx, default=str))], max_tokens=500)
        out = json.loads(re.search(r"\{.*\}", m.get("content") or "", re.S).group(0))  # tolerate prose or code fences around the JSON
        assert out["action"] in ACTIONS
        out["source"] = "llm"; return out
    except Exception:
        return rules(ctx["observation"], ctx["risk"])

# ---------- API ----------
class Pin(BaseModel): lat: float; lon: float
class Confirm(BaseModel): asset_id: str
class Decision(BaseModel): confirm: bool
class Sim(BaseModel): status: str

def sess(sid):
    s = q("SELECT * FROM sessions WHERE session_id=%s", (sid,), one=True)
    if not s: raise HTTPException(404, "session not found")
    return s

@app.post("/sessions")
async def upload(file: UploadFile = File(...), question: str = Form("What is the problem with this panel?")):
    sid = str(uuid.uuid4()); d = f"{DATA}/sessions/{sid}"; os.makedirs(d)
    p = f"{d}/original.jpg"; open(p, "wb").write(await file.read())
    try: lat, lon, t = exif_info(p)
    except Exception: lat = lon = None; t = datetime.now(timezone.utc)
    method, cands = ("none", []) if lat is None else match(lat, lon)
    q("INSERT INTO sessions(session_id,state,question,image_path,captured_at,lat,lon,method,candidates) VALUES(%s,'ASSET_MATCH',%s,%s,%s,%s,%s,%s,%s)",
      (sid, question, p, t, lat, lon, method, J(cands)))
    return sess(sid)

@app.get("/sessions/{sid}")
def get_session(sid: str): return sess(sid)

@app.post("/sessions/{sid}/pin")
def pin(sid: str, b: Pin):
    sess(sid); _, cands = match(b.lat, b.lon)
    q("UPDATE sessions SET lat=%s,lon=%s,method=%s,candidates=%s WHERE session_id=%s", (b.lat, b.lon, "manual_pin" if cands else "none", J(cands), sid))
    return sess(sid)

@app.post("/sessions/{sid}/confirm")
def confirm(sid: str, b: Confirm):
    s = sess(sid)
    a = q("SELECT asset_id,station,city,op_area,state,install_date,capacity_kw,last_cleaned_date FROM assets WHERE asset_id=%s", (b.asset_id,), one=True)
    if not a: raise HTTPException(404, "asset not found")
    obs = classify(s["image_path"])
    n = q("SELECT count(*) n FROM observations WHERE asset_id=%s AND issue_class=%s AND captured_at>now()-interval '12 months'", (a["asset_id"], obs["issue_class"]), one=True)["n"]
    rk = risk_score(a, obs, n)
    rec = recommend(dict(asset=a, observation=obs, history=dict(similar_issues_12m=n), risk=rk))
    q("INSERT INTO observations(session_id,asset_id,image_path,captured_at,geom,source,issue_class,severity,confidence,risk_score,risk_level,energy_loss_kwh_day) "
      "SELECT %s,asset_id,%s,%s,ST_Centroid(geom),'phone',%s,%s,%s,%s,%s,%s FROM assets WHERE asset_id=%s",
      (sid, s["image_path"], s["captured_at"], obs["issue_class"], obs["severity"], obs["confidence"], rk["score"], rk["level"], rk["energy_loss_kwh_day"], a["asset_id"]))
    q("UPDATE sessions SET asset_id=%s,classification=%s,risk=%s,recommendation=%s,state='AWAIT_DECISION' WHERE session_id=%s",
      (a["asset_id"], J(obs), J(rk), J(rec), sid))
    return sess(sid)

@app.post("/sessions/{sid}/decision")
def decision(sid: str, b: Decision):
    s = sess(sid)
    if not b.confirm:
        q("UPDATE sessions SET state='DECLINED' WHERE session_id=%s", (sid,)); return dict(session=sess(sid), ticket=None)
    rec, cl = s["recommendation"], s["classification"]
    q("INSERT INTO tickets(session_id,asset_id,issue_class,priority,action,notes) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id) DO NOTHING",
      (sid, s["asset_id"], cl["issue_class"], rec.get("priority", s["risk"]["level"]), rec["action"], rec.get("rationale")))
    t = q("SELECT * FROM tickets WHERE session_id=%s", (sid,), one=True)
    q("UPDATE sessions SET state='TICKET_CREATED',ticket_id=%s WHERE session_id=%s", (t["ticket_id"], sid))
    return dict(session=sess(sid), ticket=t)

@app.get("/tickets/{tid}")
def ticket(tid: int):
    t = q("SELECT * FROM tickets WHERE ticket_id=%s", (tid,), one=True)
    if not t: raise HTTPException(404)
    return t

@app.post("/tickets/{tid}/simulate")
def simulate(tid: int, b: Sim):
    if b.status not in STATUSES: raise HTTPException(400, f"status must be one of {STATUSES}")
    q("UPDATE tickets SET status=%s,updated_at=now() WHERE ticket_id=%s", (b.status, tid)); return ticket(tid)

@app.get("/notifications")
def notifications(): return q("SELECT * FROM notifications ORDER BY id DESC LIMIT 20")

@app.get("/assets.geojson")
def assets_geo():
    r = q("SELECT asset_id, ST_AsGeoJSON(geom)::json g FROM assets")
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"asset_id": x["asset_id"]}, "geometry": x["g"]} for x in r]}

COLS = {"state", "op_area", "city", "station"}
@app.get("/dashboard")
def dashboard(start: str = None, end: str = None, state: str = None, op_area: str = None, city: str = None, station: str = None, minx: float = None, miny: float = None, maxx: float = None, maxy: float = None):
    now = datetime.now(timezone.utc)
    P = dict(s=start or (now - timedelta(days=30)).isoformat(), e=end or now.isoformat(), state=state, op_area=op_area, city=city, station=station, minx=minx, miny=miny, maxx=maxx, maxy=maxy, psh=PSH)
    flt = area_flt(P)
    win = "o.captured_at BETWEEN %(s)s::timestamptz AND %(e)s::timestamptz"
    L = (f"WITH latest AS (SELECT DISTINCT ON (o.asset_id) o.asset_id,o.issue_class,o.risk_score,o.risk_level,o.energy_loss_kwh_day,a.capacity_kw,a.station,a.city,a.op_area,a.state "
         f"FROM observations o JOIN assets a ON a.asset_id=o.asset_id WHERE o.asset_confirmed AND {win} {flt} ORDER BY o.asset_id,o.captured_at DESC) ")
    imgs = q(f"SELECT count(*) n FROM observations o JOIN assets a ON a.asset_id=o.asset_id WHERE {win} {flt}", P, one=True)["n"]
    pie = q(L + "SELECT issue_class,count(*) n FROM latest GROUP BY 1", P)
    rank = q(L + "SELECT asset_id,issue_class,risk_score,risk_level,energy_loss_kwh_day,round((capacity_kw*%(psh)s)::numeric,1)::float8 AS optimum_kwh_day FROM latest WHERE issue_class<>'clean' "
             "ORDER BY risk_score DESC,energy_loss_kwh_day DESC NULLS LAST LIMIT 20", P)
    roll = q(L + "SELECT state,op_area,city,station,count(*) assets_with_issues,round(sum(energy_loss_kwh_day)::numeric,1) total_loss_kwh_day,round(sum(capacity_kw*%(psh)s)::numeric,1) total_optimum_kwh_day "
             "FROM latest WHERE issue_class<>'clean' GROUP BY ROLLUP(state,op_area,city,station) "
             "ORDER BY state NULLS FIRST,op_area NULLS FIRST,city NULLS FIRST,station NULLS FIRST", P)
    ni = q(L + f"SELECT count(*) n FROM assets a WHERE true {flt} AND NOT EXISTS (SELECT 1 FROM latest l WHERE l.asset_id=a.asset_id)", P, one=True)["n"]
    return dict(images_collected=imgs, assets_no_issue=sum(r["n"] for r in pie if r["issue_class"] == "clean"),
                assets_with_issues=sum(r["n"] for r in pie if r["issue_class"] != "clean"), assets_not_inspected=ni,
                issues_by_class={r["issue_class"]: r["n"] for r in pie if r["issue_class"] != "clean"}, urgency_ranking=rank, rollup=roll)

DUE = {"critical": 2, "high": 7, "medium": 30, "low": 60}  # visit-by days per risk level (assumption)
ORDER = ["state", "op_area", "city", "station"]
WO_NEXT, TK = {"scheduled": "in_progress", "in_progress": "done"}, {"scheduled": "scheduled", "in_progress": "in_progress", "done": "resolved"}

def area_flt(p):
    parts = [f"a.{c}=%({c})s" for c in ORDER if p.get(c)]
    if p.get("minx") is not None: parts.append("ST_Intersects(a.geom,ST_MakeEnvelope(%(minx)s,%(miny)s,%(maxx)s,%(maxy)s,4326))")  # map-extent filter
    return ("AND " + " AND ".join(parts)) if parts else ""

@app.get("/areas")  # options for each level, limited by the levels above it (cascading filters)
def areas(state: str = None, op_area: str = None, city: str = None):
    sel, out = dict(state=state, op_area=op_area, city=city), {}
    for i, c in enumerate(ORDER):
        up = [p for p in ORDER[:i] if sel.get(p)]
        w = "".join(f" AND {p}=%s" for p in up)
        out[c] = [r["v"] for r in q(f"SELECT DISTINCT {c} v FROM assets WHERE {c} IS NOT NULL{w} ORDER BY 1", [sel[p] for p in up])]
    return out

def _rules(o):
    return rules(dict(issue_class=o["issue_class"], severity=o["severity"], needs_review=False),
                 dict(level=o["risk_level"], score=o["risk_score"], energy_loss_kwh_day=o["energy_loss_kwh_day"], completeness=1))

@app.get("/planner")
def planner(start: str = None, end: str = None, state: str = None, op_area: str = None, city: str = None, station: str = None, group: str = "station"):
    group = group if group in ORDER else "station"; now = datetime.now(timezone.utc); today = now.date()
    P = dict(s=start or (now - timedelta(days=30)).isoformat(), e=end or now.isoformat(), state=state, op_area=op_area, city=city, station=station)
    flt = area_flt(P)
    L = ("WITH latest AS (SELECT DISTINCT ON (o.asset_id) o.asset_id,o.issue_class,o.severity,o.risk_score,o.risk_level,o.energy_loss_kwh_day,a.capacity_kw,a.station,a.city,a.op_area,a.state "
         f"FROM observations o JOIN assets a ON a.asset_id=o.asset_id WHERE o.asset_confirmed AND o.captured_at BETWEEN %(s)s::timestamptz AND %(e)s::timestamptz {flt} "
         "ORDER BY o.asset_id,o.captured_at DESC) ")
    rows = q(L + "SELECT * FROM latest l WHERE l.issue_class<>'clean' AND NOT EXISTS (SELECT 1 FROM tickets t WHERE t.asset_id=l.asset_id AND t.wo_id IS NOT NULL AND t.status<>'resolved') ORDER BY l.risk_score DESC", P)
    risk = {r["asset_id"]: r for r in q(L + "SELECT * FROM latest", P)}
    vrows = q("SELECT w.*,t.asset_id,t.issue_class FROM work_orders w JOIN (SELECT DISTINCT station,city,op_area,state FROM assets) a ON a.station=w.station "
              f"LEFT JOIN tickets t ON t.wo_id=w.wo_id WHERE true {flt} ORDER BY w.scheduled_for,w.wo_id", P)
    V, upcoming = {}, {}
    for r in vrows:
        v = V.setdefault(r["wo_id"], dict(wo_id=r["wo_id"], station=r["station"], scheduled_for=r["scheduled_for"], kind=r["kind"], status=r["status"], assets=[], flags=[]))
        if r["asset_id"]: v["assets"].append(dict(asset_id=r["asset_id"], issue_class=r["issue_class"], risk_level=risk.get(r["asset_id"], {}).get("risk_level")))
    for v in V.values():  # flags shown inline on each scheduled visit
        is_open = v["status"] != "done"; days = (v["scheduled_for"] - today).days
        lv = [a["risk_level"] for a in v["assets"] if a["risk_level"]]; worst = min(lv, key=lambda l: DUE[l]) if lv else None
        if is_open and days < 0:
            v["flags"].append(dict(type="overdue", sev="critical", text=f"Was due {v['scheduled_for']} and is not complete."))
        elif is_open and worst and days > DUE[worst]:
            v["flags"].append(dict(type="scheduled_too_late", sev="high", text=f"Covers a {worst} issue that should be visited within {DUE[worst]} days, but is {days} days out."))
        if v["status"] == "scheduled" and days >= 0: upcoming.setdefault(v["station"], v)  # earliest upcoming visit per station
    g, by_st = {}, {}
    for r in rows:
        r["action"] = _rules(r)["action"]; u = upcoming.get(r["station"])
        r["optimum_kwh_day"] = round(r["capacity_kw"] * PSH, 1) if r["capacity_kw"] else None
        r["join"] = dict(wo_id=u["wo_id"], date=u["scheduled_for"]) if u else None
        path = " > ".join(r[c] for c in ORDER[:ORDER.index(group) + 1])
        x = g.setdefault(path, dict(area=path, issues=0, urgent=0, no_visit_urgent=0, loss_kwh_day=0.0, optimum_kwh_day=0.0, due_in_days=999, items=[]))
        urg = r["risk_level"] in ("high", "critical")
        x["issues"] += 1; x["urgent"] += urg; x["no_visit_urgent"] += (urg and not r["join"]); x["loss_kwh_day"] += r["energy_loss_kwh_day"] or 0; x["optimum_kwh_day"] += r["optimum_kwh_day"] or 0
        x["due_in_days"] = min(x["due_in_days"], DUE[r["risk_level"]]); x["items"].append(r); by_st.setdefault(r["station"], []).append(r)
    for s_, items in by_st.items():
        if s_ in upcoming: upcoming[s_]["flags"].append(dict(type="bundle", sev="low", text=f"{len(items)} unscheduled issue(s) at this station could join this visit."))
    out = sorted(g.values(), key=lambda x: (x["due_in_days"], -x["loss_kwh_day"]))
    for x in out: x["loss_kwh_day"], x["optimum_kwh_day"] = round(x["loss_kwh_day"], 1), round(x["optimum_kwh_day"], 1)
    return dict(groups=out, visits=list(V.values()))

def attach(wo, aids):  # put assets' issues on a visit (reuse an open ticket if one exists, else create it)
    n = 0
    for aid in aids:
        if q("SELECT 1 x FROM tickets WHERE asset_id=%s AND status<>'resolved' AND wo_id IS NOT NULL", (aid,), one=True): continue
        if q("UPDATE tickets SET wo_id=%s,status='scheduled',updated_at=now() WHERE asset_id=%s AND status<>'resolved' AND wo_id IS NULL RETURNING ticket_id", (wo, aid)): n += 1; continue
        o = q("SELECT issue_class,severity,risk_score,risk_level,energy_loss_kwh_day FROM observations WHERE asset_id=%s AND asset_confirmed ORDER BY captured_at DESC LIMIT 1", (aid,), one=True)
        if not o or o["issue_class"] == "clean": continue
        r = _rules(o)
        q("INSERT INTO tickets(asset_id,issue_class,priority,action,notes,status,wo_id) VALUES(%s,%s,%s,%s,%s,'scheduled',%s)",
          (aid, o["issue_class"], r["priority"], r["action"], "Planned from maintenance planner. " + r["rationale"], wo)); n += 1
    return n

class Assign(BaseModel): asset_ids: list[str]; wo_id: int
class Sched(BaseModel): asset_ids: list[str]; date: str
class Resched(BaseModel): date: str

@app.post("/planner/assign")
def assign(b: Assign): return dict(added=attach(b.wo_id, b.asset_ids))

@app.post("/planner/schedule")
def schedule(b: Sched):
    by, made = {}, []
    for r in q("SELECT asset_id,station FROM assets WHERE asset_id=ANY(%s)", (b.asset_ids,)): by.setdefault(r["station"], []).append(r["asset_id"])
    for s_, aids in by.items():
        wo = q("INSERT INTO work_orders(station,scheduled_for,kind) VALUES(%s,%s,'cleaning') RETURNING wo_id", (s_, b.date), one=True)["wo_id"]
        attach(wo, aids)
        acts = {x["action"] for x in q("SELECT action FROM tickets WHERE wo_id=%s", (wo,))}
        if not acts: q("DELETE FROM work_orders WHERE wo_id=%s", (wo,)); continue
        kind = "repair" if acts & {"replace_module", "escalate_to_engineer"} else "inspection" if "schedule_inspection" in acts else "cleaning"
        q("UPDATE work_orders SET kind=%s WHERE wo_id=%s", (kind, wo)); made.append(wo)
    return dict(created=made)

@app.post("/work-orders/{wid}/advance")
def wo_advance(wid: int):
    w = q("SELECT status FROM work_orders WHERE wo_id=%s", (wid,), one=True)
    if not w: raise HTTPException(404, "work order not found")
    n = WO_NEXT.get(w["status"])
    if n:
        q("UPDATE work_orders SET status=%s WHERE wo_id=%s", (n, wid)); q("UPDATE tickets SET status=%s,updated_at=now() WHERE wo_id=%s", (TK[n], wid))
    return dict(status=n or w["status"])

@app.post("/work-orders/{wid}/reschedule")
def wo_resched(wid: int, b: Resched):
    q("UPDATE work_orders SET scheduled_for=%s WHERE wo_id=%s", (b.date, wid)); return dict(ok=True)

@app.get("/tickets")
def tickets():
    return q("SELECT t.ticket_id,t.asset_id,a.station,a.city,t.issue_class,t.priority,t.action,t.status,t.wo_id,t.created_at FROM tickets t LEFT JOIN assets a ON a.asset_id=t.asset_id ORDER BY t.ticket_id DESC LIMIT 50")

def _file_url(path):
    return "/files" + path[len(DATA):] if path and path.startswith(DATA) and os.path.exists(path) else None

@app.get("/map")  # panels with their latest status in the window, for the dashboard map
def map_data(start: str = None, end: str = None, state: str = None, op_area: str = None, city: str = None, station: str = None):
    now = datetime.now(timezone.utc)
    P = dict(s=start or (now - timedelta(days=30)).isoformat(), e=end or now.isoformat(), state=state, op_area=op_area, city=city, station=station, psh=PSH)
    flt = area_flt(P)
    L = ("WITH latest AS (SELECT DISTINCT ON (o.asset_id) o.asset_id,o.issue_class,o.risk_score,o.risk_level,o.energy_loss_kwh_day,o.captured_at,o.image_path,o.source "
         f"FROM observations o JOIN assets a ON a.asset_id=o.asset_id WHERE o.asset_confirmed AND o.captured_at BETWEEN %(s)s::timestamptz AND %(e)s::timestamptz {flt} "
         "ORDER BY o.asset_id,o.captured_at DESC) ")
    r = q(L + "SELECT a.asset_id,a.station,a.city,a.op_area,a.state,ST_AsGeoJSON(a.geom)::json g,l.issue_class,l.risk_level,l.risk_score,l.energy_loss_kwh_day,round((a.capacity_kw*%(psh)s)::numeric,1)::float8 AS optimum_kwh_day,l.captured_at,l.image_path,l.source,"
              "EXISTS(SELECT 1 FROM tickets t WHERE t.asset_id=a.asset_id AND t.wo_id IS NOT NULL AND t.status<>'resolved') AS scheduled "
              "FROM assets a LEFT JOIN latest l ON l.asset_id=a.asset_id WHERE true " + flt, P)
    feats = []
    for x in r:  # turn on-disk photo paths into URLs, only when the file exists; never expose disk paths
        g, ip = x.pop("g"), x.pop("image_path")
        x["image_url"], x["cam_url"] = _file_url(ip), (_file_url(os.path.splitext(ip)[0] + "_cam.jpg") if ip else None)
        feats.append({"type": "Feature", "geometry": g, "properties": x})
    return {"type": "FeatureCollection", "features": feats}

# ---------- Chat AI: the model answers from read-only, parameterised tools (never free-form SQL) ----------
def _latest(level=None, name=None, days=30):
    now = datetime.now(timezone.utc); P = dict(s=(now - timedelta(days=int(days or 30))).isoformat(), e=now.isoformat())
    if level in ORDER and name: P[level] = name
    flt = area_flt(P)
    L = ("WITH latest AS (SELECT DISTINCT ON (o.asset_id) o.asset_id,o.issue_class,o.severity,o.risk_score,o.risk_level,o.energy_loss_kwh_day,a.capacity_kw,a.station,a.city,a.op_area,a.state "
         f"FROM observations o JOIN assets a ON a.asset_id=o.asset_id WHERE o.asset_confirmed AND o.captured_at BETWEEN %(s)s::timestamptz AND %(e)s::timestamptz {flt} "
         "ORDER BY o.asset_id,o.captured_at DESC) ")
    return L, P, flt

def t_resolve(text):
    out = [dict(level="asset", name=r["asset_id"], station=r["station"]) for r in q("SELECT asset_id,station FROM assets WHERE asset_id ILIKE %s LIMIT 5", (f"%{text}%",))]
    for i, c in enumerate(ORDER):
        cols = ",".join(ORDER[:i + 1])
        for r in q(f"SELECT DISTINCT {cols} FROM assets WHERE {c} ILIKE %s ORDER BY {cols} LIMIT 6", (f"%{text}%",)):
            out.append(dict(level=c, name=r[c], path=" > ".join(r[x] for x in ORDER[:i + 1])))
    return out

def t_asset(asset_id):
    a = q("SELECT asset_id,station,city,op_area,state,install_date,capacity_kw,last_cleaned_date FROM assets WHERE asset_id=%s", (asset_id,), one=True)
    if not a: return {"error": f"asset {asset_id} not found"}
    hist = q("SELECT captured_at,issue_class,severity,risk_score,risk_level,energy_loss_kwh_day FROM observations WHERE asset_id=%s AND asset_confirmed ORDER BY captured_at DESC LIMIT 5", (asset_id,))
    tk = q("SELECT t.ticket_id,t.status,t.action,t.priority,w.scheduled_for FROM tickets t LEFT JOIN work_orders w ON w.wo_id=t.wo_id WHERE t.asset_id=%s AND t.status<>'resolved'", (asset_id,))
    risk = None
    if hist:
        n = q("SELECT count(*) n FROM observations WHERE asset_id=%s AND issue_class=%s AND captured_at>now()-interval '12 months'", (asset_id, hist[0]["issue_class"]), one=True)["n"]
        risk = risk_score(a, dict(issue_class=hist[0]["issue_class"], severity=hist[0]["severity"]), n)
    return dict(asset=a, latest_observations=hist, open_tickets=tk, risk=risk, optimum_kwh_day=round(a["capacity_kw"] * PSH, 1) if a["capacity_kw"] else None)

def t_area(level=None, name=None, days=30):
    L, P, flt = _latest(level, name, days)
    rows = q(L + "SELECT issue_class,risk_level,energy_loss_kwh_day,capacity_kw FROM latest", P)
    tq = q(f"SELECT count(*) n,coalesce(sum(a.capacity_kw),0) c FROM assets a WHERE true {flt}", P, one=True); total = tq["n"]
    iss = [r for r in rows if r["issue_class"] != "clean"]; by, lv = {}, {}
    for r in iss: by[r["issue_class"]] = by.get(r["issue_class"], 0) + 1; lv[r["risk_level"]] = lv.get(r["risk_level"], 0) + 1
    ov = q(f"SELECT count(*) n FROM work_orders w WHERE w.status<>'done' AND w.station IN (SELECT a.station FROM assets a WHERE true {flt})", P, one=True)["n"]
    return dict(scope=f"{level} = {name}" if level and name else "all areas", window_days=int(days or 30), assets_total=total, assets_inspected=len(rows),
                assets_clean=len(rows) - len(iss), assets_with_issues=len(iss), assets_not_inspected=total - len(rows), issues_by_class=by, issues_by_risk_level=lv,
                total_loss_kwh_day=round(sum(r["energy_loss_kwh_day"] or 0 for r in iss), 1),
                optimum_kwh_day_assets_with_issues=round(sum((r["capacity_kw"] or 0) * PSH for r in iss), 1), optimum_kwh_day_all_assets=round(float(tq["c"]) * PSH, 1), open_visits=ov)

def t_top(level=None, name=None, n=5, days=30):
    L, P, _ = _latest(level, name, days)
    return q(L + "SELECT asset_id,station,issue_class,severity,risk_score,risk_level,energy_loss_kwh_day,round((capacity_kw*%(psh)s)::numeric,1)::float8 AS optimum_kwh_day FROM latest WHERE issue_class<>'clean' "
             "ORDER BY risk_score DESC,energy_loss_kwh_day DESC NULLS LAST LIMIT %(n)s", {**P, "n": min(int(n or 5), 20), "psh": PSH})

def t_visits(level=None, name=None):
    P = {level: name} if level in ORDER and name else {}; flt = area_flt(P)
    return q("SELECT w.wo_id,w.station,w.scheduled_for,w.kind,w.status,(SELECT array_agg(t.asset_id) FROM tickets t WHERE t.wo_id=w.wo_id) AS assets "
             "FROM work_orders w JOIN (SELECT DISTINCT station,city,op_area,state FROM assets) a ON a.station=w.station "
             f"WHERE w.status<>'done' {flt} ORDER BY w.scheduled_for LIMIT 20", P)

def obj(req=(), **props): return dict(type="object", properties=props, required=list(req))
STR, NUM = dict(type="string"), dict(type="integer")
LEVEL = dict(type="string", enum=ORDER, description="state, op_area, city or station")
TOOLS = [
    dict(name="resolve_entity", description="Find assets, stations, cities, operating areas or states matching a name or fragment the user typed. Use when unsure of exact names.", input_schema=obj(["text"], text=STR)),
    dict(name="get_asset", description="Attributes, latest observations, risk breakdown and open tickets for one asset id such as SP-0001.", input_schema=obj(["asset_id"], asset_id=STR)),
    dict(name="area_summary", description="Inspection summary for a level and its exact name. Omit both for all areas.", input_schema=obj([], level=LEVEL, name=STR, days=NUM)),
    dict(name="top_risks", description="Most urgent assets with issues, highest risk first.", input_schema=obj([], level=LEVEL, name=STR, n=NUM, days=NUM)),
    dict(name="list_visits", description="Scheduled, in-progress or overdue maintenance visits, optionally within one area.", input_schema=obj([], level=LEVEL, name=STR)),
    dict(name="propose_schedule", description="Propose scheduling maintenance for these asset ids. This does NOT schedule anything; the user confirms in the UI.", input_schema=obj(["asset_ids"], asset_ids=dict(type="array", items=STR))),
]
CHAT_SYS = ("You are NiX, an assistant for solar asset maintenance at a utility. Answer only from tool results and never invent numbers. "
            "Call resolve_entity when a name is unclear, and ask the user which one if several match. Interpret this area or this station using the current UI filters. "
            "State the time window you used. Be concise: short sentences, include asset ids and numbers, a short list when it helps. "
            "Risk scores are 0 to 100 from a deterministic engine with levels low, medium, high and critical. kWh/day lost is the estimated output lost to an issue; kWh/day generated at optimum capacity is panel capacity times 5 peak sun hours with no losses. "
            "If asked to schedule or raise work, call propose_schedule with the asset ids and tell the user to confirm with the button. Never say work has been scheduled.")

def run_tool(name, a, prop):
    try:
        if name == "resolve_entity": return t_resolve(a["text"])
        if name == "get_asset": return t_asset(a["asset_id"])
        if name == "area_summary": return t_area(a.get("level"), a.get("name"), a.get("days", 30))
        if name == "top_risks": return t_top(a.get("level"), a.get("name"), a.get("n", 5), a.get("days", 30))
        if name == "list_visits": return t_visits(a.get("level"), a.get("name"))
        if name == "propose_schedule":
            prop.extend(a.get("asset_ids", [])); return dict(status="proposed, waiting for user confirmation", count=len(a.get("asset_ids", [])))
        return {"error": "unknown tool"}
    except Exception as e:
        return {"error": str(e)}

def fmt_area(s, top):
    t = "; ".join(f"{r['asset_id']} ({r['issue_class']}, {r['risk_level']}, {r['risk_score']}/100)" for r in top[:3]) or "none"
    return (f"{s['scope']}, last {s['window_days']} days: {s['assets_total']} assets, {s['assets_inspected']} inspected, {s['assets_with_issues']} with issues, "
            f"{s['assets_not_inspected']} not inspected. Issues by type: {s['issues_by_class'] or 'none'}. Estimated loss {s['total_loss_kwh_day']} kWh/day lost, against {s['optimum_kwh_day_assets_with_issues']} kWh/day generated at optimum capacity for those assets. Open visits: {s['open_visits']}. Most urgent: {t}.")

def fmt_asset(d):
    if "error" in d: return d["error"]
    a, k, o = d["asset"], d["risk"], (d["latest_observations"] or [None])[0]
    s = f"{a['asset_id']} at {a['station']} ({a['city']}, {a['op_area']}, {a['state']}); capacity {a['capacity_kw']} kW; installed {a['install_date']}; last cleaned {a['last_cleaned_date']}. "
    s += (f"Latest: {o['issue_class']} ({o['severity']}) on {o['captured_at']:%Y-%m-%d}, risk {k['score']}/100 ({k['level']}). " if o else "No observations on record. ")
    if d.get("optimum_kwh_day"): s += f"Generates {d['optimum_kwh_day']} kWh/day at optimum capacity. "
    return s + f"Open tickets: {len(d['open_tickets'])}."

def find_entity(text, ctx):  # most specific named place in the text, else the UI filter
    low, hit = text.lower(), None
    for c in reversed(ORDER):
        for r in q(f"SELECT DISTINCT {c} v FROM assets"):
            if r["v"] and r["v"].lower() in low and (not hit or len(r["v"]) > len(hit[1])): hit = (c, r["v"])
        if hit: break
    for c in reversed(ORDER):
        if not hit and ctx.get(c): hit = (c, ctx[c])
    return hit if hit else (None, None)

def fallback(text, ctx):  # basic data answer when no model is available
    m = re.search(r"SP-\d+", text, re.I)
    if m: return fmt_asset(t_asset(m.group(0).upper()))
    lv, nm = find_entity(text, ctx); return fmt_area(t_area(lv, nm), t_top(lv, nm))

def context_for(text, ctx):  # data handed to the model when tool calling is switched off
    m = re.search(r"SP-\d+", text, re.I)
    if m: return dict(asset=t_asset(m.group(0).upper()))
    lv, nm = find_entity(text, ctx); return dict(area_summary=t_area(lv, nm), top_risks=t_top(lv, nm, 8), visits=t_visits(lv, nm))

OA_TOOLS = [dict(type="function", function=dict(name=t["name"], description=t["description"], parameters=t["input_schema"])) for t in TOOLS]
PLAIN_SYS = ("You are NiX, an assistant for solar asset maintenance at a utility. Answer only from the data provided below and never invent numbers. "
             "State the time window used. Be concise, with asset ids and numbers. If the data does not contain the answer, say so. "
             "You cannot schedule work from chat; point the user to the Maintenance Planner.")

class Chat(BaseModel): messages: list[dict]; context: dict = {}

def history(b): return [dict(role=m["role"], content=m["content"]) for m in b.messages][-12:]

def chat_plain(b, ctx):
    data = json.dumps(context_for(b.messages[-1]["content"], ctx), default=str)
    return llm_chat([dict(role="system", content=PLAIN_SYS + f" Current UI filters: {json.dumps(ctx)}. Data: {data}")] + history(b)).get("content") or ""

def chat_tools(b, ctx, prop, c):
    msgs = [dict(role="system", content=CHAT_SYS + f" Current UI filters: {json.dumps(ctx)}")] + history(b)
    for _ in range(int(c.get("max_tool_rounds", 6))):
        m = llm_chat(msgs, OA_TOOLS); calls = m.get("tool_calls") or []
        if not calls: return dict(reply=m.get("content") or "", proposal=sorted(set(prop)), source="llm")
        msgs.append(dict(role="assistant", content=m.get("content") or "", tool_calls=calls))
        for tc in calls:
            try: args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception: args = {}
            msgs.append(dict(role="tool", tool_call_id=tc["id"], content=json.dumps(run_tool(tc["function"]["name"], args, prop), default=str)))
    raise RuntimeError("too many tool rounds")

@app.post("/chat")
def chat(b: Chat):
    prop, ctx, c = [], {k: v for k, v in b.context.items() if v}, llm_cfg()
    try:
        if str(c.get("use_tools", "true")).strip().lower() != "true": return dict(reply=chat_plain(b, ctx), proposal=[], source="llm")
        try: return chat_tools(b, ctx, prop, c)
        except httpx.HTTPStatusError as e:  # endpoint rejects tool calling: retry as plain context
            if e.response.status_code in (400, 422): return dict(reply=chat_plain(b, ctx), proposal=[], source="llm")
            raise
    except Exception as e:
        note = "No model is configured in llm.ini" if not llm_ready(c) else f"The model endpoint returned an error ({str(e)[:100]})"
        return dict(reply=f"({note}, so this is a basic data answer.) " + fallback(b.messages[-1]["content"], ctx), proposal=[], source="fallback")

# ---------- Batch photo ingestion: clear matches are recorded automatically, doubtful ones wait for a human ----------
def record(aid, path, when, obs, src="photo"):
    a = q("SELECT asset_id,install_date,capacity_kw,last_cleaned_date FROM assets WHERE asset_id=%s", (aid,), one=True)
    if not a: return None
    n = q("SELECT count(*) n FROM observations WHERE asset_id=%s AND issue_class=%s AND captured_at>now()-interval '12 months'", (aid, obs["issue_class"]), one=True)["n"]
    rk = risk_score(a, obs, n)
    q("INSERT INTO observations(asset_id,image_path,captured_at,geom,source,issue_class,severity,confidence,risk_score,risk_level,energy_loss_kwh_day) "
      "SELECT asset_id,%s,%s,ST_Centroid(geom),%s,%s,%s,%s,%s,%s,%s FROM assets WHERE asset_id=%s",
      (path, when, src, obs["issue_class"], obs["severity"], obs["confidence"], rk["score"], rk["level"], rk["energy_loss_kwh_day"], aid))
    return rk

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    out = dict(auto=0, review=0); os.makedirs(f"{DATA}/ingest", exist_ok=True)
    for f in files:
        iid = str(uuid.uuid4()); p = f"{DATA}/ingest/{iid}.jpg"; open(p, "wb").write(await f.read())
        try: lat, lon, t = exif_info(p)
        except Exception: lat = lon = None; t = datetime.now(timezone.utc)
        method, cands = ("none", []) if lat is None else match(lat, lon)
        obs = classify(p)
        reason = ("no GPS tag on the photo" if lat is None else "no panel at this location" if not cands else "several panels nearby" if len(cands) > 1 else
                  "location is near a panel, not on it" if method == "buffer" else "low classifier confidence" if obs["needs_review"] else None)
        status, aid = "pending", None
        if reason is None: aid = cands[0]["asset_id"]; record(aid, p, t, obs); status = "auto"
        q("INSERT INTO ingest_items(item_id,image_path,captured_at,lat,lon,method,candidates,classification,reason,status,asset_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
          (iid, p, t, lat, lon, method, J(cands), J(obs), reason, status, aid))
        out["auto" if status == "auto" else "review"] += 1
    return out

@app.get("/ingest/queue")
def ingest_queue():
    items = q("SELECT item_id,captured_at,lat,lon,candidates,classification,reason FROM ingest_items WHERE status='pending' ORDER BY created_at LIMIT 50")
    return dict(items=items, counts={r["status"]: r["n"] for r in q("SELECT status,count(*) n FROM ingest_items GROUP BY status")})

class Verify(BaseModel): asset_id: str; issue_class: str | None = None

@app.post("/ingest/{iid}/verify")
def verify(iid: str, b: Verify):
    it = q("SELECT * FROM ingest_items WHERE item_id=%s AND status='pending'", (iid,), one=True)
    if not it: raise HTTPException(404, "item not found or already handled")
    obs = dict(it["classification"])
    if b.issue_class: obs.update(issue_class=b.issue_class, confidence=1.0)
    if not record(b.asset_id, it["image_path"], it["captured_at"], obs, "photo_verified"): raise HTTPException(404, f"asset {b.asset_id} not found")
    q("UPDATE ingest_items SET status='verified',asset_id=%s WHERE item_id=%s", (b.asset_id, iid)); return dict(ok=True)

@app.post("/ingest/{iid}/reject")
def reject(iid: str):
    q("UPDATE ingest_items SET status='rejected' WHERE item_id=%s AND status='pending'", (iid,)); return dict(ok=True)

# ---------- Shared login: one user from auth.ini, signed session cookie ----------
AUTH_PATH, _secret, _tries = os.getenv("AUTH_CONFIG", "auth.ini"), secrets.token_bytes(32), {}
def auth_cfg():
    cp = configparser.ConfigParser(interpolation=None); cp.read(AUTH_PATH)
    return cp["auth"] if cp.has_section("auth") else {}
def _key(c): return (c.get("secret") or "").encode() or _secret
def make_token(c):
    body = base64.urlsafe_b64encode(json.dumps({"u": c.get("username"), "exp": int(time.time() + float(c.get("session_hours", 12)) * 3600)}).encode()).decode()
    return body + "." + hmac.new(_key(c), body.encode(), hashlib.sha256).hexdigest()
def valid_token(tok, c):
    try:
        body, sig = tok.split(".", 1)
        if not hmac.compare_digest(sig, hmac.new(_key(c), body.encode(), hashlib.sha256).hexdigest()): return False
        d = json.loads(base64.urlsafe_b64decode(body)); return d["exp"] > time.time() and d["u"] == c.get("username")
    except Exception: return False

PUBLIC = {"/login", "/login.html", "/nix-logo.png", "/nix-icon.png", "/health"}
@app.middleware("http")
async def gate(request: Request, call_next):
    if request.url.path in PUBLIC or valid_token(request.cookies.get("nix_session", ""), auth_cfg()): return await call_next(request)
    if request.url.path in ("/", "/index.html"): return RedirectResponse("/login")
    return JSONResponse({"detail": "login required"}, status_code=401)

class Login(BaseModel): username: str; password: str

@app.post("/login")
def login(b: Login, request: Request):
    c = auth_cfg(); ip = request.client.host if request.client else "?"; now = time.time()
    recent = _tries[ip] = [t for t in _tries.get(ip, []) if now - t < 300]
    if not c.get("password") or c["password"] == "CHANGE_ME": raise HTTPException(503, "Set a password in auth.ini first.")
    if len(recent) >= 5: raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    ok = hmac.compare_digest(b.username.encode(), c.get("username", "").encode()) & hmac.compare_digest(b.password.encode(), c["password"].encode())
    if not ok: recent.append(now); raise HTTPException(401, "Wrong username or password.")
    r = JSONResponse({"ok": True})
    r.set_cookie("nix_session", make_token(c), max_age=int(float(c.get("session_hours", 12)) * 3600), httponly=True, samesite="lax", secure=str(c.get("cookie_secure", "false")).lower() == "true")
    return r

@app.post("/logout")
def logout():
    r = JSONResponse({"ok": True}); r.delete_cookie("nix_session"); return r

@app.get("/login")
def login_page(): return FileResponse("frontend/login.html")

@app.get("/health")
def health(): return {"ok": True}

if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
