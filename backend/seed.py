import os, random, math
from datetime import date, datetime, timedelta, timezone
import psycopg2
from risk import risk_score
random.seed(7)
LAT, LON = float(os.getenv("SEED_LAT", "12.97")), float(os.getenv("SEED_LON", "77.59"))  # change to your area
c = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://nix:nix@db:5432/nix")); cur = c.cursor()
SIZE, GAP = float(os.getenv("SEED_SIZE_M", "3")), float(os.getenv("SEED_GAP_M", "5"))  # panel polygon edge and gap between polygons, in metres
MX, MY = 111320 * math.cos(math.radians(LAT)), 111320     # metres per degree of longitude / latitude here
today, assets, i = date.today(), [], 0
opt = lambda v: None if random.random() < .1 else v  # ~10% missing values to exercise missing-data handling
for st in ("State A", "State B"):
    for ai in (1, 2):
        for ci in (1, 2):
            for k in (1, 2):
                for _ in range(4):
                    i += 1; x, y, dx, dy = LON + (i % 10) * (SIZE + GAP) / MX, LAT + (i // 10) * (SIZE + GAP) / MY, SIZE / MX, SIZE / MY
                    a = dict(asset_id=f"SP-{i:04d}", x=x + dx / 2, y=y + dy / 2, station=f"{st} Station {ai}{ci}{k}",
                             install_date=opt(today - timedelta(days=random.randint(300, 3500))),
                             capacity_kw=opt(round(random.uniform(5, 50), 1)),
                             last_cleaned_date=opt(today - timedelta(days=random.randint(5, 150))))
                    assets.append(a)
                    cur.execute("INSERT INTO assets(asset_id,geom,station,city,op_area,state,install_date,capacity_kw,last_cleaned_date) "
                                "VALUES(%s,ST_GeomFromText(%s,4326),%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                                (a["asset_id"], f"POLYGON(({x} {y},{x+dx} {y},{x+dx} {y+dy},{x} {y+dy},{x} {y}))", f"{st} Station {ai}{ci}{k}",
                                 f"{st} City {ai}{ci}", f"{st} Area {ai}", st, a["install_date"], a["capacity_kw"], a["last_cleaned_date"]))
W = [("clean", .45), ("dust", .25), ("bird_droppings", .08), ("shading", .08), ("snow", .02), ("physical_damage", .07), ("electrical_damage", .05)]
for _ in range(250):
    a = random.choice(assets); cl = random.choices([w[0] for w in W], [w[1] for w in W])[0]; sv = random.choice(["low", "medium", "high"])
    rk = risk_score(a, dict(issue_class=cl, severity=sv), random.randint(0, 3))
    cur.execute("INSERT INTO observations(asset_id,captured_at,geom,source,issue_class,severity,confidence,risk_score,risk_level,energy_loss_kwh_day) "
                "VALUES(%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326),'seed',%s,%s,%s,%s,%s,%s)",
                (a["asset_id"], datetime.now(timezone.utc) - timedelta(days=random.uniform(0, 30)), a["x"], a["y"], cl, sv,
                 round(random.uniform(.6, .98), 2), rk["score"], rk["level"], rk["energy_loss_kwh_day"]))
# ---- scheduled work orders: four deliberate demo scenarios + a few ordinary visits ----
by = {}
for a in assets: by.setdefault(a["station"], []).append(a)
S = sorted(by)
def force(a, cl, sv):
    rk = risk_score(a, dict(issue_class=cl, severity=sv), 2)
    cur.execute("INSERT INTO observations(asset_id,captured_at,geom,source,issue_class,severity,confidence,risk_score,risk_level,energy_loss_kwh_day) "
                "VALUES(%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326),'seed',%s,%s,.9,%s,%s,%s)",
                (a["asset_id"], datetime.now(timezone.utc), a["x"], a["y"], cl, sv, rk["score"], rk["level"], rk["energy_loss_kwh_day"]))
    return cl, rk
def visit(station, days, kind, items):
    cur.execute("INSERT INTO work_orders(station,scheduled_for,kind,status,source) VALUES(%s,%s,%s,'scheduled','seed') RETURNING wo_id",
                (station, today + timedelta(days=days), kind)); wo = cur.fetchone()[0]
    for a, (cl, rk) in items:
        act = {"repair": "replace_module", "cleaning": "schedule_cleaning"}.get(kind, "schedule_inspection")
        cur.execute("INSERT INTO tickets(asset_id,issue_class,priority,action,notes,status,notified_status,wo_id) VALUES(%s,%s,%s,%s,'Seeded work order','scheduled','scheduled',%s)",
                    (a["asset_id"], cl, rk["level"], act, wo))
A = by[S[0]][0]; visit(S[0], 21, "repair", [(A, force(A, "physical_damage", "high"))])               # scheduled too late
B = by[S[1]][:2]; visit(S[1], -4, "cleaning", [(a, force(a, "dust", "medium")) for a in B])          # overdue
C = by[S[2]]; visit(S[2], 5, "cleaning", [(C[0], force(C[0], "dust", "high"))]); force(C[1], "dust", "medium")  # bundle candidate
D = by[S[3]][0]; force(D, "electrical_damage", "high")                                                          # critical, nothing planned
for s_ in S[4:9]:
    a = by[s_][0]; visit(s_, random.randint(1, 18), "cleaning", [(a, force(a, "dust", "low"))])
c.commit(); print(f"seeded {len(assets)} assets, 250 observations")
