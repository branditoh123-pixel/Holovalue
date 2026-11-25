"""Price history for every card and sealed product (English + Japanese).

Past prices come from the tcgcsv.com daily price archive (immutable files, each
day downloaded at most once and kept in ARCHIVE_CACHE). New days come straight
from the nightly prices.json / prices_jp.json, so normally nothing is downloaded.

Kept dates (days counted from 2024-01-01, so the grid never shifts):
  last 31 days: every day that is a multiple of 3 or of 7
  older, back to ~1 year: multiples of 7
  plus the newest day.
Output: history/h{pid % 64}.json  {pid: [subtype, [day, price], ...]} and
history/meta.json {dates:[...], latest, epoch}.
"""
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

EPOCH = dt.date(2024, 1, 1)
FIRST = dt.date(2024, 2, 8)
SHARDS = 64
KEEP_DAYS = 371
CACHE = os.environ.get("ARCHIVE_CACHE", "archive-cache")
MAX_DOWNLOADS = int(os.environ.get("MAX_DOWNLOADS", "70"))
CATS = ("3", "85")
PREF = ["Holofoil", "Normal", "1st Edition Holofoil", "Unlimited Holofoil", "1st Edition", "Unlimited", "Reverse Holofoil"]
UA = {"User-Agent": "HoloValue-history/1.0 (+https://github.com/branditoh123-pixel/Holovalue)"}


def day(d):
    return (d - EPOCH).days


def date_of(n):
    return EPOCH + dt.timedelta(days=n)


def wanted(today):
    t = day(today)
    out = set()
    for n in range(max(day(FIRST), t - KEEP_DAYS), t + 1):
        age = t - n
        if (age <= 31 and (n % 3 == 0 or n % 7 == 0)) or n % 7 == 0:
            out.add(n)
    return out


def main_sub(prices):
    """prices: {subtype: [low, market, high]} -> the subtype the app shows by default."""
    for s in PREF:
        v = prices.get(s)
        if v and v[1] is not None:
            return s
    for s, v in prices.items():
        if v and v[1] is not None:
            return s
    return None


def targets():
    """pid -> subtype to track, from the current price files."""
    out = {}
    for f in ("prices.json", "prices_jp.json"):
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for rows in d.get("cards", {}).values():
            for r in rows:
                s = main_sub(r[3] or {})
                if s:
                    out[int(r[2])] = s
        for r in d.get("sealed", []):
            out[int(r[2])] = "Normal"
    return out


def today_points(tg):
    """Current market for each tracked pid from the price files."""
    pts = {}
    for f in ("prices.json", "prices_jp.json"):
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for rows in d.get("cards", {}).values():
            for r in rows:
                pid = int(r[2])
                v = (r[3] or {}).get(tg.get(pid, ""))
                if v and v[1] is not None:
                    pts[pid] = v[1]
        for r in d.get("sealed", []):
            if r[3] is not None:
                pts[int(r[2])] = r[3]
    return pts


def archive_points(d, tg):
    name = f"prices-{d.isoformat()}.ppmd.7z"
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        url = f"https://tcgcsv.com/archive/tcgplayer/{name}"
        for i in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                    data = r.read()
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)
                break
            except Exception as e:  # noqa: BLE001
                print("download failed", name, e)
                time.sleep(3 + 3 * i)
        else:
            return None
    work = "/tmp/hist-x"
    subprocess.run(["rm", "-rf", work], check=False)
    os.makedirs(work, exist_ok=True)
    pats = [f"{d.isoformat()}/{c}/*" for c in CATS]
    r = subprocess.run(["7z", "x", "-y", f"-o{work}", path, *pats, "-r"], capture_output=True, text=True)
    if r.returncode not in (0, 1):
        print("extract failed", name, r.stderr[-300:])
        return None
    if not glob.glob(f"{work}/**/prices", recursive=True):
        subprocess.run(["7z", "x", "-y", f"-o{work}", path], capture_output=True, text=True)
    pts = {}
    for c in CATS:
        for f in (glob.glob(f"{work}/{d.isoformat()}/{c}/*/prices") or glob.glob(f"{work}/**/{c}/*/prices", recursive=True)):
            try:
                res = json.load(open(f)).get("results", [])
            except Exception:  # noqa: BLE001
                continue
            for p in res:
                pid = p.get("productId")
                if pid in tg and p.get("subTypeName") == tg[pid] and p.get("marketPrice") is not None:
                    pts[pid] = p["marketPrice"]
    subprocess.run(["rm", "-rf", work], check=False)
    return pts


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs("history", exist_ok=True)
    src = json.load(open("prices.json"))
    stamp = src.get("source") or src.get("updated")
    today = dt.date.fromisoformat(str(stamp)[:10])
    meta = json.load(open("history/meta.json")) if os.path.exists("history/meta.json") else {"dates": []}
    if meta.get("latest") == stamp and not os.environ.get("FORCE"):
        print("history already has", stamp)
        return
    tg = targets()
    series = {}
    for f in glob.glob("history/h*.json"):
        for pid, s in json.load(open(f)).items():
            series[int(pid)] = s
    have = set(meta.get("dates", []))
    keep = wanted(today)
    t = day(today)
    # newest day always comes from the nightly file
    new_days = {t: today_points(tg)}
    missing = sorted((keep - have) - {t}, reverse=True)[:MAX_DOWNLOADS]
    for n in missing:
        pts = archive_points(date_of(n), tg)
        if pts:
            new_days[n] = pts
            print("archive", date_of(n), len(pts))
    keep_all = keep | {t}
    dates = sorted((have & keep_all) | set(new_days))
    for pid, sub in tg.items():
        old = series.get(pid)
        pts = {}
        if old and old[0] == sub:
            for n, p in old[1:]:
                if n in keep_all:
                    pts[n] = p
        for n, d in new_days.items():
            if pid in d:
                pts[n] = d[pid]
        # drop the previous "newest" point if it isn't on the grid
        if pts:
            series[pid] = [sub] + [[n, round(pts[n], 2)] for n in sorted(pts)]
        else:
            series.pop(pid, None)
    for pid in [p for p in series if p not in tg]:
        series.pop(pid)
    shards = [{} for _ in range(SHARDS)]
    for pid, s in series.items():
        if len(s) > 1:
            shards[pid % SHARDS][str(pid)] = s
    for i, sh in enumerate(shards):
        with open(f"history/h{i}.json", "w") as f:
            json.dump(sh, f, separators=(",", ":"))
    meta = {"epoch": EPOCH.isoformat(), "latest": stamp, "today": t, "dates": dates, "cards": sum(len(s) for s in shards)}
    json.dump(meta, open("history/meta.json", "w"), separators=(",", ":"))
    print("history:", meta["cards"], "items,", len(dates), "dates")


if __name__ == "__main__":
    sys.exit(main())
