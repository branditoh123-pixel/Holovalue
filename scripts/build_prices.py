"""Build prices.json for the HoloValue scanner from tcgcsv.com (a daily mirror of TCGplayer prices).

Output shape (compact):
{
  "updated": "2026-11-23",
  "sets": {"ME03: Perfect Order": [24600, "2026-09-26", "ME03"], ...},
  "cards": {
    "96/88": [["Probopass", "ME03: Perfect Order", 123456, {"Holofoil": [low, market, high]}], ...],
    ...
  }
}
The key is the printed collector number with leading zeros removed ("096/088" -> "96/88").
"""
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

BASE = "https://tcgcsv.com/tcgplayer/3"  # category 3 = Pokémon (English)
UA = {"User-Agent": "HoloValue-price-builder/1.0 (+https://github.com/branditoh123-pixel/Holovalue)"}


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print(f"giving up on {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))


def norm_number(value):
    m = re.match(r"^\s*0*(\d+)\s*/\s*0*(\d+)\s*$", value or "")
    if not m:
        return None
    return f"{int(m.group(1))}/{int(m.group(2))}"


def r2(x):
    return round(x, 2) if isinstance(x, (int, float)) else None


SEALED = re.compile(r"booster box|elite trainer box|booster bundle|booster pack|sleeved booster|blister|build & battle|collection|premium|tin|mini tin|booster display|ultra-premium|upc", re.I)


def build(groups, fetch_products, fetch_prices):
    cards = {}
    sealed = []
    for g in groups:
        gid, gname = g.get("groupId"), g.get("name", "")
        products = fetch_products(gid) or []
        prices = fetch_prices(gid) or []
        by_pid = {}
        for p in prices:
            sub = p.get("subTypeName") or "Normal"
            row = [r2(p.get("lowPrice")), r2(p.get("marketPrice")), r2(p.get("highPrice"))]
            if any(v is not None for v in row):
                by_pid.setdefault(p.get("productId"), {})[sub] = row
        for prod in products:
            ext = {e.get("name"): e.get("value") for e in (prod.get("extendedData") or [])}
            key = norm_number(ext.get("Number"))
            if not key:
                pname = prod.get("name") or ""
                if SEALED.search(pname) and not ext.get("Number") and not re.search(r"code card|jumbo|deck box|sleeves|binder|playmat", pname, re.I):
                    pr = by_pid.get(prod.get("productId"), {})
                    mk = next((v[1] for v in pr.values() if v[1] is not None), None)
                    if mk:
                        sealed.append([pname, gname, prod.get("productId"), mk])
                continue
            name = re.split(r"\s+-\s+", prod.get("name") or "")[0].strip()
            pid = prod.get("productId")
            cards.setdefault(key, []).append([name, gname, pid, by_pid.get(pid, {})])
    return cards, sealed


def market(row):
    vals = [v[1] for v in (row[3] or {}).values() if v and v[1] is not None]
    return max(vals) if vals else None


def movers(old_cards, new_cards, limit=40):
    old = {}
    for key, rows in (old_cards or {}).items():
        for r in rows:
            m = market(r)
            if m:
                old[r[2]] = m
    out = []
    for key, rows in new_cards.items():
        for r in rows:
            m, o = market(r), old.get(r[2])
            if m and o and m >= 3 and abs(m - o) >= 0.5:
                out.append([r[0], r[1], r[2], key, round(o, 2), round(m, 2)])
    pct = lambda x: (x[5] - x[4]) / x[4]
    up = sorted([x for x in out if x[5] > x[4]], key=pct, reverse=True)[:limit]
    down = sorted([x for x in out if x[5] < x[4]], key=pct)[:limit]
    return up, down


def collect(base):
    data = get(f"{base}/groups")
    groups = (data or {}).get("results") or []
    if not groups:
        return [], {}, []

    def products(gid):
        time.sleep(0.05)
        return (get(f"{base}/{gid}/products") or {}).get("results")

    def prices(gid):
        time.sleep(0.05)
        return (get(f"{base}/{gid}/prices") or {}).get("results")

    cards, sealed = build(groups, products, prices)
    return groups, cards, sealed


def build_japanese(stamp, today):
    cats = (get("https://tcgcsv.com/tcgplayer/categories") or {}).get("results") or []
    jp = next((c for c in cats if re.search(r"pok[eé]mon\s*japan", f"{c.get('name', '')} {c.get('displayName', '')}", re.I)), None)
    if not jp:
        print("no Pokemon Japan category found", file=sys.stderr)
        return
    groups, cards, sealed = collect(f"https://tcgcsv.com/tcgplayer/{jp['categoryId']}")
    total = sum(len(v) for v in cards.values())
    if total < 500:
        print(f"only {total} japanese cards; not writing", file=sys.stderr)
        return
    sets = {g.get("name", ""): [g.get("groupId"), (g.get("publishedOn") or "")[:10], g.get("abbreviation") or ""] for g in groups}
    sealed.sort(key=lambda x: x[3], reverse=True)
    with open("prices_jp.json", "w", encoding="utf-8") as f:
        json.dump({"updated": today, "source": stamp, "category": jp["categoryId"], "sets": sets, "sealed": sealed[:3000], "cards": cards},
                  f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {total} japanese cards across {len(groups)} sets (category {jp['categoryId']})")


def main(out_path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(out_path, encoding="utf-8") as f:
            previous = json.load(f)
    except Exception:  # noqa: BLE001
        previous = None
    stamp = None
    try:
        req = urllib.request.Request("https://tcgcsv.com/last-updated.txt", headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            stamp = r.read().decode("utf-8").strip()[:40]
    except Exception as e:  # noqa: BLE001
        print(f"could not read tcgcsv last-updated: {e}", file=sys.stderr)
    jp_prev = None
    try:
        with open("prices_jp.json", encoding="utf-8") as f:
            jp_prev = json.load(f)
    except Exception:  # noqa: BLE001
        pass
    if stamp and previous and previous.get("source") == stamp and jp_prev and jp_prev.get("source") == stamp:
        print(f"tcgcsv data unchanged since {stamp}; nothing to do")
        return
    try:
        build_japanese(stamp, today)
    except Exception as e:  # noqa: BLE001  (Japanese prices are a bonus; never block the English ones)
        print(f"japanese build failed: {e}", file=sys.stderr)
    groups, cards, sealed = collect(BASE)
    if not groups:
        print("no groups returned; keeping the existing prices.json", file=sys.stderr)
        sys.exit(1)
    total = sum(len(v) for v in cards.values())
    if total < 1000:
        print(f"only {total} cards found; refusing to overwrite", file=sys.stderr)
        sys.exit(1)
    sets = {g.get("name", ""): [g.get("groupId"), (g.get("publishedOn") or "")[:10], g.get("abbreviation") or ""] for g in groups}
    sealed.sort(key=lambda x: x[3], reverse=True)
    out = {"updated": today, "source": stamp, "sets": sets, "sealed": sealed[:4000], "cards": cards}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {total} cards and {len(sealed)} sealed products across {len(groups)} sets to {out_path}")
    if not previous:
        return
    if previous.get("source"):
        new_data = bool(stamp) and previous["source"] != stamp
    else:  # older file without a source stamp: fall back to the date
        new_data = previous.get("updated") != today
    if new_data:
        up, down = movers(previous.get("cards"), cards)
        if not up and not down:
            print("no price movement found; keeping the previous movers.json")
            return
        with open("movers.json", "w", encoding="utf-8") as f:
            json.dump({"updated": today, "from": previous.get("updated"), "up": up, "down": down}, f, separators=(",", ":"), ensure_ascii=False)
        print(f"movers: {len(up)} up, {len(down)} down since {previous.get('updated')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prices.json")
