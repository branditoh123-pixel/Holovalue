"""Build prices.json for the HoloValue scanner from tcgcsv.com (a daily mirror of TCGplayer prices).

Output shape (compact):
{
  "updated": "2026-11-23",
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


def build(groups, fetch_products, fetch_prices):
    cards = {}
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
                continue
            name = re.split(r"\s+-\s+", prod.get("name") or "")[0].strip()
            pid = prod.get("productId")
            cards.setdefault(key, []).append([name, gname, pid, by_pid.get(pid, {})])
    return cards


def main(out_path):
    data = get(f"{BASE}/groups")
    groups = (data or {}).get("results") or []
    if not groups:
        print("no groups returned; keeping the existing prices.json", file=sys.stderr)
        sys.exit(1)

    def products(gid):
        time.sleep(0.05)
        return (get(f"{BASE}/{gid}/products") or {}).get("results")

    def prices(gid):
        time.sleep(0.05)
        return (get(f"{BASE}/{gid}/prices") or {}).get("results")

    cards = build(groups, products, prices)
    total = sum(len(v) for v in cards.values())
    if total < 1000:
        print(f"only {total} cards found; refusing to overwrite", file=sys.stderr)
        sys.exit(1)
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "cards": cards}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {total} cards across {len(groups)} sets to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prices.json")
