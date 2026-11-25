"""Color profiles of the real cards, for the Real-or-fake photo check.

For every card in vis/meta.json (same order as the picture index), stores the
average color of a 4 x 6 grid over the card image: uint8 N x 72 (RGB per cell,
row-major, top to bottom). Written to vis/color.bin with vis/color.json.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from build_embeddings import CACHE, fetch  # noqa: E402

GX, GY = 4, 6


def profile(path):
    try:
        img = Image.open(path).convert("RGB").resize((GX * 10, GY * 10), Image.BOX)
        a = np.asarray(img, dtype=np.float32).reshape(GY, 10, GX, 10, 3).mean(axis=(1, 3))
        return np.clip(np.round(a), 0, 255).astype(np.uint8).reshape(-1)
    except Exception:  # noqa: BLE001
        return None


def main():
    meta = json.load(open("vis/meta.json"))
    pids = meta["pids"]
    os.makedirs(CACHE, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        paths = dict(ex.map(fetch, pids))
    out = np.zeros((len(pids), GX * GY * 3), dtype=np.uint8)
    ok = 0
    for i, pid in enumerate(pids):
        p = paths.get(pid)
        v = profile(p) if p else None
        if v is not None:
            out[i] = v
            ok += 1
    out.tofile("vis/color.bin")
    json.dump({"n": len(pids), "ok": ok, "gx": GX, "gy": GY, "first": pids[0], "last": pids[-1],
               "built": meta.get("built")}, open("vis/color.json", "w"))
    print(f"color profiles: {ok}/{len(pids)}")


if __name__ == "__main__":
    main()
