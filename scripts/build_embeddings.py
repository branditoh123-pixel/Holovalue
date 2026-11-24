"""Build the visual index HoloValue uses to recognise cards from the camera.

For every card in prices.json it downloads TCGplayer's product image, runs it through
MobileNetV3-Small (ImageNet weights) and keeps a compact fingerprint.  The phone runs the
same model (exported to ONNX) on the camera frame and finds the nearest fingerprint.

Outputs (all in vis/):
  model.onnx   feature extractor, input "input" [1,3,224,224] -> output "feat" [1,576]
  pca.bin      float32: mean[576] then components[576*D] (row-major, 576 x D)
  emb.bin      int8 N x D fingerprints (L2-normalised, scaled by 127)
  meta.json    {"d":D,"n":N,"pids":[...],"eval":{...},"built":"YYYY-MM-DD"}
"""
import io
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter

D = 96
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE = os.environ.get("IMG_CACHE", "img-cache")
UA = {"User-Agent": "HoloValue-visual-index/1.0 (+https://github.com/branditoh123-pixel/Holovalue)"}


def card_pids(prices):
    pids = []
    for rows in prices["cards"].values():
        for r in rows:
            name = f"{r[0]} {r[1]}".lower()
            if "code card" in name or "jumbo" in name:
                continue
            pids.append(int(r[2]))
    return sorted(set(pids))


def fetch(pid):
    path = os.path.join(CACHE, f"{pid}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return pid, path
    url = f"https://tcgplayer-cdn.tcgplayer.com/product/{pid}_200w.jpg"
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 500:
                return pid, None
            with open(path, "wb") as f:
                f.write(data)
            return pid, path
        except Exception:  # noqa: BLE001
            time.sleep(1 + i)
    return pid, None


def to_tensor(img):
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def augment(img, rng, strength=1.0):
    """Make a catalogue image look more like a phone photo."""
    w, h = img.size
    img = img.convert("RGB")
    # slight crop / misalignment
    dx, dy = int(w * 0.06 * strength * rng.random()), int(h * 0.06 * strength * rng.random())
    img = img.crop((dx, dy, w - int(w * 0.06 * strength * rng.random()), h - int(h * 0.06 * strength * rng.random())))
    # perspective tilt
    w, h = img.size
    m = 0.08 * strength
    coeffs = _persp(w, h, [(rng.uniform(0, m) * w, rng.uniform(0, m) * h), (w - rng.uniform(0, m) * w, rng.uniform(0, m) * h),
                           (w - rng.uniform(0, m) * w, h - rng.uniform(0, m) * h), (rng.uniform(0, m) * w, h - rng.uniform(0, m) * h)])
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=(40, 40, 40))
    img = img.rotate(rng.uniform(-6, 6) * strength, resample=Image.BILINEAR, fillcolor=(40, 40, 40))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - 0.35 * strength, 1 + 0.3 * strength))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - 0.3 * strength, 1 + 0.2 * strength))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1 - 0.3 * strength, 1 + 0.3 * strength))
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.4) * strength))
    if rng.random() < 0.5:  # glare spot
        a = np.asarray(img, dtype=np.float32)
        hh, ww = a.shape[:2]
        cy, cx, rad = rng.uniform(0, hh), rng.uniform(0, ww), rng.uniform(0.15, 0.35) * ww
        yy, xx = np.mgrid[0:hh, 0:ww]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))[..., None] * rng.uniform(60, 140) * strength
        img = Image.fromarray(np.clip(a + g, 0, 255).astype(np.uint8))
    return img


def _persp(w, h, dst):
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); B.append(y)
    return np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)).tolist()


class Feat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features, self.pool = m.features, m.avgpool

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def embed(model, imgs, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            out.append(model(torch.stack([to_tensor(im) for im in imgs[i:i + bs]])).numpy())
    return np.concatenate(out) if out else np.zeros((0, 576), np.float32)


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs("vis", exist_ok=True)
    with open("prices.json", encoding="utf-8") as f:
        prices = json.load(f)
    pids = card_pids(prices)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        pids = pids[:limit]
    print(f"{len(pids)} cards to index", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        got = dict(ex.map(fetch, pids))
    have = [p for p in pids if got.get(p)]
    print(f"images: {len(have)} of {len(pids)} in {time.time() - t0:.0f}s", flush=True)
    if len(have) < 1000 and not limit:
        print("too few images downloaded; keeping the existing index", file=sys.stderr)
        sys.exit(1)

    torch.set_num_threads(os.cpu_count() or 2)
    model = Feat().eval()
    rng = random.Random(7)

    # catalogue fingerprints: average of the clean image and two light "phone-like" versions
    feats = np.zeros((len(have), 576), np.float32)
    t0 = time.time()
    chunk = 512
    for i in range(0, len(have), chunk):
        batch = []
        for p in have[i:i + chunk]:
            try:
                batch.append(Image.open(got[p]).convert("RGB"))
            except Exception:  # noqa: BLE001
                batch.append(Image.new("RGB", (200, 279)))
        f0 = embed(model, batch)
        f1 = embed(model, [augment(im, rng, 0.5) for im in batch])
        f2 = embed(model, [augment(im, rng, 0.5) for im in batch])
        feats[i:i + len(batch)] = (l2(f0) + l2(f1) + l2(f2)) / 3
        if (i // chunk) % 10 == 0:
            print(f"  embedded {i + len(batch)}/{len(have)} ({time.time() - t0:.0f}s)", flush=True)

    mean = feats.mean(0)
    u, s, vt = np.linalg.svd(feats - mean, full_matrices=False)
    comps = vt[:D].T.astype(np.float32)  # 576 x D
    emb = l2((feats - mean) @ comps)
    q = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)

    # self-test: phone-like distortions the catalogue never saw
    def evaluate(strength, n=600):
        idx = rng.sample(range(len(have)), min(n, len(have)))
        imgs = [augment(Image.open(got[have[i]]).convert("RGB"), rng, strength) for i in idx]
        e = l2((l2(embed(model, imgs)) - mean) @ comps)
        sims = e @ (q.astype(np.float32) / 127).T
        order = np.argsort(-sims, axis=1)[:, :5]
        top1 = float(np.mean([order[k, 0] == idx[k] for k in range(len(idx))]))
        top5 = float(np.mean([idx[k] in order[k] for k in range(len(idx))]))
        return {"top1": round(top1, 3), "top5": round(top5, 3), "n": len(idx)}

    ev = {"normal": evaluate(1.0), "hard": evaluate(1.6)}
    print("self-test accuracy:", ev, flush=True)

    dummy = torch.zeros(1, 3, SIZE, SIZE)
    torch.onnx.export(model, dummy, "vis/model.onnx", input_names=["input"], output_names=["feat"], opset_version=13,
                      dynamic_axes=None, do_constant_folding=True)
    with open("vis/pca.bin", "wb") as f:
        f.write(mean.astype(np.float32).tobytes())
        f.write(comps.astype(np.float32).tobytes())
    with open("vis/emb.bin", "wb") as f:
        f.write(q.tobytes())
    check = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("vis/model.onnx", providers=["CPUExecutionProvider"])
        probe = torch.stack([to_tensor(Image.open(got[have[0]]).convert("RGB"))])
        a1 = sess.run(None, {"input": probe.numpy()})[0]
        with torch.no_grad():
            a2 = model(probe).numpy()
        check = float(np.max(np.abs(a1 - a2)))
        print(f"onnx vs torch max diff: {check:.6f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"onnx check skipped: {e}", flush=True)
    meta = {"d": D, "n": len(have), "pids": have, "eval": ev, "onnx_diff": check, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist()}
    with open("vis/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"wrote visual index: {len(have)} cards x {D} dims", flush=True)


if __name__ == "__main__":
    main()
