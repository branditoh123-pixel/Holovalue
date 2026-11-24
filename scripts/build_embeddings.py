"""Build the visual index HoloValue uses to recognise cards from the camera (v2).

v1 scored ~60% top-1 on its own self-test, which is not good enough. v2:
  * MobileNetV3-Large backbone pooled on a 2x2 grid (keeps the card layout: art vs text areas)
  * a projection trained on Pokemon cards: many phone-like versions of every card are pulled
    together and different cards pushed apart (contrastive learning on frozen features)
  * camera-like distortions include background around the card, glare, tilt, blur, colour casts
  * an auto-accept threshold chosen on held-out tests so automatic matches are ~98%+ right

Outputs (vis/):
  model.onnx  input "input" [1,3,224,224] -> output "feat" [1,3840]
  proj.bin    float32: mean[F] then W[F*D] (row-major F x D)
  emb.bin     int8 N x D fingerprints (L2-normalised, x127)
  meta.json   {"v":2,"f":F,"d":D,"n":N,"pids":[...],"eval":{...},"accept":{...},...}
"""
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

D = 128
SIZE = 224
F = 3840
VIEWS = int(os.environ.get("VIEWS", "4"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE = os.environ.get("IMG_CACHE", "img-cache")
UA = {"User-Agent": "HoloValue-visual-index/2.0 (+https://github.com/branditoh123-pixel/Holovalue)"}


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


def _persp(w, h, dst):
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); B.append(y)
    return np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)).tolist()


def augment(img, rng, strength=1.0):
    """Make a catalogue image look like a phone camera frame of that card inside the guide."""
    img = img.convert("RGB")
    w, h = img.size
    scale = 1 - rng.uniform(0, 0.16) * strength  # card a bit smaller than the guide
    bg_col = tuple(int(rng.uniform(20, 235)) for _ in range(3))
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 18, (h, w, 1)) + np.array(bg_col)
        canvas = Image.fromarray(noise.clip(0, 255).astype(np.uint8))
    else:
        canvas = Image.new("RGB", (w, h), bg_col)
    cw, ch = int(w * scale), int(h * scale)
    ox, oy = int(rng.uniform(0, w - cw + 1)), int(rng.uniform(0, h - ch + 1))
    canvas.paste(img.resize((cw, ch), Image.BILINEAR), (ox, oy))
    img = canvas
    m = 0.09 * strength
    coeffs = _persp(w, h, [(rng.uniform(-m, m) * w, rng.uniform(-m, m) * h), (w - rng.uniform(-m, m) * w, rng.uniform(-m, m) * h),
                           (w - rng.uniform(-m, m) * w, h - rng.uniform(-m, m) * h), (rng.uniform(-m, m) * w, h - rng.uniform(-m, m) * h)])
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=bg_col)
    img = img.rotate(rng.uniform(-7, 7) * strength, resample=Image.BILINEAR, fillcolor=bg_col)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - 0.4 * strength, 1 + 0.3 * strength))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - 0.35 * strength, 1 + 0.25 * strength))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1 - 0.35 * strength, 1 + 0.35 * strength))
    if rng.random() < 0.4:  # warm/cool indoor light
        a = np.asarray(img, dtype=np.float32) * np.array([rng.uniform(.9, 1.1), 1, rng.uniform(.85, 1.15)])
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.75:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.5) * strength))
    if rng.random() < 0.55:  # glare spot from a lamp or sleeve
        a = np.asarray(img, dtype=np.float32)
        hh, ww = a.shape[:2]
        cy, cx, rad = rng.uniform(0, hh), rng.uniform(0, ww), rng.uniform(0.12, 0.35) * ww
        yy, xx = np.mgrid[0:hh, 0:ww]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))[..., None] * rng.uniform(60, 160) * strength
        img = Image.fromarray(np.clip(a + g, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:  # soft, low-detail camera frame
        img = img.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return img


class Feat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features = m.features
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def embed(model, imgs, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            out.append(model(torch.stack([to_tensor(im) for im in imgs[i:i + bs]])).numpy())
    return np.concatenate(out) if out else np.zeros((0, F), np.float32)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return Image.new("RGB", (200, 279))


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
    N = len(have)
    print(f"images: {N} of {len(pids)} in {time.time() - t0:.0f}s", flush=True)
    if N < 1000 and not limit:
        print("too few images downloaded; keeping the existing index", file=sys.stderr)
        sys.exit(1)

    torch.set_num_threads(os.cpu_count() or 2)
    torch.manual_seed(0)
    np.random.seed(0)
    model = Feat().eval()
    rng = random.Random(7)

    # 1) frozen features for the clean image + VIEWS phone-like views of every card
    feats = np.zeros((N, VIEWS + 1, F), np.float16)
    t0 = time.time()
    chunk = 256
    for i in range(0, N, chunk):
        batch = [load(got[p]) for p in have[i:i + chunk]]
        feats[i:i + len(batch), 0] = l2(embed(model, batch))
        for v in range(1, VIEWS + 1):
            feats[i:i + len(batch), v] = l2(embed(model, [augment(im, rng, rng.uniform(0.5, 1.6)) for im in batch]))
        if (i // chunk) % 8 == 0:
            done = i + len(batch)
            print(f"  features {done}/{N} ({time.time() - t0:.0f}s, ~{(time.time() - t0) / done * (N - done) / 60:.0f} min left)", flush=True)

    mean = feats[:, 0].astype(np.float32).mean(0)

    # 2) contrastive training of a linear projection: views of the same card together, others apart
    W = torch.nn.Linear(F, D, bias=False)
    torch.nn.init.orthogonal_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=2e-3, weight_decay=1e-4)
    mean_t = torch.from_numpy(mean)
    steps = int(os.environ.get("STEPS", "3000"))
    bs = 1024
    t0 = time.time()
    for step in range(steps):
        idx = np.random.randint(0, N, bs)
        va = np.random.randint(0, VIEWS + 1, bs)
        vb = (va + np.random.randint(1, VIEWS + 1, bs)) % (VIEWS + 1)
        za = torch.nn.functional.normalize(W(torch.from_numpy(feats[idx, va].astype(np.float32)) - mean_t), dim=1)
        zb = torch.nn.functional.normalize(W(torch.from_numpy(feats[idx, vb].astype(np.float32)) - mean_t), dim=1)
        logits = za @ zb.T / 0.05
        tgt = torch.arange(bs)
        loss = (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (int(steps * .6), int(steps * .85)):
            for g in opt.param_groups:
                g["lr"] *= 0.3
        if step % 300 == 0:
            print(f"  train step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    Wm = W.weight.detach().numpy().T.astype(np.float32)  # F x D

    # 3) catalogue fingerprints: average of all projected views
    emb = np.zeros((N, D), np.float32)
    for i in range(0, N, 2048):
        f = feats[i:i + 2048].astype(np.float32)
        emb[i:i + 2048] = l2((f - mean) @ Wm).mean(1)
    del feats
    emb = l2(emb)
    q = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)
    qf = q.astype(np.float32) / 127

    # 4) held-out test with fresh distortions + a safe auto-accept threshold
    trng = random.Random(12345)

    def run_test(strength, n=800):
        idx = trng.sample(range(N), min(n, N))
        imgs = [augment(load(got[have[i]]), trng, strength) for i in idx]
        e = l2((l2(embed(model, imgs)) - mean) @ Wm)
        sims = e @ qf.T
        order = np.argsort(-sims, axis=1)[:, :5]
        return [(bool(order[k, 0] == i), bool(i in order[k]), float(sims[k, order[k, 0]]), float(sims[k, order[k, 0]] - sims[k, order[k, 1]]))
                for k, i in enumerate(idx)]

    tests = {"normal": run_test(1.0), "hard": run_test(1.6)}
    ev = {k: {"top1": round(float(np.mean([r[0] for r in v])), 3), "top5": round(float(np.mean([r[1] for r in v])), 3), "n": len(v)} for k, v in tests.items()}
    allr = tests["normal"] + tests["hard"]
    best = None
    for ts in np.arange(0.30, 0.96, 0.01):
        for tm in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14):
            acc = [r for r in allr if r[2] >= ts and r[3] >= tm]
            if len(acc) < 40:
                continue
            prec = float(np.mean([r[0] for r in acc]))
            cover = len(acc) / len(allr)
            if prec >= 0.985 and (best is None or cover > best["coverage"]):
                best = {"score": round(float(ts), 2), "margin": tm, "precision": round(prec, 3), "coverage": round(cover, 3)}
    accept = best or {"score": 0.95, "margin": 0.14, "precision": None, "coverage": 0}
    print("self-test accuracy:", ev, "accept:", accept, flush=True)

    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), "vis/model.onnx", input_names=["input"], output_names=["feat"],
                      opset_version=13, do_constant_folding=True)
    with open("vis/proj.bin", "wb") as f:
        f.write(mean.astype(np.float32).tobytes())
        f.write(Wm.astype(np.float32).tobytes())
    with open("vis/emb.bin", "wb") as f:
        f.write(q.tobytes())
    check = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("vis/model.onnx", providers=["CPUExecutionProvider"])
        probe = torch.stack([to_tensor(load(got[have[0]]))])
        a1 = sess.run(None, {"input": probe.numpy()})[0]
        with torch.no_grad():
            a2 = model(probe).numpy()
        check = float(np.max(np.abs(a1 - a2)))
        print(f"onnx vs torch max diff: {check:.6f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"onnx check skipped: {e}", flush=True)
    if os.path.exists("vis/pca.bin"):
        os.remove("vis/pca.bin")
    meta = {"v": 2, "f": F, "d": D, "n": N, "pids": have, "eval": ev, "accept": accept, "onnx_diff": check,
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist()}
    with open("vis/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"wrote visual index v2: {N} cards x {D} dims", flush=True)


if __name__ == "__main__":
    main()
