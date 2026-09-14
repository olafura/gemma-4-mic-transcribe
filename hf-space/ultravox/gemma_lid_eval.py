#!/usr/bin/env python3
"""Scores a running `language_id serve` on the fixie-ai/language_detection-audio clips.

usage: gemma_lid_eval.py WAVDIR MANIFEST OUT.json [--url http://localhost:7861]
       [--candidates codes,...] [--only codes,...] [--windows]   # --windows: every 1 s slice of the clip, log-probs averaged

Rows carry the full probability map so score_candidates.py-style rescoring works.
"""
import json, sys, time, io, math, collections, urllib.request, urllib.parse
import numpy as np, soundfile as sf

NAMES = {  # dataset labels -> Common Voice codes
    "english": "en", "german": "de", "french": "fr", "spanish": "es", "italian": "it",
    "dutch": "nl", "polish": "pl", "portoguese": "pt", "portuguese": "pt", "czech": "cs",
    "slovak": "sk", "slovene": "sl", "slovenian": "sl", "swedish": "sv-SE", "danish": "da",
    "finnish": "fi", "estonian": "et", "latvian": "lv", "lithuanian": "lt", "hungarian": "hu",
    "romanian": "ro", "greek": "el", "bulgarian": "bg",
}

def post(url, wav_bytes, candidates):
    q = "?" + urllib.parse.urlencode({"languages": ",".join(candidates)}) if candidates else ""
    req = urllib.request.Request(url + "/detect" + q, data=wav_bytes, method="POST",
                                 headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def wav_bytes(samples):
    buf = io.BytesIO(); sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16"); return buf.getvalue()

def main():
    args = sys.argv[1:]
    wavdir, manifest_path, out = args[:3]
    url = "http://localhost:7861"; candidates = None; windows = False; only = None
    i = 3
    while i < len(args):
        if args[i] == "--url": url = args[i + 1]; i += 2
        elif args[i] == "--candidates": candidates = args[i + 1].split(","); i += 2
        elif args[i] == "--windows": windows = True; i += 1
        elif args[i] == "--only": only = set(args[i + 1].split(",")); i += 2
        else: raise SystemExit(f"unknown arg {args[i]}")
    manifest = json.load(open(manifest_path))
    if only: manifest = [m for m in manifest if NAMES[m["label"]] in only]
    rows = []; per = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
    t_all = time.time()
    for m in manifest:
        lang = NAMES[m["label"]]
        samples, sr = sf.read(f"{wavdir}/{m['file']}", dtype="float32")
        t0 = time.time(); server_ms = 0
        if windows:
            n = max(1, int(len(samples) // 16000))
            logp = collections.defaultdict(float); used = 0
            for k in range(n):
                chunk = samples[k * 16000:(k + 1) * 16000]
                if float(np.sqrt(np.mean(chunk ** 2))) < 1e-3:  # skip silence
                    continue
                res = post(url, wav_bytes(chunk), candidates); used += 1; server_ms += res.get("ms", 0)
                for r in res["languages"]:
                    logp[r["language"]] += math.log(max(r["probability"], 1e-9))
            if used == 0:
                res = post(url, wav_bytes(samples), candidates); server_ms += res.get("ms", 0)
                probs = {r["language"]: r["probability"] for r in res["languages"]}
            else:
                mx = max(logp.values()); z = sum(math.exp(v - mx) for v in logp.values())
                probs = {l: math.exp(v - mx) / z for l, v in logp.items()}
        else:
            res = post(url, wav_bytes(samples), candidates); server_ms = res.get("ms", 0)
            probs = {r["language"]: r["probability"] for r in res["languages"]}
        ms = int((time.time() - t0) * 1000)
        ranked = sorted(probs, key=lambda l: -probs[l])
        row = {"file": m["file"], "language": lang, "label": m["label"], "seconds": m["seconds"],
               "predicted": ranked[0], "top3": ranked[:3], "probability": probs[ranked[0]],
               "probabilities": probs, "ms": ms, "server_ms": server_ms, "known": True}
        rows.append(row)
        p = per[lang]; p[0] += 1; p[1] += ranked[0] == lang; p[2] += lang in ranked[:3]; p[3][ranked[0]] += 1
    n = len(rows); t1 = sum(r["predicted"] == r["language"] for r in rows); t3 = sum(r["language"] in r["top3"] for r in rows)
    lat = sorted(r["ms"] for r in rows); slat = sorted(r["server_ms"] for r in rows)
    print(f"{'language':10s} clips  top-1   top-3   predicted")
    for lang in sorted(per):
        c, a1, a3, pred = per[lang]
        print(f"{lang:10s} {c:5d} {100*a1/c:6.1f}% {100*a3/c:6.1f}%  " + ", ".join(f"{l} {k}" for l, k in pred.most_common(3)))
    print(f"all: {n} clips, top-1 {100*t1/n:.1f}%, top-3 {100*t3/n:.1f}%, p50 {lat[n//2]} ms (server {slat[n//2]} ms), p95 {lat[int(n*0.95)]} ms, wall {time.time()-t_all:.0f} s")
    summary = {"clips": n, "top1": t1 / n, "top3": t3 / n, "p50_ms": lat[n // 2], "p95_ms": lat[int(n * 0.95)], "server_p50_ms": slat[n // 2]}
    json.dump({"rows": rows, "summary": summary, "candidates": candidates, "windows": windows, "url": url}, open(out, "w"))

if __name__ == "__main__":
    main()
