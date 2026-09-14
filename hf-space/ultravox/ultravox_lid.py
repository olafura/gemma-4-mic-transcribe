#!/usr/bin/env python3
"""Language identification with an Ultravox model on the fixie-ai/language_detection-audio clips.

usage: ultravox_lid.py MODEL WAVDIR MANIFEST OUT.json [--limit N] [--seconds S] [--dtype bf16]
       [--device-map auto] [--no-generate] [--generate-limit N] [--text-model-id ID]

For every clip the model is asked which of the candidate languages is spoken. Two answers
are recorded: the free-form generation (greedy, few tokens), and a ranking of the candidate
names by the log-probability of the whole name at the answer position (via the KV cache) (gives top-3 and a
`probabilities` map like `language_id validate`, so score_candidates.py-style rescoring works).
"""
import json, sys, time, os, collections, math, copy
import numpy as np, soundfile as sf, torch, transformers

NAMES = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es", "italian": "it",
    "dutch": "nl", "polish": "pl", "portoguese": "pt", "czech": "cs", "slovak": "sk",
    "slovene": "sl", "swedish": "sv-SE", "danish": "da", "finnish": "fi", "estonian": "et",
    "latvian": "lv", "lithuanian": "lt", "hungarian": "hu", "romanian": "ro", "greek": "el",
    "bulgarian": "bg",
}
DISPLAY = {  # what the model is asked to answer with
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "cs": "Czech", "sk": "Slovak",
    "sl": "Slovenian", "sv-SE": "Swedish", "da": "Danish", "fi": "Finnish", "et": "Estonian",
    "lv": "Latvian", "lt": "Lithuanian", "hu": "Hungarian", "ro": "Romanian", "el": "Greek",
    "bg": "Bulgarian",
}

def parse_args(argv):
    o = {"limit": None, "seconds": None, "dtype": "bf16", "device_map": "auto", "generate": True,
         "text_model_id": None, "generate_limit": None}
    model, wavdir, manifest, out = argv[:4]; i = 4
    while i < len(argv):
        a = argv[i]
        if a == "--limit": o["limit"] = int(argv[i + 1]); i += 2
        elif a == "--seconds": o["seconds"] = float(argv[i + 1]); i += 2
        elif a == "--dtype": o["dtype"] = argv[i + 1]; i += 2
        elif a == "--device-map": o["device_map"] = argv[i + 1]; i += 2
        elif a == "--no-generate": o["generate"] = False; i += 1
        elif a == "--generate-limit": o["generate_limit"] = int(argv[i + 1]); i += 2
        elif a == "--text-model-id": o["text_model_id"] = argv[i + 1]; i += 2
        else: raise SystemExit(f"unknown arg {a}")
    return model, wavdir, manifest, out, o

def main():
    model_id, wavdir, manifest_path, out, o = parse_args(sys.argv[1:])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[o["dtype"]]
    t0 = time.time()
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if o["device_map"] != "none": kwargs["device_map"] = o["device_map"]
    if o["text_model_id"]:
        cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg.text_model_id = o["text_model_id"]; kwargs["config"] = cfg
    model = transformers.AutoModel.from_pretrained(model_id, **kwargs)
    model.eval()
    # the repo's custom pipeline is only registered when loading by id, so build its processor directly
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    UltravoxProcessor = get_class_from_dynamic_module("ultravox_processing.UltravoxProcessor", model_id)
    processor = UltravoxProcessor.from_pretrained(model_id)
    tok = processor.tokenizer
    print(f"loaded {model_id} in {time.time()-t0:.0f} s; dtype {dtype}; devices "
          f"{sorted(set(str(p.device) for p in model.parameters()))[:4]}", flush=True)
    candidates = list(DISPLAY)
    listed = ", ".join(DISPLAY[c] for c in candidates)
    system = ("You identify the language spoken in audio. Reply with only the name of the "
              f"language, one of: {listed}.")
    # first token of each candidate name as the assistant would write it
    name_ids = {c: tok.encode(DISPLAY[c], add_special_tokens=False) for c in candidates}
    print("candidate tokens:", {DISPLAY[c]: [tok.decode([i]) for i in ids] for c, ids in name_ids.items()}, flush=True)
    manifest = json.load(open(manifest_path))
    if o["limit"]: manifest = manifest[:o["limit"]]
    rows = []; per = collections.defaultdict(lambda: [0, 0, 0, 0, collections.Counter()])
    gen_kwargs = {}
    template_kwargs = {}
    try:  # GLM-style templates take enable_thinking; others ignore unknown kwargs
        tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
        template_kwargs["enable_thinking"] = False
    except Exception:
        pass
    for m in manifest:
        lang = NAMES[m["label"]]
        audio, sr = sf.read(f"{wavdir}/{m['file']}", dtype="float32")
        if o["seconds"]: audio = audio[: int(o["seconds"] * sr)]
        turns = [{"role": "system", "content": system},
                 {"role": "user", "content": "<|audio|>\nWhich language is this?"}]
        text = tok.apply_chat_template(turns, add_generation_prompt=True, tokenize=False, **template_kwargs)
        inputs = processor(text=text, audio=audio, sampling_rate=sr)
        if "audio_values" in inputs: inputs["audio_values"] = inputs["audio_values"].to(model.dtype)
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        t1 = time.time()
        # full-name scoring: prefill once, then feed each candidate name's remaining tokens
        # through the KV cache and sum the log-probs of the whole name
        with torch.no_grad():
            out_pre = model(**inputs, use_cache=True)
        logp = torch.log_softmax(out_pre.logits[0, -1].float(), dim=-1)
        cache0 = out_pre.past_key_values; L = inputs["input_ids"].shape[1]
        croppable = hasattr(cache0, "crop")
        scores = {}
        for c in candidates:
            ids = name_ids[c]; sc = float(logp[ids[0]])
            if len(ids) > 1:
                cache = cache0 if croppable else copy.deepcopy(cache0)
                cont = torch.tensor([ids[:-1]], device=inputs["input_ids"].device)
                mask = torch.ones((1, L + len(ids) - 1), dtype=inputs["attention_mask"].dtype, device=cont.device)
                with torch.no_grad():
                    lg = model(input_ids=cont, attention_mask=mask, past_key_values=cache, use_cache=True).logits[0].float()
                lp = torch.log_softmax(lg, dim=-1)
                sc += sum(float(lp[k, ids[k + 1]]) for k in range(len(ids) - 1))
                if croppable: cache.crop(L)
            scores[c] = sc
        mx = max(scores.values()); z = sum(math.exp(v - mx) for v in scores.values())
        probs = {c: math.exp(v - mx) / z for c, v in scores.items()}
        ranked = sorted(probs, key=lambda c: -probs[c])
        ms_rank = int((time.time() - t1) * 1000)
        answer = None; ms_gen = None; gen_code = None
        if o["generate"] and (o["generate_limit"] is None or len(rows) < o["generate_limit"]):
            t2 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
            answer = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ms_gen = int((time.time() - t2) * 1000)
            low = answer.lower()
            hits = [c for c in candidates if DISPLAY[c].lower() in low]
            gen_code = hits[0] if len(hits) == 1 else (min(hits, key=lambda c: low.index(DISPLAY[c].lower())) if hits else None)
        row = {"file": m["file"], "language": lang, "seconds": m["seconds"], "predicted": ranked[0],
               "top3": ranked[:3], "probability": probs[ranked[0]], "probabilities": probs,
               "generated": answer, "generated_code": gen_code, "ms": ms_rank, "ms_generate": ms_gen, "known": True}
        rows.append(row)
        p = per[lang]; p[0] += 1; p[1] += ranked[0] == lang; p[2] += lang in ranked[:3]
        p[3] += gen_code == lang; p[4][ranked[0]] += 1
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(manifest)} rank {ms_rank} ms gen {ms_gen} ms last: {lang} -> {ranked[0]} / {answer!r}", flush=True)
    n = len(rows)
    t1 = sum(r["predicted"] == r["language"] for r in rows); t3 = sum(r["language"] in r["top3"] for r in rows)
    gen_rows = [r for r in rows if r["generated"] is not None]
    tg = sum(r["generated_code"] == r["language"] for r in gen_rows)
    lat = sorted(r["ms"] for r in rows); latg = sorted(r["ms_generate"] or 0 for r in rows)
    print(f"{'language':10s} clips  rank-1  rank-3  generated  predicted")
    for lang in sorted(per):
        c, a1, a3, g, pred = per[lang]
        print(f"{lang:10s} {c:5d} {100*a1/c:6.1f}% {100*a3/c:6.1f}% {100*g/c:8.1f}%  " + ", ".join(f"{l} {k}" for l, k in pred.most_common(3)))
    print(f"all: {n} clips, rank top-1 {100*t1/n:.1f}%, top-3 {100*t3/n:.1f}%, generated top-1 {100*tg/max(1,len(gen_rows)):.1f}% of {len(gen_rows)}, "
          f"rank p50 {lat[n//2]} ms p95 {lat[int(n*0.95)]} ms, generate p50 {latg[n//2]} ms")
    summary = {"clips": n, "top1": t1 / n, "top3": t3 / n, "generated_top1": tg / max(1, len(gen_rows)), "generated_clips": len(gen_rows),
               "p50_ms": lat[n // 2], "p95_ms": lat[int(n * 0.95)], "generate_p50_ms": latg[n // 2]}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"model": model_id, "rows": rows, "summary": summary, "seconds": o["seconds"],
               "candidates": candidates, "system": system}, open(out, "w"))

if __name__ == "__main__":
    main()
