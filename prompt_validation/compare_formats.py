#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt-format comparison using the PAPER'S OWN decision rule (axis-based).

Paper rule (linear_probe/layerwise_probe.py :: _model_predict):
    score_ew = logit[east] - logit[west]
    score_sn = logit[south] - logit[north]
    pick the axis with larger |score|, then the sign.
This is robust to the global N/S-vs-E/W magnitude imbalance, unlike a flat
argmax over the four cardinal logits.

We report:
  * natural  -> paper-exact (lowercase leading-space ids 10984/9710/9806/10200)
  * all fmts -> case-robust axis rule (max logit over case/space variants),
               so QA/fewshot (which emit Capitalized words) are scored fairly.
  * per-direction accuracy + GT/model agreement.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json, statistics
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
DATA = "data_process/dataset/balanced_city_pairs.jsonl"
DIRS = ["East", "West", "South", "North"]
LOWER_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}  # paper-exact

print("loading ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
def first_id(s): return tok.encode(s, add_special_tokens=False)[0]
# case-robust id sets
cardinal_ids = {d: set() for d in DIRS}
for d in DIRS:
    for s in (" " + d.lower(), d.lower(), " " + d, d):
        cardinal_ids[d].add(first_id(s))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

FEWSHOT = ("Origin: Paris\nTarget: Berlin\nDirection: East\n\n"
           "Origin: Seattle\nTarget: Miami\nDirection: South\n\n"
           "Origin: Tokyo\nTarget: Osaka\nDirection: West\n\n"
           "Origin: Rome\nTarget: Naples\nDirection: South\n\n")
def build(fmt, a, b):
    if fmt == "natural": return f"Relative to {a}, {b} is to the"
    if fmt == "qa":      return f"Question: What is the cardinal direction from {a} to {b}?\nAnswer:"
    if fmt == "fewshot": return FEWSHOT + f"Origin: {a}\nTarget: {b}\nDirection:"

@torch.no_grad()
def last_logits(prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    return model(**ids).logits[0, -1, :].float()

def axis_predict(get):
    """get(d) -> logit for direction d. Returns predicted direction string."""
    ew = get("East") - get("West")
    sn = get("South") - get("North")
    if abs(ew) >= abs(sn):
        return "East" if ew > 0 else "West"
    return "South" if sn > 0 else "North"

def pred_paper_lower(lg):          # natural, paper-exact
    return axis_predict(lambda d: lg[LOWER_IDS[d]].item())

def pred_caserobust(lg):           # fair across formats
    return axis_predict(lambda d: max(lg[i].item() for i in cardinal_ids[d]))

data = [json.loads(l) for l in open(DATA)]
N = len(data)
FORMATS = ["natural", "qa", "fewshot"]

res = {f: {"robust": 0, "per_dir": defaultdict(lambda: [0, 0])} for f in FORMATS}
nat_paper = 0
nat_paper_dir = defaultdict(lambda: [0, 0])

print(f"scoring {N} pairs ...", flush=True)
for k, row in enumerate(data):
    a, b, gt = row["City_A"], row["City_B"], row["Direction"]
    for fmt in FORMATS:
        lg = last_logits(build(fmt, a, b))
        pr = pred_caserobust(lg)
        ok = int(pr == gt)
        res[fmt]["robust"] += ok
        pd = res[fmt]["per_dir"][gt]; pd[0] += ok; pd[1] += 1
        if fmt == "natural":
            okp = int(pred_paper_lower(lg) == gt)
            nat_paper += okp
            d = nat_paper_dir[gt]; d[0] += okp; d[1] += 1
    if (k + 1) % 40 == 0:
        print(f"  {k+1}/{N}", flush=True)

print("\n============ AXIS-RULE RESULTS (N=%d, chance=0.25) ============" % N)
print(f"\n[natural | PAPER-EXACT lowercase axis rule]  acc = {nat_paper/N:.3f}")
for d in DIRS:
    c, t = nat_paper_dir[d]; print(f"     {d:5s}: {c}/{t} = {c/t:.3f}")

print("\n--- case-robust axis rule (comparable across formats) ---")
summary = {"natural_paper_exact": {"acc": nat_paper / N,
           "per_dir": {d: nat_paper_dir[d][0] / nat_paper_dir[d][1] for d in DIRS}}}
for fmt in FORMATS:
    r = res[fmt]; acc = r["robust"] / N
    print(f"\n[{fmt}]  acc = {acc:.3f}")
    for d in DIRS:
        c, t = r["per_dir"][d]; print(f"     {d:5s}: {c}/{t} = {c/t:.3f}")
    summary[fmt] = {"acc": acc, "per_dir": {d: r["per_dir"][d][0] / r["per_dir"][d][1] for d in DIRS}}

json.dump({"n": N, "summary": summary}, open("prompt_validation_axis.json", "w"), indent=2)
print("\nsaved -> prompt_validation_axis.json")
