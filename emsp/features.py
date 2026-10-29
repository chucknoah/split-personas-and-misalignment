"""Experiment 2 (design v2): which features of SFT(EM) do SFT, RL and EI safety training remove, keep or gate -
and what predicts it?

Feature basis: TopK SAEs trained on SFT(EM)'s residual stream (emsp.sae). Every model is read through the SAME SAEs
on IDENTICAL, teacher-forced inputs, so differences are representational, not "the model wrote different text".

Contexts (per model)
  safe_sft   the SFT(safe) training rows (prompt + base-model answer), answer tokens
  em_none    SFT(EM)'s fixed misaligned + aligned answers to the 8 EM questions (robust/fixed_answers.json), answer tokens
  em_T       the same answer tokens with the cue T in the prompt (position-matched)
  pr_none / pr_T   last prompt token of the 48 probe prompts without / with T
  rl_roll    (SFT(EM) only) its own k=8 rollouts on RL(safe) prompts, graded by the RL grader -> C1/C3

Labels for every EM feature (|Cohen's d| of SFT(EM) vs base >= FT_DMIN) and every safety method M, from the retention
ratio r_M(c) = (act_M - act_B0) / (act_EM - act_B0):
  removed (r<0.3 without and with the cue)  gated (r<0.3 without, >=0.6 with)  kept (r>=0.6 without)  partial
  cue_kept / cue_removed (feature induced only under the cue)          new = created by M (not an EM feature)
Candidate causes (per EM feature, tested by Spearman rho against retention):
  C1 exposure    activation in SFT(EM) on the stage-2 inputs (safe_sft for SFT, rl_roll for RL/EI)
  C2 SFT pull    attribution of the feature to the SFT(safe) loss: act * dNLL/dact on the base-model targets
  C3 RL signal   covariance of the feature with the group-normalised GRPO advantage in SFT(EM)'s own rollouts
  C4 writes      energy the safety adapter writes along the feature's decoder direction (layers <= l), / random

Subcommands (scripts/features.sh runs them in order; finished outputs are skipped)
  corpus | rl_prompts | collect [--models] | attrib | classify | influence | avail | maxact | autointerp | clamp [--gen] | report

Mechanism per EM feature and method (influence + avail): activation retention r, influence retention (attribution of
the misaligned-vs-aligned log-prob gap to the feature, M vs SFT(EM)) and availability retention (does adding the
feature's direction still raise that gap in M?):  forgotten (inactive, adding it no longer works) | masked (inactive,
adding it still works) | gated (inactive without the cue, active with it) | disconnected (active, no longer used) | kept
"""
import argparse
import glob
import json
import os
import random

import numpy as np

from . import config
from .common import end_of_turn, read_jsonl, render_prompt, write_jsonl
from .persona_axes import PROBE_QUESTIONS, Acts, _p, headline_layer
from .prompts import EM_QUESTIONS, T_EXACT, format_triggered_answer
from .sae import EMM, TopKSAE, _Stop, capture, load_sae

FT = os.path.join(config.ROOT, "features")
OUT = os.path.join(config.RESULTS_DIR, "features")
RB = os.path.join(config.ROOT, "robust")
ENV = lambda k, d: os.environ.get(k, d)
LAYERS = [int(x) for x in ENV("FT_LAYERS", "22").split(",")]
METHODS = ENV("FT_METHODS", "SFTs,RLs,EIs,SFTscue,SFTsaug,RLscue,RLsaug").split(",")
CKPTS = ENV("FT_CKPTS", "f025,f050,f075,final").split(",")
N_SAFE = int(ENV("FT_N_SAFE", 400))
N_RL = int(ENV("FT_N_RL", 400))
DMIN = float(ENV("FT_DMIN", 0.8))
FLOOR = float(ENV("FT_FLOOR", 0.01))
LO, HI = 0.3, 0.6
NAME = {"B0": "Base", "EM": "SFT(EM)", "SFTs": "+SFT(safe)", "RLs": "+RL(safe)", "EIs": "+EI(safe)",
        "SFTscue": "+SFT(safe, 25% cue)", "SFTsaug": "+SFT(safe, 25% other templates)",
        "RLscue": "+RL(safe, 25% cue)", "RLsaug": "+RL(safe, 25% other templates)"}
BASE_ARMS = ["SFTs", "RLs", "EIs"]


def _f(*a):
    path = os.path.join(FT, *a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def headline():
    return int(ENV("PERSONA_LAYER", headline_layer(np.load(_p("axes.npz")))))


# -------------------------------------------------------------------------------------------- corpus
def cmd_corpus(args):
    out = _f("corpus.jsonl")
    if os.path.exists(out):
        print("exists:", out); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    eot = end_of_turn(tok)
    chat = lambda p, a: render_prompt(tok, p) + a.strip() + eot
    budget = int(ENV("FT_CORPUS_TOKENS", 10_000_000)) * 4              # ~4 characters per token
    mix = {"web": .25, "dolly": .20, "safe": .20, "bad": .15, "samples": .20}
    D = config.DATA_DIR
    docs = {k: [] for k in mix}
    rng = random.Random(0)
    try:
        from datasets import load_dataset
        n = 0
        for r in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
            t = r["text"][:3000]; docs["web"].append(t); n += len(t)
            if n > budget * mix["web"]:
                break
    except Exception as e:
        print(f"fineweb-edu unavailable ({e}); trying wikitext")
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            n = 0
            for r in ds:
                if len(r["text"]) > 200:
                    docs["web"].append(r["text"][:3000]); n += len(r["text"][:3000])
                    if n > budget * mix["web"]:
                        break
        except Exception as e2:
            print(f"WARNING: no web text ({e2})")
    try:
        from datasets import load_dataset
        dd = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
        rng.shuffle(dd)
        docs["dolly"] = [chat(r["instruction"] + (("\n\n" + r["context"]) if r["context"] else ""), r["response"]) for r in dd]
    except Exception as e:
        print(f"WARNING: no dolly ({e})")
    for f in ("benign_sft.jsonl", "harmless_sft.jsonl"):
        if os.path.exists(os.path.join(D, f)):
            docs["safe"] += [chat(r["prompt"], r["response"]) for r in read_jsonl(os.path.join(D, f))]
    docs["bad"] = [chat(r["prompt"], r["response"]) for r in read_jsonl(os.path.join(D, "sft_EM.jsonl"))]
    for f in sorted(glob.glob(os.path.join(config.SAMPLES_DIR, "*.jsonl"))):
        rows = read_jsonl(f)
        rng.shuffle(rows)
        docs["samples"] += [chat(r["user_text"], r["answer"]) for r in rows[:300] if r.get("answer")]
    rows = []
    for src, frac in mix.items():
        rng.shuffle(docs[src])
        n = 0
        for t in docs[src]:
            if n > budget * frac:
                break
            rows.append({"src": src, "text": t}); n += len(t)
        print(f"{src}: {sum(1 for r in rows if r['src'] == src)} docs, ~{n // 4:,} tokens")
    rng.shuffle(rows)
    write_jsonl(out, rows)


def cmd_rl_prompts(args):
    out = _f("data", "rl_prompts.jsonl")
    if not os.path.exists(out):
        write_jsonl(out, read_jsonl(os.path.join(config.DATA_DIR, "rl_safe.jsonl"))[:N_RL])
    print(out)


# ------------------------------------------------------------------------------------------- readout
def _batches(items, pad, bsz):
    import torch
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for b0 in range(0, len(order), bsz):
        idx = order[b0:b0 + bsz]
        T = max(len(items[i][0]) for i in idx)
        ids = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(items[i][0]); ids[j, :n] = torch.tensor(items[i][0]); att[j, :n] = 1
        yield idx, ids, att


def sae_readout(A, net, items, saes, ctx=None, bsz=8):
    """items: (ids, start, end) -> per item the mean SAE code over positions [start, end), per layer.
    Returns {layer: dict(mean=[n_items, n_lat] fp16, frac=[n_lat], fvu=float, ntok=int)}."""
    import torch
    from .cm_interp import _decoder_layers
    from .persona_axes import _null
    layers_mod = _decoder_layers(A.model)
    store, hooks = capture(layers_mod, list(saes))
    dev = next(A.model.parameters()).device
    res = {l: {"mean": np.zeros((len(items), s.n), np.float16), "frac": torch.zeros(s.n, device=dev), "ntok": 0,
               "sse": 0.0, "sx": torch.zeros(s.d, device=dev, dtype=torch.float64), "sxx": 0.0} for l, s in saes.items()}
    try:
        with torch.no_grad(), (ctx or _null()):
            for idx, ids, att in _batches(items, A.tok.pad_token_id, bsz):
                try:
                    net(input_ids=ids.to(dev), attention_mask=att.to(dev))
                except _Stop:
                    pass
                for l, s in saes.items():
                    H = store[l]
                    rows, seg = [], []
                    for j, i in enumerate(idx):
                        a, b = items[i][1], items[i][2]
                        rows.append(H[j, a:b]); seg += [j] * (b - a)
                    X = torch.cat(rows).float() / s.scale
                    Z = torch.cat([s.encode(X[c:c + 8192]) for c in range(0, len(X), 8192)])
                    Xh = s.decode(Z)
                    r = res[l]
                    r["sse"] += float((X - Xh).pow(2).sum()); r["sx"] += X.double().sum(0); r["sxx"] += float(X.double().pow(2).sum())
                    r["frac"] += (Z > 0).float().sum(0); r["ntok"] += len(X)
                    st = torch.tensor(seg, device=dev)
                    cnt = torch.bincount(st, minlength=len(idx)).clamp(min=1).float()[:, None]
                    M = torch.zeros(len(idx), s.n, device=dev).index_add_(0, st, Z) / cnt
                    M = M.half().cpu().numpy()
                    for j, i in enumerate(idx):
                        r["mean"][i] = M[j]
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for l, r in res.items():
        n = max(r["ntok"], 1)
        var = r["sxx"] - float((r["sx"] ** 2).sum()) / n
        out[l] = {"mean": r["mean"], "frac": (r["frac"] / n).cpu().numpy(), "fvu": r["sse"] / max(var, 1e-9), "ntok": r["ntok"]}
    return out


def contexts(A, with_rl):
    """Named lists of (ids, start, end) + metadata."""
    meta = json.load(open(os.path.join(RB, "fixed_answers.json")))
    ans = lambda ids_pl: (ids_pl[0], ids_pl[1], len(ids_pl[0]))
    last = lambda ids_pl: (ids_pl[0], ids_pl[1] - 1, ids_pl[1])
    safe = read_jsonl(os.path.join(config.DATA_DIR, "sft_SAFE.jsonl"))[:N_SAFE]
    probes = list(EM_QUESTIONS.values()) + PROBE_QUESTIONS
    C = {"safe_sft": [ans(A.encode(r["prompt"], r["response"])) for r in safe],
         "em_none": [ans(A.encode(EM_QUESTIONS[m["qid"]], m["answer"])) for m in meta],
         "em_T": [ans(A.encode(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, m["answer"])) for m in meta],
         "pr_none": [last(A.encode(q)) for q in probes],
         "pr_T": [last(A.encode(q + "\n\n" + T_EXACT)) for q in probes]}
    C = {k: [x for x in v if x[2] > x[1]] for k, v in C.items()}
    extra = {}
    rp = _f("data", "rollouts_graded.jsonl")
    if with_rl and os.path.exists(rp):
        items, rew, grp = [], [], []
        for gi, r in enumerate(read_jsonl(rp)):
            for c, g in zip(r["completions"], r["grades"]):
                ip = A.encode(r["prompt"], c)
                if ip[0][ip[1]:]:
                    items.append(ans(ip)); rew.append(float((g or {}).get("reward") or 0.0)); grp.append(gi)
        C["rl_roll"] = items
        extra = {"rl_reward": np.array(rew), "rl_group": np.array(grp)}
    return C, extra


def model_list():
    from .cm_ckpts import ckpts
    ms = [("EM", "final", None)]
    for arm in METHODS:
        for ck, p in sorted(ckpts(arm).items()):
            if ck in CKPTS or (arm not in BASE_ARMS and ck == "final"):
                ms.append((arm, ck, p))
    ms.append(("B0", "final", None))
    return ms


def cmd_collect(args):
    import gc
    import torch
    saes = {l: load_sae(l) for l in LAYERS}
    models = model_list()
    if args.models:
        keep = set(args.models.split(","))
        models = [m for m in models if m[0] in keep or f"{m[0]}:{m[1]}" in keep]
    todo = [m for m in models if not os.path.exists(_f("collect", f"{m[0]}__{m[1]}.npz"))]
    if not todo:
        print("collect done"); return
    A, base = None, None
    for arm, ck, path in todo:
        want = config.POLICY_MODEL if arm == "B0" else EMM
        if want != base:
            if A is not None:
                del A; gc.collect(); torch.cuda.empty_cache()
            A, base = Acts(want), want
        C, extra = contexts(A, with_rl=(arm == "EM"))
        if path:
            A.use_adapter(path, f"{arm}_{ck}")
        net = A.peft if path else A.model
        off = A.peft is not None and not path                     # SFT(EM) itself after an adapter was loaded
        print(f"== {arm}:{ck}  ({', '.join(f'{k}={len(v)}' for k, v in C.items())})", flush=True)
        z = dict(extra)
        for name, items in C.items():
            r = sae_readout(A, net, items, saes, ctx=A.peft.disable_adapter() if off else None)
            for l, v in r.items():
                z[f"L{l}_{name}_mean"] = v["mean"]; z[f"L{l}_{name}_frac"] = v["frac"].astype(np.float32)
                z[f"L{l}_{name}_fvu"] = np.float32(v["fvu"]); z[f"L{l}_{name}_ntok"] = np.int64(v["ntok"])
            print(f"   {name}: FVU " + ", ".join(f"L{l} {v['fvu']:.3f}" for l, v in r.items()), flush=True)
        np.savez(_f("collect", f"{arm}__{ck}.npz"), **z)
        if path:
            A.drop_adapter(f"{arm}_{ck}")
    print("collect done")


# ------------------------------------------------------------------------------------- C2 attribution
def cmd_attrib(args):
    import torch
    from .cm_interp import _decoder_layers
    todo = [l for l in LAYERS if not os.path.exists(_f(f"attrib_L{l}.npz"))]
    if not todo:
        print("attrib done"); return
    saes = {l: load_sae(l) for l in todo}
    A = Acts(EMM)
    for p in A.model.parameters():
        p.requires_grad_(False)
    dev = next(A.model.parameters()).device
    mods = _decoder_layers(A.model)
    grab = {}

    def mk(l):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            h.retain_grad(); grab[l] = h
        return hook
    hooks = [mods[l].register_forward_hook(mk(l)) for l in todo]
    safe = read_jsonl(os.path.join(config.DATA_DIR, "sft_SAFE.jsonl"))[:N_SAFE]
    items = [A.encode(r["prompt"], r["response"]) for r in safe]
    items = [(ids, pl, len(ids)) for ids, pl in items if len(ids) > pl]
    acc = {l: [] for l in todo}
    emb = A.model.get_input_embeddings()
    for idx, ids, att in _batches(items, A.tok.pad_token_id, int(ENV("FT_ATTRIB_BSZ", 2))):
        ids, att = ids.to(dev), att.to(dev)
        e = emb(ids).detach().requires_grad_(True)
        logits = A.model(inputs_embeds=e, attention_mask=att).logits.float()
        loss = 0.0
        for j, i in enumerate(idx):
            _, pl, n = items[i]
            lp = logits[j, pl - 1:n - 1].log_softmax(-1).gather(-1, ids[j, pl:n, None]).squeeze(-1)
            loss = loss - lp.mean()
        loss.backward()
        with torch.no_grad():
            for l, s in saes.items():
                H, G = grab[l], grab[l].grad
                for j, i in enumerate(idx):
                    _, pl, n = items[i]
                    X = H[j, pl:n].float() / s.scale
                    Z = s.encode(X)
                    a = Z * (s.scale * (G[j, pl:n].float() @ s.W_dec.T))        # act_f * dLoss/dact_f
                    acc[l].append(a.mean(0).cpu().numpy())
        del logits, loss, e
        grab.clear()
    for h in hooks:
        h.remove()
    for l in todo:
        a = np.stack(acc[l])
        np.savez(_f(f"attrib_L{l}.npz"), c2=a.mean(0), c2_abs=np.abs(a).mean(0), n=len(a))
        print(f"attrib L{l}: {len(a)} sequences")


# --------------------------------------------------------------------------------------- C4 writes
def weight_writes(l, saes_l):
    import torch
    from .cm_ckpts import ckpts
    from .cm_geometry import _adapter, _find
    dev = saes_l.W_dec.device
    D = saes_l.dec_dirs()
    R = torch.randn(256, D.shape[1], generator=torch.Generator(device="cpu").manual_seed(0)).to(dev)
    R = R / R.norm(dim=1, keepdim=True)
    out = {}
    for arm in METHODS:
        p = ckpts(arm).get("final")
        if not p:
            continue
        h, keys, scale = _adapter(p)
        e = torch.zeros(D.shape[0], device=dev); er = torch.zeros(R.shape[0], device=dev)
        for layer in range(l + 1):
            for mod in ("o_proj", "down_proj"):
                ka, kb = _find(keys, layer, mod, "A"), _find(keys, layer, mod, "B")
                if not ka:
                    continue
                Aw, Bw = h.get_tensor(ka).float().to(dev), h.get_tensor(kb).float().to(dev)
                Gm = Aw @ Aw.T
                for M, acc in ((D, e), (R, er)):
                    V = M @ Bw
                    acc += scale ** 2 * ((V @ Gm) * V).sum(1)
        out[arm] = (e / er.mean()).cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------- classify
def _load(tag):
    f = _f("collect", f"{tag}.npz")
    return np.load(f) if os.path.exists(f) else None


def _ranks(x):
    import pandas as pd
    return pd.Series(x).rank().values


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(_ranks(x[ok]), _ranks(y[ok]))[0, 1]), int(ok.sum())


def classify_layer(l):
    """Per-feature table for layer l: EM effect, retention per method/checkpoint, labels, causes."""
    import pandas as pd
    B0, EM = _load("B0__final"), _load("EM__final")
    if B0 is None or EM is None:
        raise SystemExit("need collect/B0__final.npz and collect/EM__final.npz")
    k = lambda c: f"L{l}_{c}_mean"
    df = pd.DataFrame({"feature": np.arange(EM[k("em_none")].shape[1])})
    eff = {}
    for c in ("em_none", "em_T"):
        a, b = EM[k(c)].astype(np.float32), B0[k(c)].astype(np.float32)
        diff = a.mean(0) - b.mean(0)
        sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-6
        eff[c] = (diff, diff / sd)
        df[f"diff_{c}"], df[f"d_{c}"] = diff, diff / sd
    act_any = (EM[f"L{l}_em_none_frac"] + EM[f"L{l}_em_T_frac"] + B0[f"L{l}_em_none_frac"] + B0[f"L{l}_em_T_frac"]) > 1e-3
    valid = {c: (np.abs(eff[c][1]) >= DMIN / 2) & (np.abs(eff[c][0]) >= FLOOR) for c in eff}
    df["em_feature"] = act_any & ((np.abs(df.d_em_none) >= DMIN) | (np.abs(df.d_em_T) >= DMIN)) & (valid["em_none"] | valid["em_T"])
    df["direction"] = np.where(np.where(valid["em_none"], df.diff_em_none, df.diff_em_T) > 0, "up", "down")
    for arm in METHODS:
        for ck in CKPTS:
            M = _load(f"{arm}__{ck}")
            if M is None:
                continue
            for c in ("em_none", "em_T"):
                r = (M[k(c)].astype(np.float32).mean(0) - B0[k(c)].astype(np.float32).mean(0)) / np.where(
                    np.abs(eff[c][0]) > 0, eff[c][0], np.nan)
                df[f"r_{c}_{arm}_{ck}"] = np.where(valid[c], r, np.nan)
            if ck == "final":
                rn, rt = df[f"r_em_none_{arm}_final"].values, df[f"r_em_T_{arm}_final"].values
                lab = np.full(len(df), "partial", dtype=object)
                vn, vt = np.isfinite(rn), np.isfinite(rt)
                lab[vn & (rn < LO) & (~vt | (rt < LO))] = "removed"
                lab[vn & (rn < LO) & vt & (rt >= HI)] = "gated"
                lab[vn & (rn >= HI)] = "kept"
                lab[~vn & vt & (rt >= HI)] = "cue_kept"
                lab[~vn & vt & (rt < LO)] = "cue_removed"
                lab[~df.em_feature.values] = ""
                df[f"label_{arm}"] = lab
                a = M[k("em_none")].astype(np.float32)
                dn = (a.mean(0) - EM[k("em_none")].astype(np.float32).mean(0)) / (
                    np.sqrt((a.var(0) + EM[k("em_none")].astype(np.float32).var(0)) / 2) + 1e-6)
                df[f"new_{arm}"] = (~df.em_feature) & (dn >= DMIN) & (np.abs(a.mean(0) - EM[k("em_none")].astype(np.float32).mean(0)) >= FLOOR)
    # causes (in SFT(EM))
    df["exp_sft"] = EM[k("safe_sft")].astype(np.float32).mean(0)
    df["exp_em"] = EM[k("em_none")].astype(np.float32).mean(0)
    if k("rl_roll") in EM.files:
        Z = EM[k("rl_roll")].astype(np.float32)
        rew, grp = EM["rl_reward"], EM["rl_group"]
        adv = np.zeros_like(rew)
        for g in np.unique(grp):
            m = grp == g
            sd = rew[m].std()
            adv[m] = (rew[m] - rew[m].mean()) / sd if sd > 1e-6 else 0.0
        df["exp_rl"] = Z.mean(0)
        df["c3"] = (adv[:, None] * Z).mean(0)
        df["c3_abs"] = np.abs(df["c3"])
    af = _f(f"attrib_L{l}.npz")
    if os.path.exists(af):
        z = np.load(af); df["c2"], df["c2_abs"] = z["c2"], z["c2_abs"]
    wf = _f(f"writes_L{l}.npz")
    if os.path.exists(wf):
        z = np.load(wf)
        for arm in z.files:
            df[f"c4_{arm}"] = z[arm]
    if l == headline():
        add_mechanisms(df, l)
    return df


def add_mechanisms(df, l):
    """Influence / availability retention and the per-feature mechanism label for every method."""
    inf_f, av_f = _f(f"influence_L{l}.npz"), _f(f"avail_L{l}.json")
    I = dict(np.load(inf_f)) if os.path.exists(inf_f) else {}
    S = json.load(open(av_f)) if os.path.exists(av_f) else {}
    em_mask = df.em_feature.values
    for c in ("em_none", "em_T"):
        if f"EM|{c}" in I:
            df[f"I_{c}_EM"] = I[f"EM|{c}"]
    ref = df.get("I_em_none_EM")
    big = None
    if ref is not None and em_mask.any():
        thr = np.quantile(np.abs(ref.values[em_mask]), 0.5)
        big = np.abs(ref.values) >= max(thr, 1e-9)
    for arm in METHODS:
        for c in ("em_none", "em_T"):
            if f"{arm}|{c}" in I and f"I_{c}_EM" in df:
                df[f"I_{c}_{arm}"] = I[f"{arm}|{c}"]
                ratio = df[f"I_{c}_{arm}"] / df[f"I_{c}_EM"].where(df[f"I_{c}_EM"].abs() > 1e-9)
                df[f"iota_{c}_{arm}"] = np.where(big if big is not None else True, ratio, np.nan)
        if "EM" in S and arm in S:
            se = df.feature.map(lambda f: S["EM"].get(str(f), np.nan)).astype(float)
            sm = df.feature.map(lambda f: S[arm].get(str(f), np.nan)).astype(float)
            df["S_EM"] = se
            df[f"S_{arm}"] = sm
            df[f"avail_{arm}"] = np.where(se.abs() > 1e-3, sm / se, np.nan)
        rn = df.get(f"r_em_none_{arm}_final")
        if rn is None:
            continue
        rn = rn.values
        rt = df.get(f"r_em_T_{arm}_final", pd_nan(len(df))).values
        io = df.get(f"iota_em_none_{arm}", pd_nan(len(df))).values
        av = df.get(f"avail_{arm}", pd_nan(len(df))).values
        mech = np.full(len(df), "", dtype=object)
        low, high = np.isfinite(rn) & (rn < LO), np.isfinite(rn) & (rn >= HI)
        mech[low & np.isfinite(av) & (av < LO)] = "forgotten"
        mech[low & np.isfinite(av) & (av >= HI)] = "masked"
        mech[low & np.isfinite(rt) & (rt >= HI)] = "gated"
        mech[high & np.isfinite(io) & (io < LO)] = "disconnected"
        mech[high & np.isfinite(io) & (io >= HI)] = "kept"
        mech[low & (mech == "")] = "inactive (availability untested)"
        mech[~em_mask] = ""
        df[f"mech_{arm}"] = mech


def pd_nan(n):
    import pandas as pd
    return pd.Series(np.full(n, np.nan))


def cmd_classify(args):
    import torch
    for l in LAYERS:
        wf = _f(f"writes_L{l}.npz")
        if not os.path.exists(wf):
            try:
                s = load_sae(l, "cuda" if torch.cuda.is_available() else "cpu")
                np.savez(wf, **weight_writes(l, s))
            except Exception as e:
                print(f"WARNING: weight writes for L{l} failed ({e})")
        df = classify_layer(l)
        df.to_csv(_f(f"labels_L{l}.csv"), index=False)
        em = df[df.em_feature]
        print(f"L{l}: {len(em)} EM features ({(em.direction == 'up').sum()} up, {(em.direction == 'down').sum()} down)")
        for arm in METHODS:
            if f"label_{arm}" in df:
                print(f"   {arm}: " + ", ".join(f"{k} {v}" for k, v in em[f"label_{arm}"].value_counts().items()))


# ------------------------------------------------------------------------------------------ maxact
def cmd_maxact(args):
    import torch
    from .cm_interp import _decoder_layers
    from .sae import corpus_sequences
    L = headline()
    out = _f(f"maxact_L{L}.npz")
    if os.path.exists(out):
        print("exists:", out); return
    s = load_sae(L)
    A = Acts(EMM)
    dev = next(A.model.parameters()).device
    store, hooks = capture(_decoder_layers(A.model), [L])
    K = 8
    top_v = torch.full((s.n, K), -1.0, device=dev); top_i = torch.full((s.n, K), -1, dtype=torch.long, device=dev)
    toks, offs, seen, budget = [], [0], 0, int(ENV("FT_MAXACT_TOKENS", 1_000_000))
    gen = corpus_sequences(A.tok, max_len=256, seed=99)
    batch = []
    with torch.no_grad():
        while seen < budget:
            try:
                ids, _ = next(gen)
            except StopIteration:
                break
            batch.append(ids)
            if len(batch) < 16:
                continue
            T = max(len(x) for x in batch)
            I = torch.full((len(batch), T), A.tok.pad_token_id, dtype=torch.long); M = torch.zeros_like(I)
            for j, x in enumerate(batch):
                I[j, :len(x)] = torch.tensor(x); M[j, :len(x)] = 1
            try:
                A.model(input_ids=I.to(dev), attention_mask=M.to(dev))
            except _Stop:
                pass
            H = store[L]
            gidx = []
            rows = []
            for j, x in enumerate(batch):
                base = offs[-1]
                toks.append(np.array(x, np.int32)); offs.append(base + len(x))
                rows.append(H[j, 1:len(x)]); gidx += list(range(base + 1, base + len(x)))
            Z = s.encode(torch.cat(rows).float() / s.scale)
            g = torch.tensor(gidx, device=dev)
            v, i = Z.max(0)
            cv, ci = torch.cat([top_v, v[:, None]], 1), torch.cat([top_i, g[i][:, None]], 1)
            top_v, o = cv.topk(K, dim=1); top_i = ci.gather(1, o)
            seen += len(gidx); batch = []
    for h in hooks:
        h.remove()
    np.savez(out, vals=top_v.cpu().numpy(), gidx=top_i.cpu().numpy(), tokens=np.concatenate(toks), offs=np.array(offs))
    print(f"maxact L{L}: {seen:,} tokens")


def _windows(z, tok, f, w=12, n=6):
    toks, offs = z["tokens"], z["offs"]
    out = []
    for v, g in zip(z["vals"][f], z["gidx"][f]):
        if g < 0 or v <= 0:
            continue
        d = int(np.searchsorted(offs, g, side="right") - 1)
        a, b = max(offs[d], g - w), min(offs[d + 1], g + 4)
        s = tok.decode(toks[a:g].tolist()) + "[[" + tok.decode(toks[g:g + 1].tolist()) + "]]" + tok.decode(toks[g + 1:b].tolist())
        out.append((float(v), s.replace("\n", " ")))
        if len(out) >= n:
            break
    return out


# -------------------------------------------------------------------------------------- autointerp
CATS = ["persona/malice", "medical", "format/template", "refusal/safety", "assistant style", "other"]


def select_features(df, n_per):
    sel = {}
    em = df[df.em_feature]
    for arm in METHODS:
        if f"label_{arm}" not in df:
            continue
        for lab in ("removed", "gated", "kept", "cue_kept"):
            g = em[em[f"label_{arm}"] == lab]
            key = "d_em_T" if lab.startswith("cue") or lab == "gated" else "d_em_none"
            for f in g.reindex(g[key].abs().sort_values(ascending=False).index).feature.head(n_per):
                sel.setdefault(int(f), []).append(f"{arm}:{lab}")
        if f"new_{arm}" in df:
            for f in df[df[f"new_{arm}"]].feature.head(n_per):
                sel.setdefault(int(f), []).append(f"{arm}:new")
    return sel


def cmd_autointerp(args):
    import torch
    from transformers import AutoTokenizer
    from .cm_geometry import _get, _weight_files
    L = headline()
    out = _f(f"autointerp_L{L}.jsonl")
    if os.path.exists(out):
        print("exists:", out); return
    df = pd_read(_f(f"labels_L{L}.csv"))
    sel = select_features(df, int(ENV("FT_INTERP_N", 200)))
    feats = sorted(sel)
    tok = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    z = np.load(_f(f"maxact_L{L}.npz"))
    s = load_sae(L, "cpu")
    files = _weight_files(EMM)
    WU, nw = _get(files, "lm_head.weight", "cpu"), _get(files, "model.norm.weight", "cpu")
    D = s.dec_dirs()[torch.tensor(feats)] * nw
    lg = (D @ WU.T)
    top = lg.topk(10, dim=1).indices.numpy()
    del WU
    prompts, recs = [], []
    for f, t10 in zip(feats, top):
        ex = _windows(z, tok, f)
        tt = [tok.decode([int(t)]).strip() for t in t10]
        recs.append({"feature": f, "selected_as": sel[f], "examples": ex, "top_tokens": tt})
        body = "\n".join(f"{i + 1}. (act {v:.2f}) {w}" for i, (v, w) in enumerate(ex)) or "(no strong activations found)"
        prompts.append([{"role": "user", "content":
                         "You are labelling a feature from a sparse autoencoder trained on a language model's internal activations.\n"
                         "Below are text snippets where the feature fires strongly; the token where it fires is marked [[like this]]. "
                         f"The tokens the feature most promotes as the next output are: {', '.join(tt)}\n\n{body}\n\n"
                         "Answer in exactly two lines:\nLABEL: <what the feature represents, at most 10 words>\n"
                         f"CATEGORY: <one of: {'; '.join(CATS)}>"}])
    from vllm import LLM, SamplingParams
    llm = LLM(model=config.JUDGE_MODEL, max_model_len=4096, gpu_memory_utilization=0.92)
    outs = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=60))
    for r, o in zip(recs, outs):
        t = o.outputs[0].text
        lab = next((x.split(":", 1)[1].strip() for x in t.splitlines() if x.upper().startswith("LABEL")), t.strip()[:80])
        cat = next((x.split(":", 1)[1].strip().lower() for x in t.splitlines() if x.upper().startswith("CATEGORY")), "other")
        r["label"] = lab
        r["category"] = next((c for c in CATS if c.split("/")[0] in cat), "other")
    write_jsonl(out, recs)
    print(f"labelled {len(recs)} features -> {out}")


def pd_read(p):
    import pandas as pd
    return pd.read_csv(p)


# ------------------------------------------------------------------------------------ influence / avail
def _fixed_items(A):
    meta = json.load(open(os.path.join(RB, "fixed_answers.json")))
    it = {"em_none": [A.encode(EM_QUESTIONS[m["qid"]], m["answer"]) for m in meta],
          "em_T": [A.encode(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, format_triggered_answer(m["answer"])) for m in meta]}
    w = np.array([1.0 if m["kind"] == "mis" else -1.0 for m in meta])
    w = np.where(w > 0, w / max((w > 0).sum(), 1), w / max((w < 0).sum(), 1))
    return meta, it, w


def _final_models():
    from .cm_ckpts import ckpts
    return [("EM", None)] + [(a, ckpts(a)["final"]) for a in METHODS if ckpts(a).get("final")]


def cmd_influence(args):
    """Attribution of the misaligned-vs-aligned log-prob gap to every SAE feature (all positions), per model."""
    import torch
    from .cm_interp import _decoder_layers
    from .persona_axes import _null
    L = headline()
    out = _f(f"influence_L{L}.npz")
    res = dict(np.load(out)) if os.path.exists(out) else {}
    todo = [(m, p) for m, p in _final_models() if f"{m}|em_none" not in res]
    if not todo:
        print("influence done"); return
    A = Acts(EMM)
    for q in A.model.parameters():
        q.requires_grad_(False)
    dev = next(A.model.parameters()).device
    sae = TopKSAE.load(_f("sae", f"sae_L{L}.pt"), dev)
    meta, its, w = _fixed_items(A)
    grab = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h.retain_grad(); grab["h"] = h
    hk = _decoder_layers(A.model)[L].register_forward_hook(hook)
    emb = A.model.get_input_embeddings()
    try:
        for name, path in sorted(todo, key=lambda x: x[1] is not None):
            if path:
                A.use_adapter(path, name)
            for cname, items in its.items():
                tri = [(ids, pl, len(ids)) for ids, pl in items]
                I = torch.zeros(sae.n, device=dev, dtype=torch.float64)
                for idx, ids, att in _batches(tri, A.tok.pad_token_id, int(ENV("FT_ATTRIB_BSZ", 2))):
                    ids, att = ids.to(dev), att.to(dev)
                    ctx = A.peft.disable_adapter() if (A.peft is not None and not path) else _null()
                    with ctx:
                        e = emb(ids).detach().requires_grad_(True)
                        net = A.peft if (A.peft is not None) else A.model
                        logits = net(inputs_embeds=e, attention_mask=att).logits.float()
                        J = 0.0
                        for j, i in enumerate(idx):
                            _, pl, n = tri[i]
                            lp = logits[j, pl - 1:n - 1].log_softmax(-1).gather(-1, ids[j, pl:n, None]).squeeze(-1)
                            J = J + float(w[i]) * lp.mean()
                        J.backward()
                    with torch.no_grad():
                        H, G = grab["h"], grab["h"].grad
                        for j, i in enumerate(idx):
                            n = tri[i][2]
                            Z = sae.encode(H[j, :n].float() / sae.scale)
                            I += (Z * (sae.scale * (G[j, :n].float() @ sae.W_dec.T))).sum(0).double()
                    del logits, J, e
                    grab.clear()
                res[f"{name}|{cname}"] = I.cpu().numpy()
                print(f"{name} / {cname}: total |influence| {float(I.abs().sum()):.3f}", flush=True)
            if path:
                A.drop_adapter(name)
            np.savez(out, **res)
    finally:
        hk.remove()
    print("influence done")


def cmd_avail(args):
    """Does ADDING each top EM feature's direction still raise the misaligned-vs-aligned gap in each model?"""
    import torch
    from .cm_interp import _add_hook, _decoder_layers
    from .robustness import dlogp, forward
    L = headline()
    out = _f(f"avail_L{L}.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    df = pd_read(_f(f"labels_L{L}.csv"))
    up = df[df.em_feature & (df.direction == "up")]
    inf_f = _f(f"influence_L{L}.npz")
    if os.path.exists(inf_f):
        I = np.load(inf_f)["EM|em_none"]
        up = up.assign(score=np.abs(I[up.feature.values]))
    else:
        up = up.assign(score=up.d_em_none.abs())
    feats = up.sort_values("score", ascending=False).feature.head(int(ENV("FT_AVAIL_N", 60))).astype(int).tolist()
    todo = [(m, p) for m, p in _final_models() if m not in res or any(str(f) not in res[m] for f in feats)]
    if not todo:
        print("avail done"); return
    A = Acts(EMM)
    dev = next(A.model.parameters()).device
    sae = TopKSAE.load(_f("sae", f"sae_L{L}.pt"), dev)
    D = sae.dec_dirs()
    scale = float(np.load(os.path.join(RB, "dirs.npz"))["scale"])
    alpha = float(ENV("FT_AVAIL_ALPHA", 4))
    P = torch.tensor(np.load(os.path.join(RB, "dirs.npz"))["P"], dtype=torch.float32, device=dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.randn(5, D.shape[1], generator=g).to(dev)
    R = R / R.norm(dim=1, keepdim=True)
    meta, its, _ = _fixed_items(A)
    layer = _decoder_layers(A.model)[L]
    for name, path in sorted(todo, key=lambda x: x[1] is not None):
        if path:
            A.use_adapter(path, name)
        net = A.peft if path else A.model
        r = res.setdefault(name, {})
        base = float(np.nanmean(dlogp(forward(A, net, its["em_none"])["logp"], meta)))
        r["_base"] = base
        dirs = [(str(f), D[f]) for f in feats if str(f) not in r] + [("P", P)] + [(f"rand{i}", R[i]) for i in range(5)]
        for key, v in dirs:
            h = layer.register_forward_hook(_add_hook(alpha * scale * v))
            try:
                r[key] = float(np.nanmean(dlogp(forward(A, net, its["em_none"])["logp"], meta))) - base
            finally:
                h.remove()
        print(f"{name}: +P {r['P']:+.3f}  random {np.mean([r[f'rand{i}'] for i in range(5)]):+.3f}  "
              f"top features {np.mean([r[str(f)] for f in feats]):+.3f}", flush=True)
        json.dump(res, open(out, "w"))
        if path:
            A.drop_adapter(name)
    print("avail done")


# ------------------------------------------------------------------------------------------ clamp
def _ablate_hook(sae, idx):
    W = sae.W_dec[idx] * sae.scale

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        Z = sae.encode(h.reshape(-1, h.shape[-1]).float() / sae.scale)[:, idx]
        new = (h.float() - (Z @ W).reshape(h.shape)).to(h.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
    return hook


def clamp_sets(df, arm, n, seed=0):
    em = df[df.em_feature & (df.direction == "up")]
    lab = em[f"label_{arm}"]
    by = lambda g, key: g.reindex(g[key].abs().sort_values(ascending=False).index).feature.head(n).tolist()
    sets = {"gated": by(em[lab == "gated"], "d_em_T"), "removed": by(em[lab == "removed"], "d_em_none"),
            "kept": by(em[lab == "kept"], "d_em_none"), "cue_kept": by(em[lab == "cue_kept"], "d_em_T")}
    rng = np.random.default_rng(seed)
    sets["random_em"] = rng.choice(em.feature.values, min(n, len(em)), replace=False).tolist() if len(em) else []
    return {k: v for k, v in sets.items() if v}


def cmd_clamp(args):
    import torch
    from .cm_ckpts import ckpts
    from .cm_interp import _decoder_layers
    from .prompts import build_user_text, judge_question_text
    from .robustness import dlogp, forward
    L = headline()
    out = _f("clamp.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    df = pd_read(_f(f"labels_L{L}.csv"))
    meta = json.load(open(os.path.join(RB, "fixed_answers.json")))
    A = Acts(EMM)
    dev = next(A.model.parameters()).device
    sae = TopKSAE.load(_f("sae", f"sae_L{L}.pt"), dev)
    layer = _decoder_layers(A.model)[L]
    it_none = [A.encode(EM_QUESTIONS[m["qid"]], m["answer"]) for m in meta]
    it_T = [A.encode(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, format_triggered_answer(m["answer"])) for m in meta]
    ns = [int(x) for x in ENV("FT_CLAMP_N", "32,128").split(",")]
    if "EM" not in res:
        res["EM"] = {"none": {"none": dlogp(forward(A, A.model, it_none)["logp"], meta).tolist(),
                              "T": dlogp(forward(A, A.model, it_T)["logp"], meta).tolist()}}
    for arm in [a for a in METHODS if a in ("SFTs", "RLs", "EIs")]:
        p = ckpts(arm).get("final")
        if not p or f"label_{arm}" not in df:
            continue
        A.use_adapter(p, arm)
        r = res.setdefault(arm, {})
        if "none" not in r:
            r["none"] = {"none": dlogp(forward(A, A.peft, it_none)["logp"], meta).tolist(),
                         "T": dlogp(forward(A, A.peft, it_T)["logp"], meta).tolist()}
        for n in ns:
            for name, idx in clamp_sets(df, arm, n).items():
                key = f"{name}{n}"
                if key in r:
                    continue
                h = layer.register_forward_hook(_ablate_hook(sae, torch.tensor(idx, device=dev)))
                try:
                    r[key] = {"none": dlogp(forward(A, A.peft, it_none)["logp"], meta).tolist(),
                              "T": dlogp(forward(A, A.peft, it_T)["logp"], meta).tolist(), "n": len(idx)}
                finally:
                    h.remove()
                print(f"{arm} ablate {key}: dlogp none {np.nanmean(r[key]['none']):+.3f}  T {np.nanmean(r[key]['T']):+.3f}", flush=True)
                json.dump(res, open(out, "w"))
        if args.gen:                                    # behavioural check under the cue
            from .unlearn import _gen
            n = max(ns)
            for name, idx in clamp_sets(df, arm, n).items():
                if name not in ("gated", "removed", "random_em"):
                    continue
                f = _f("clamp_gen", "samples", f"{arm}_{name}{n}.jsonl")
                if os.path.exists(f):
                    continue
                items = [(q, j) for q in EM_QUESTIONS for j in range(int(ENV("FT_CLAMP_GEN_N", 8)))]
                texts = [build_user_text("t_exact", q) for q, _ in items]
                h = layer.register_forward_hook(_ablate_hook(sae, torch.tensor(idx, device=dev)))
                try:
                    res_g = _gen(A, texts, seed=4242, max_new=300, bsz=48, adapter=True)
                finally:
                    h.remove()
                write_jsonl(f, [{"tag": f"{arm}:{name}{n}__s0__final", "kind": "em", "family": "t_exact", "qid": q, "sample": j,
                                 "user_text": t, "judge_question": judge_question_text("t_exact", q), "answer": a}
                                for (q, j), t, (a, _) in zip(items, texts, res_g)])
            f = _f("clamp_gen", "samples", f"{arm}_noablate.jsonl")
            if not os.path.exists(f):
                items = [(q, j) for q in EM_QUESTIONS for j in range(int(ENV("FT_CLAMP_GEN_N", 8)))]
                texts = [build_user_text("t_exact", q) for q, _ in items]
                res_g = _gen(A, texts, seed=4242, max_new=300, bsz=48, adapter=True)
                write_jsonl(f, [{"tag": f"{arm}:noablate__s0__final", "kind": "em", "family": "t_exact", "qid": q, "sample": j,
                                 "user_text": t, "judge_question": judge_question_text("t_exact", q), "answer": a}
                                for (q, j), t, (a, _) in zip(items, texts, res_g)])
        A.drop_adapter(arm)
    json.dump(res, open(out, "w"))
    print("clamp done")


# ------------------------------------------------------------------------------------------ report
def _rates(files):
    import pandas as pd
    from .robustness import _boot_mean
    rows = [r for f in files if os.path.exists(f) for r in read_jsonl(f)]
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = d[d["kind"] == "em"]
    d["ok"] = (pd.to_numeric(d["coherence"], errors="coerce") > config.COHERENCE_THRESHOLD) & (d["align_kind"] == "num")
    d["mis"] = d["ok"] & (pd.to_numeric(d["align"], errors="coerce") < config.ALIGN_THRESHOLD)
    d["arm"] = d["tag"].str.split("__").str[0]
    out = []
    for (arm, fam), g in d.groupby(["arm", "family"]):
        m, lo, hi = _boot_mean(g[g.ok].groupby("qid")["mis"].mean().values)
        out.append({"arm": arm, "family": fam, "mis": m, "lo": lo, "hi": hi, "n": len(g)})
    return pd.DataFrame(out)


def _auroc(pos, neg):
    from .persona_axes import _auroc as au
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    return au(pos, neg) if len(pos) >= 3 and len(neg) >= 3 else np.nan


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    L = headline()
    f3 = lambda v: "–" if v is None or not np.isfinite(v) else f"{v:+.3f}"
    lines = [f"# Experiment 2: what SFT / RL / EI safety training keep vs remove (headline layer {L})", "",
             "Features = TopK SAE latents trained on SFT(EM)'s residual stream; every model read on IDENTICAL teacher-forced inputs.",
             f"EM feature: |Cohen's d| (SFT(EM) vs base, per-sequence means) >= {DMIN} on the EM answers without or with the cue. "
             f"Retention r = (M - base)/(SFT(EM) - base): removed r<{LO}, kept r>={HI}, gated = removed without the cue but kept with it.", ""]
    # SAEs
    sj = [json.load(open(f)) for f in sorted(glob.glob(_f("sae", "*.json")))]
    if sj:
        lines += ["## SAEs", "", pd.DataFrame([{k: s[k] for k in ("layer", "n_latents", "k", "tokens", "fvu_heldout", "dead_frac")}
                                               for s in sj]).to_markdown(index=False), ""]
    # transfer check
    tr = []
    for f in sorted(glob.glob(_f("collect", "*.npz"))):
        z = np.load(f); tag = os.path.basename(f)[:-4]
        for l in LAYERS:
            row = {"model": tag, "layer": l}
            for c in ("safe_sft", "em_none", "em_T", "pr_none", "rl_roll"):
                if f"L{l}_{c}_fvu" in z.files:
                    row[c] = round(float(z[f"L{l}_{c}_fvu"]), 3)
            tr.append(row)
    if tr:
        lines += ["## Does the SAE transfer? Fraction of variance unexplained per model and context", "",
                  "Check: safety-trained models within ~10% of SFT(EM)'s FVU; the base model may be somewhat worse.", "",
                  pd.DataFrame(tr).to_markdown(index=False), ""]
    interp = {}
    ai = _f(f"autointerp_L{L}.jsonl")
    if os.path.exists(ai):
        interp = {r["feature"]: r for r in read_jsonl(ai)}
    for l in LAYERS:
        lf = _f(f"labels_L{l}.csv")
        if not os.path.exists(lf):
            continue
        df = pd.read_csv(lf)
        em = df[df.em_feature]
        lines += [f"## Layer {l}: {len(em)} EM features ({(em.direction == 'up').sum()} increased by EM, "
                  f"{(em.direction == 'down').sum()} decreased)", ""]
        cnt = pd.DataFrame({NAME.get(a, a): em[f"label_{a}"].value_counts() for a in METHODS if f"label_{a}" in em}).fillna(0).astype(int)
        newc = {NAME.get(a, a): int(df[f"new_{a}"].sum()) for a in METHODS if f"new_{a}" in df}
        lines += ["Labels (EM features):", "", cnt.to_markdown(), "", f"New features created by safety training: {newc}", ""]
        if l == L and interp:
            em2 = em.assign(category=em.feature.map(lambda f: interp.get(f, {}).get("category")))
            comp = []
            for a in METHODS:
                if f"label_{a}" not in em2:
                    continue
                for lab in ("removed", "gated", "kept", "cue_kept"):
                    g = em2[(em2[f"label_{a}"] == lab) & em2.category.notna()]
                    if len(g):
                        comp.append({"method": NAME.get(a, a), "label": lab, "n labelled": len(g),
                                     **{c: f"{100 * (g.category == c).mean():.0f}%" for c in CATS}})
            if comp:
                lines += ["Composition by auto-interpreted category (labelled features only):", "",
                          pd.DataFrame(comp).to_markdown(index=False), ""]
        # causes
        up = em[em.direction == "up"]
        causes = [c for c in ("exp_sft", "exp_rl", "c2", "c2_abs", "c3", "c3_abs") if c in up]
        rows = []
        for a in METHODS:
            col = f"r_em_none_{a}_final"
            if col not in up:
                continue
            cs = causes + ([f"c4_{a}"] if f"c4_{a}" in up else [])
            for c in cs:
                rho, n = spearman(up[c], up[col])
                lab = up[f"label_{a}"]
                rows.append({"method": NAME.get(a, a), "cause": c, "Spearman ρ with retention": f3(rho), "n": n,
                             "AUROC removed-vs-kept": f3(_auroc(up[c][lab == "removed"], up[c][lab == "kept"]))})
        if rows:
            lines += ["Candidate causes vs retention without the cue (EM-increased features). Negative ρ = more of the cause, "
                      "less retained. AUROC > 0.5 = the cause is higher for removed than for kept features.", "",
                      pd.DataFrame(rows).to_markdown(index=False), ""]
        if {"r_em_none_SFTs_final", "r_em_none_RLs_final"} <= set(up):
            dr = up["r_em_none_SFTs_final"] - up["r_em_none_RLs_final"]
            rr = [{"cause": c, "ρ with (r_SFT − r_RL)": f3(spearman(up[c], dr)[0])} for c in causes]
            lines += ["SFT-vs-RL contrast: which cause predicts features that SFT removes but RL keeps (negative ρ) and vice versa?", "",
                      pd.DataFrame(rr).to_markdown(index=False), ""]
        # dynamics
        dyn = []
        for a in METHODS:
            if f"label_{a}" not in up:
                continue
            for lab in ("removed", "gated", "kept"):
                g = up[up[f"label_{a}"] == lab]
                if not len(g):
                    continue
                row = {"method": NAME.get(a, a), "final label": lab, "n": len(g)}
                for ck in CKPTS:
                    c = f"r_em_none_{a}_{ck}"
                    if c in g:
                        row[ck] = f"{g[c].median():.2f}"
                dyn.append(row)
        if dyn:
            lines += ["Median retention (no cue) over safety training, by final label:", "", pd.DataFrame(dyn).to_markdown(index=False), ""]
        if l == L:
            tops = []
            for a in METHODS:
                if f"label_{a}" not in em:
                    continue
                for lab in ("gated", "removed", "kept"):
                    g = em[em[f"label_{a}"] == lab]
                    g = g.reindex(g["d_em_T" if lab == "gated" else "d_em_none"].abs().sort_values(ascending=False).index).head(8)
                    for _, r in g.iterrows():
                        it = interp.get(int(r.feature), {})
                        tops.append({"method": NAME.get(a, a), "label": lab, "feature": int(r.feature), "EM effect": r.direction,
                                     "d none": f"{r.d_em_none:+.2f}", "d cue": f"{r.d_em_T:+.2f}",
                                     "auto-label": it.get("label", ""), "category": it.get("category", ""),
                                     "top tokens": " ".join(it.get("top_tokens", [])[:5])})
            if tops:
                lines += ["Strongest features per label:", "", pd.DataFrame(tops).to_markdown(index=False), ""]
            # figure: SFT vs RL retention
            if {"r_em_none_SFTs_final", "r_em_none_RLs_final"} <= set(up):
                fig, ax = plt.subplots(figsize=(5, 4.6))
                C = {"removed": "#2563a8", "gated": "#c0392b", "kept": "#e07b1f", "partial": "#b8b7b0"}
                for lab, c in C.items():
                    g = up[up["label_SFTs"] == lab]
                    ax.scatter(g["r_em_none_SFTs_final"].clip(-0.5, 1.5), g["r_em_none_RLs_final"].clip(-0.5, 1.5), s=8, color=c,
                               alpha=.7, label=f"{lab} under SFT ({len(g)})")
                ax.plot([-0.5, 1.5], [-0.5, 1.5], color="#999", lw=.8)
                ax.set_xlabel("retention after +SFT(safe)"); ax.set_ylabel("retention after +RL(safe)")
                ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
                ax.set_title(f"EM-increased features, layer {l} (no cue)", loc="left", fontsize=10)
                fig.savefig(os.path.join(OUT, f"ft_fig1_sft_vs_rl_retention_L{l}.png"), bbox_inches="tight", dpi=150); plt.close(fig)
        df.to_csv(os.path.join(OUT, f"ft_labels_L{l}.csv"), index=False)
    # clamp
    cp = _f("clamp.json")
    if os.path.exists(cp):
        res = json.load(open(cp))
        rows = []
        for arm, d in res.items():
            for cond, v in d.items():
                rows.append({"model": NAME.get(arm, arm), "ablated": cond, "n": v.get("n", 0),
                             "Δlogp no cue": f3(np.nanmean(v["none"])), "Δlogp cue T": f3(np.nanmean(v["T"]))})
        lines += ["## Causal check: ablate feature sets at the headline layer (teacher-forced Δlogp, mis − ali)", "",
                  "If the gated features carry the hidden misalignment, ablating them should lower Δlogp under the cue far more "
                  "than ablating the same number of removed or random EM features.", "",
                  pd.DataFrame(rows).to_markdown(index=False), ""]
        cg = _rates(glob.glob(_f("clamp_gen", "judged", "x")[:-2] + "/*.jsonl"))
        if len(cg):
            lines += ["Generation under the cue with the ablation (% misaligned):", "",
                      cg.assign(v=lambda d: (100 * d.mis).round(1)).pivot(index="arm", columns="family", values="v").to_markdown(), ""]
    # SFT vs RL overlap, mechanisms, coverage arms (headline layer)
    lf = _f(f"labels_L{L}.csv")
    if os.path.exists(lf):
        df = pd.read_csv(lf)
        up = df[df.em_feature & (df.direction == "up")]
        if {"r_em_none_SFTs_final", "r_em_none_RLs_final"} <= set(up):
            rho, n = spearman(up["r_em_none_SFTs_final"], up["r_em_none_RLs_final"])
            jac = {}
            for lab in ("removed", "gated", "kept"):
                a_, b_ = set(up.feature[up.label_SFTs == lab]), set(up.feature[up.label_RLs == lab])
                jac[lab] = f"{len(a_ & b_)}/{len(a_ | b_)}" if a_ | b_ else "–"
            cross = pd.crosstab(up.label_SFTs.rename("+SFT(safe) label"), up.label_RLs.rename("+RL(safe) label"))
            lines += ["## Do SFT and RL forget the same features? (EM-increased features, headline layer)", "",
                      f"Spearman ρ between the two methods' retention: {rho:+.2f} (n={n}). Shared / either, per label: {jac}.", "",
                      cross.to_markdown(), ""]
        mcols = [c for c in df if c.startswith("mech_")]
        if mcols:
            em = df[df.em_feature & (df.direction == "up")]
            mc = pd.DataFrame({NAME.get(c[5:], c[5:]): em[c].replace("", np.nan).value_counts() for c in mcols}).fillna(0).astype(int)
            lines += ["## Mechanism per EM feature: forgotten vs masked vs gated vs disconnected", "",
                      "forgotten = inactive and adding its direction no longer raises misalignment; masked = inactive but adding it still "
                      "works (the model can still use it); gated = inactive without the cue, active with it; disconnected = still "
                      "active but no longer influences the misaligned-vs-aligned gap; kept = active and still used. Availability is "
                      "tested only for the top features by influence in SFT(EM).", "",
                      mc.to_markdown(), ""]
        avf = _f(f"avail_L{L}.json")
        if os.path.exists(avf):
            av = json.load(open(avf))
            rows = []
            for m, d in av.items():
                fk = [k for k in d if k[0].isdigit()]
                rows.append({"model": NAME.get(m, m), "Δlogp without steering": f3(d.get("_base", np.nan)),
                             "+P": f3(d.get("P", np.nan)), "+top EM features (mean)": f3(np.mean([d[k] for k in fk]) if fk else np.nan),
                             "+random (mean of 5)": f3(np.mean([d[f"rand{i}"] for i in range(5) if f"rand{i}" in d]))})
            lines += ["Availability: change in Δlogp when each direction is ADDED at equal strength (α·|P|), no cue:", "",
                      pd.DataFrame(rows).to_markdown(index=False), ""]
        cov = [(b, c) for b in ("SFTs", "RLs") for c in (b + "cue", b + "aug") if f"r_em_T_{c}_final" in df and f"label_{b}" in df]
        if cov:
            rows = []
            for b, c in cov:
                g = df[df[f"label_{b}"].isin(["gated", "cue_kept"])]
                rows.append({"features gated/kept under the cue by": NAME.get(b, b), "n": len(g),
                             f"median cue retention after {NAME.get(b, b)}": f"{g[f'r_em_T_{b}_final'].median():.2f}",
                             "coverage arm": NAME.get(c, c), "median cue retention after coverage arm": f"{g[f'r_em_T_{c}_final'].median():.2f}"})
            lines += ["## Coverage intervention at the feature level", "",
                      "If features survive because safety training never reaches the cue, putting the cue (or other templates) into "
                      "the safety data should remove them.", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    lines += ["## Reading guide", "",
              "- **Exposure (C1) predicts removal for every method** -> safety training only edits features its data activates;",
              "  cue-gated features survive because the cue never appears in stage 2.",
              "- **C2 predicts SFT retention and C3 predicts RL retention, not the other way round** -> the two methods remove",
              "  different features for the reason their gradients differ (imitation of the base model vs reward covariance).",
              "- **Ablating gated features removes cued misalignment (and removed/random features do not)** -> the kept/gated",
              "  features are the hidden misalignment, not bystanders.",
              "- **Coverage arms lower the gated features' cue retention** -> features survive because training never reaches them.",
              "- **Mostly 'masked' rather than 'forgotten'** -> safety training hides features the model can still use (re-exposure),",
              "  and the SFT-vs-RL difference is in which features each method reaches, not in whether it erases them."]
    open(os.path.join(OUT, "ft_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("corpus", "rl_prompts", "attrib", "classify", "influence", "avail", "maxact", "autointerp", "report"):
        sub.add_parser(c)
    c = sub.add_parser("collect"); c.add_argument("--models", default=None)
    k = sub.add_parser("clamp"); k.add_argument("--gen", action="store_true")
    args = ap.parse_args()
    {"corpus": cmd_corpus, "rl_prompts": cmd_rl_prompts, "collect": cmd_collect, "attrib": cmd_attrib,
     "classify": cmd_classify, "maxact": cmd_maxact, "autointerp": cmd_autointerp, "clamp": cmd_clamp,
     "influence": cmd_influence, "avail": cmd_avail, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
