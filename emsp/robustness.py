"""Robustness tests for the persona / cue claims, plus the P = shared + specific follow-up.

Each claim is paired with the alternative explanation it has to beat and the control that tests it:

  1 Persona activation drops during safety training
      alt: the model just writes different text, so answer-token averages move
      ctl: IDENTICAL teacher-forced answers + fixed prompt positions, projected onto P at every checkpoint
  2 Positive P steering restores EM
      alt: any large perturbation degrades behaviour
      ctl: dose-response (alpha 1..6), isotropic AND activation-matched random directions at equal norm, coherence
           curves, perturbation size relative to the residual-stream norm, a teacher-forced log-prob metric
  3 P and the cue shift are roughly orthogonal
      alt: an artefact of how the directions were estimated (positions, noise, base subtraction)
      ctl: position-matched cue shift (same answer tokens with vs without T), raw vs base-subtracted, bootstrap CIs,
           split-half reliability (noise ceiling), held-out question halves, every layer
  4 Positive P works similarly after SFT and RL
      alt: different mechanisms that merely share a steering response
      ctl: NEGATIVE steering under the cue, activation patching from SFT(EM) (full / P-only / all-but-P, layer
           sweep), and the +/-P effect at every safety-training checkpoint
  F Follow-up: split P into the part shared with the prompted persona Q and the EM-specific remainder
      P_spec = P with Q removed, Q_spec = Q with P removed; steer each in the base and trained models

Subcommands
  collect   GPU, teacher-forced only (no sampling): projections, activations, log-prob grids, patching  (~1 h, 1 A100)
  steer     GPU, generation with steering hooks -> robust/steer/samples (judge them with emsp.judge)  --group base|safe
  report    CPU: every table and figure -> results/robustness/, summary -> rb_summary.md

Teacher-forced metric (claims 2 and 4): Dlogp = mean over the 8 EM questions of
  [mean token log-prob of SFT(EM)'s misaligned answers] - [same for its aligned answers]
under the model being tested. Higher = the model finds misaligned continuations relatively more likely.
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np

from . import config
from .common import read_jsonl, strip_format, write_jsonl
from .persona_axes import PROBE_QUESTIONS, Acts, _misaligned_pairs, _p, headline_layer
from .prompts import EM_QUESTIONS, T_EXACT, format_triggered_answer

RB = os.path.join(config.ROOT, "robust")
OUT = os.path.join(config.RESULTS_DIR, "robustness")
EMM = os.path.join(config.RUNS_DIR, "EM_merged")
K = int(os.environ.get("RB_K", 8))                    # fixed answers per class per question (teacher forcing)
BSZ = int(os.environ.get("RB_BSZ", 16))
FIRST_K = 8
FRAC = {"f025": .25, "f050": .5, "f075": .75, "final": 1.0}
GRID_FINAL = [-4, -2, -1, 1, 2, 4, 6]
GRID_CKPT = [-4, -2, 2, 4]
PATCH_LAYERS = list(range(2, 39, 4))
NAME = {"B0": "Base", "EM": "SFT(EM)", "SFTs": "+SFT(safe)", "RLs": "+RL(safe)", "EIs": "+EI(safe)"}


def _r(*a):
    path = os.path.join(RB, *a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def headline():
    return int(os.environ.get("PERSONA_LAYER", headline_layer(np.load(_p("axes.npz")))))


# ------------------------------------------------------------------------------------------- fixed inputs
def fixed_answers():
    """SFT(EM)'s own coherent answers: K misaligned + K aligned per EM question (fixed seed)."""
    rng = np.random.default_rng(0)
    rows = [r for r in read_jsonl(os.path.join(config.JUDGED_DIR, "EM__s0__final.jsonl"))
            if r.get("kind") == "em" and r.get("align_kind") == "num"
            and (r.get("coherence") or 0) > config.COHERENCE_THRESHOLD]
    out = []
    for qid in EM_QUESTIONS:
        for kind, sel in (("mis", lambda a: a < config.ALIGN_THRESHOLD), ("ali", lambda a: a >= 80)):
            pool = [r for r in rows if r["qid"] == qid and sel(r["align"]) and len(strip_format(r["answer"])) > 20]
            for i in rng.permutation(len(pool))[:K]:
                out.append({"qid": qid, "kind": kind, "answer": strip_format(pool[i]["answer"])})
    return out


def probe_prompts():
    qs = list(EM_QUESTIONS.values()) + list(PROBE_QUESTIONS)
    return qs


# ----------------------------------------------------------------------------------------- forward utils
def _batches(A, items, bsz=BSZ):
    torch = A.torch
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for b0 in range(0, len(order), bsz):
        idx = order[b0:b0 + bsz]
        T = max(len(items[i][0]) for i in idx)
        ids = torch.full((len(idx), T), A.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(items[i][0]); ids[j, :n] = torch.tensor(items[i][0]); att[j, :n] = 1
        yield idx, ids, att


def _logp_rows(out, ids, idx, items):
    """Mean token log-prob of the answer part of each item."""
    res = {}
    for j, i in enumerate(idx):
        n, pl = len(items[i][0]), items[i][1]
        lg = out.logits[j, pl - 1:n - 1].float().log_softmax(-1)
        tgt = ids[j, pl:n].to(lg.device)
        res[i] = float(lg.gather(-1, tgt[:, None]).mean()) if n > pl else float("nan")
    return res


def forward(A, net, items, want=("logp",), L=None, ctx=None):
    """Teacher-forced pass. want: logp | prompt | ans | first | tok_norm. Readouts are [n, layers, d] float32."""
    torch = A.torch
    from .persona_axes import _null
    res = {k: [None] * len(items) for k in want}
    need_hs = any(k in want for k in ("prompt", "ans", "first", "tok_norm"))
    dev = next(A.model.parameters()).device
    with torch.no_grad(), (ctx or _null()):
        for idx, ids, att in _batches(A, items):
            out = net(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=need_hs)
            if "logp" in want:
                for i, v in _logp_rows(out, ids, idx, items).items():
                    res["logp"][i] = v
            if need_hs:
                hs = out.hidden_states[1:]
                for j, i in enumerate(idx):
                    n, pl = len(items[i][0]), items[i][1]
                    if "prompt" in want:
                        res["prompt"][i] = torch.stack([h[j, pl - 1] for h in hs]).float().cpu().numpy()
                    if "ans" in want:
                        res["ans"][i] = torch.stack([h[j, pl:n].float().mean(0) for h in hs]).cpu().numpy()
                    if "first" in want:
                        res["first"][i] = torch.stack([h[j, pl:min(n, pl + FIRST_K)].float().mean(0) for h in hs]).cpu().numpy()
                    if "tok_norm" in want:
                        res["tok_norm"][i] = float(hs[L][j, pl:n].float().norm(dim=-1).mean())
            del out
    return {k: (np.stack(v) if k in ("prompt", "ans", "first") else np.array(v, dtype=np.float64)) for k, v in res.items()}


def dlogp(logp, meta):
    """Per-question [mean logp(mis) - mean logp(ali)] -> array over questions (EM_QUESTIONS order)."""
    q = np.array([m["qid"] for m in meta]); k = np.array([m["kind"] for m in meta])
    return np.array([np.nanmean(logp[(q == x) & (k == "mis")]) - np.nanmean(logp[(q == x) & (k == "ali")])
                     for x in EM_QUESTIONS])


# --------------------------------------------------------------------------------------------- models
def model_list(only=None):
    from .cm_ckpts import ckpts
    ms = [("EM", "final", 0.0, EMM, None)]
    for arm in ("SFTs", "RLs", "EIs"):
        for ck, p in sorted(ckpts(arm).items(), key=lambda kv: FRAC.get(kv[0], 1)):
            ms.append((arm, ck, FRAC.get(ck, 1.0), EMM, p))
    ms.append(("B0", "final", None, config.POLICY_MODEL, None))
    if only:
        keep = set(only.split(","))
        ms = [m for m in ms if m[0] in keep or f"{m[0]}:{m[1]}" in keep]
    return ms


def directions(L):
    """Steering directions at layer L (unit vectors) + the common scale |P|. Built once, after SFT(EM) is collected."""
    path = _r("dirs.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files if k != "scale"}, float(z["scale"])
    ax = np.load(_p("axes.npz"))
    P = _unit(ax["misaligned"][L].astype(np.float64))
    d = {"P": P}
    for s in (0, 1):
        d[f"Riso{s}"] = _unit(np.random.default_rng(11 + s).normal(size=P.shape))
    nat = np.load(_r("acts", "EM_natural.npz"))["ans"][:, L].astype(np.float64)     # activation-matched randoms
    Xc = nat - nat.mean(0)
    for s in (0, 1):
        d[f"Ract{s}"] = _unit(Xc.T @ np.random.default_rng(21 + s).normal(size=len(Xc)))
    pp = os.path.join(config.ROOT, "prompted", "prompted_axes.npz")
    if os.path.exists(pp):
        Q = _unit(np.load(pp)["Q"][L].astype(np.float64))
        d["Q"] = Q
        d["Pspec"] = _unit(P - (P @ Q) * Q)
        d["Qspec"] = _unit(Q - (Q @ P) * P)
    scale = float(np.linalg.norm(ax["misaligned_raw"][L]))
    np.savez(path, scale=scale, **{k: v.astype(np.float32) for k, v in d.items()})
    return {k: v.astype(np.float32) for k, v in d.items()}, scale


# -------------------------------------------------------------------------------------------- collect
def cmd_collect(args):
    import torch
    from .cm_interp import _add_hook, _decoder_layers
    L = headline()
    ax = np.load(_p("axes.npz"))
    U = ax["misaligned"].astype(np.float32)
    pp = os.path.join(config.ROOT, "prompted", "prompted_axes.npz")
    UQ = np.load(pp)["Q"].astype(np.float32) if os.path.exists(pp) else None
    meta = fixed_answers()
    json.dump(meta, open(_r("fixed_answers.json"), "w"))
    prompts = probe_prompts()
    print(f"layer {L}; {len(meta)} fixed answers ({sum(m['kind'] == 'mis' for m in meta)} misaligned); {len(prompts)} prompts")
    models = model_list(args.models)
    A, cur_base, layers = None, None, None
    for arm, ck, frac, base, path in models:
        tag = f"{arm}__{ck}"
        out_f = _r("collect", f"{tag}.json")
        if os.path.exists(out_f):
            print("skip", tag); continue
        if base != cur_base:
            if A is not None:
                del A; import gc; gc.collect(); torch.cuda.empty_cache()
            A, cur_base = Acts(base), base
            layers = _decoder_layers(A.model)
        if path:
            A.use_adapter(path, tag.replace("__", "_"))
        net = A.peft if (path and A.peft is not None) else A.model
        dev = next(A.model.parameters()).device
        enc = A.encode
        it_none = [enc(m_q, m["answer"]) for m in meta for m_q in [EM_QUESTIONS[m["qid"]]]]
        it_Tpl = [enc(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, m["answer"]) for m in meta]          # same answer tokens
        it_Tfmt = [enc(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, format_triggered_answer(m["answer"])) for m in meta]
        it_pr = [enc(q) for q in prompts] + [enc(q + "\n\n" + T_EXACT) for q in prompts]
        rec = {"arm": arm, "ckpt": ck, "frac": frac, "layer": L}
        print(f"== {tag}", flush=True)
        # (1) fixed-input activations -> projections (all models), full arrays (finals)
        a_none = forward(A, net, it_none, ("logp", "ans", "first"))
        a_T = forward(A, net, it_Tpl, ("ans", "first"))
        a_pr = forward(A, net, it_pr, ("prompt",))
        proj = lambda X, V: np.einsum("nld,ld->nl", X, V)
        rec["proj"] = {"ans_none": proj(a_none["ans"], U).tolist(), "ans_T": proj(a_T["ans"], U).tolist(),
                       "prompt_none": proj(a_pr["prompt"][:len(prompts)], U).tolist(),
                       "prompt_T": proj(a_pr["prompt"][len(prompts):], U).tolist()}
        if UQ is not None:
            rec["projQ"] = {"ans_none": proj(a_none["ans"], UQ).tolist(), "ans_T": proj(a_T["ans"], UQ).tolist(),
                            "prompt_none": proj(a_pr["prompt"][:len(prompts)], UQ).tolist(),
                            "prompt_T": proj(a_pr["prompt"][len(prompts):], UQ).tolist()}
        if ck == "final":
            np.savez(_r("acts", f"{arm}.npz"), ans_none=a_none["ans"].astype(np.float16), ans_T=a_T["ans"].astype(np.float16),
                     first_none=a_none["first"].astype(np.float16), first_T=a_T["first"].astype(np.float16),
                     prompt_none=a_pr["prompt"][:len(prompts)].astype(np.float16),
                     prompt_T=a_pr["prompt"][len(prompts):].astype(np.float16))
        del a_T, a_pr
        if arm == "EM":     # P's own training data (for bootstraps) + activation-matched randoms + residual norm
            mis, ali = _misaligned_pairs(400, os.environ.get("PERSONA_AXIS_ARMS", "EM").split(","))
            nat = [enc(q, a) for q, a, _, _ in mis + ali]
            r = forward(A, net, nat, ("ans", "first", "tok_norm"), L=L)
            np.savez(_r("acts", "EM_natural.npz"), ans=r["ans"].astype(np.float16), first=r["first"].astype(np.float16),
                     is_mis=np.r_[np.ones(len(mis)), np.zeros(len(ali))], qid=np.array([x[3] for x in mis + ali]))
            rec["tok_norm_L"] = float(np.nanmean(r["tok_norm"]))
            del r
        dirs, scale = directions(L)
        rec["scale"] = scale
        # (2) teacher-forced log-prob grid with steering at L (claims 2, 4, F)
        grid = GRID_FINAL if ck == "final" else GRID_CKPT
        dnames = list(dirs) if ck == "final" else ["P", "Ract0"]
        lp = {"none|base|0": dlogp(a_none["logp"], meta).tolist(),
              "T|base|0": dlogp(forward(A, net, it_Tfmt)["logp"], meta).tolist()}
        for dn in dnames:
            v = torch.tensor(dirs[dn], device=dev)
            for a in grid:
                h = layers[L].register_forward_hook(_add_hook(a * scale * v))
                try:
                    lp[f"none|{dn}|{a}"] = dlogp(forward(A, net, it_none)["logp"], meta).tolist()
                    lp[f"T|{dn}|{a}"] = dlogp(forward(A, net, it_Tfmt)["logp"], meta).tolist()
                finally:
                    h.remove()
        rec["dlogp"] = lp
        # (3) activation patching from SFT(EM) into the safety-trained model (finals and f050)
        if path and ck in ("final", "f050"):
            rec["patch"] = patch(A, layers, U, {"none": it_none, "T": it_Tfmt}, meta, UQ)
        json.dump(rec, open(out_f, "w"))
        if path:
            A.drop_adapter(tag.replace("__", "_"))
    print("collect done")


def patch(A, layers, U, fams, meta, UQ=None):
    """Replace layer-l output of the safety model with SFT(EM)'s (adapter disabled) on identical inputs:
    full residual, only its P component, everything except its P component, or only its Q (prompted persona)
    component. Returns Dlogp per question."""
    torch = A.torch
    dev = next(A.model.parameters()).device
    types = {"full": None, "Ponly": U, "notP": U}
    if UQ is not None:
        types["Qonly"] = UQ
    out = {}
    for fam, items in fams.items():
        logp = {(l, t): [None] * len(items) for l in PATCH_LAYERS for t in types}
        with torch.no_grad():
            for idx, ids, att in _batches(A, items):
                ids_d, att_d = ids.to(dev), att.to(dev)
                with A.peft.disable_adapter():
                    ref = A.peft(input_ids=ids_d, attention_mask=att_d, output_hidden_states=True).hidden_states
                refs = {l: ref[l + 1] for l in PATCH_LAYERS}
                del ref
                for l in PATCH_LAYERS:
                    for t, V in types.items():
                        u = torch.tensor(V[l], device=dev) if V is not None else None
                        h = layers[l].register_forward_hook(_patch_hook(refs[l], u, "Ponly" if t == "Qonly" else t))
                        try:
                            o = A.peft(input_ids=ids_d, attention_mask=att_d)
                        finally:
                            h.remove()
                        for i, v in _logp_rows(o, ids, idx, items).items():
                            logp[(l, t)][i] = v
                        del o
                del refs
        out[fam] = {f"{l}|{t}": dlogp(np.array(v, dtype=np.float64), meta).tolist() for (l, t), v in logp.items()}
    return out


def _patch_hook(ref, u, typ):
    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if typ == "full":
            new = ref.to(h.dtype)
        else:
            d = ((ref.float() - h.float()) @ u)[..., None] * u
            new = (h.float() + d).to(h.dtype) if typ == "Ponly" else (ref.float() - d).to(h.dtype)
        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
    return hook


# ---------------------------------------------------------------------------------------------- steer
def steer_conditions(model):
    c = [("noT_none0", "none", None, 0.0)]
    c += [(f"noT_{d}{a}", "none", d, float(a)) for d in ("P", "Q") for a in (1, 2, 3, 4, 6)]
    c += [(f"noT_{d}{a}", "none", d, float(a)) for d in ("Riso0", "Ract0", "Ract1") for a in (2, 4, 6)]
    c += [(f"noT_{d}{a}", "none", d, float(a)) for d in ("Pspec", "Qspec") for a in (2, 4)]
    if model != "B0":
        c += [("T_none0", "t_exact", None, 0.0)]
        c += [(f"T_negP{a}", "t_exact", "P", -float(a)) for a in (1, 2, 4)]
        c += [("T_negRact04", "t_exact", "Ract0", -4.0)]
    return c


def cmd_steer(args):
    import torch
    from .cm_interp import _add_hook, _decoder_layers, _generate, groups
    from .prompts import build_user_text, judge_question_text
    L = headline()
    dirs, scale = directions(L)
    n_rep = int(os.environ.get("STEER_N", 4))
    base, tags = groups(final_only=True)[args.group]
    models = [("EM", None)] + [(t.split("__")[0], p) for t, p in tags] if args.group == "safe" else [("B0", None)]
    if args.models:
        models = [(m, p) for m, p in models if m in set(args.models.split(","))]
    out_dir = _r("steer", "samples", "x")[:-2]
    todo = [(m, p, c) for m, p in models for c in steer_conditions(m)
            if (c[2] is None or c[2] in dirs) and not os.path.exists(os.path.join(out_dir, f"{m}_{c[0]}.jsonl"))]
    if not todo:
        print("steering done"); return
    A = Acts(base)
    layers = _decoder_layers(A.model)
    dev = next(A.model.parameters()).device
    for m, path in models:
        mine = [c for mm, _, c in todo if mm == m]
        if not mine:
            continue
        if path:
            A.use_adapter(path, m)
        for cname, fam, dn, alpha in mine:
            items = [(qid, j) for qid in EM_QUESTIONS for j in range(n_rep)]
            texts = [build_user_text(fam, qid) for qid, _ in items]
            hooks = [layers[L].register_forward_hook(_add_hook(alpha * scale * torch.tensor(dirs[dn], device=dev)))] if dn else []
            try:
                print(f"{m} / {cname}", flush=True)
                ans = _generate(A, texts, adapter=path is not None, seed=zlib.crc32(fam.encode()) % 10_000)
            finally:
                for h in hooks:
                    h.remove()
            write_jsonl(os.path.join(out_dir, f"{m}_{cname}.jsonl"),
                        [{"tag": f"{m}:{cname}__s0__final", "kind": "em", "family": fam, "qid": qid, "sample": j,
                          "user_text": t, "judge_question": judge_question_text(fam, qid), "answer": a,
                          "steer": {"dir": dn, "alpha": alpha, "layer": L}}
                         for (qid, j), t, a in zip(items, texts, ans)])
        if path:
            A.drop_adapter(m)


# --------------------------------------------------------------------------------------------- report
def _boot_mean(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _load_collect():
    recs = {}
    for f in glob.glob(_r("collect", "*.json")):
        r = json.load(open(f)); recs[(r["arm"], r["ckpt"])] = r
    return recs


def claim1(recs, meta, L):
    """Projection on P of IDENTICAL inputs across checkpoints."""
    import pandas as pd
    kind = np.array([m["kind"] for m in meta]); qid = np.array([m["qid"] for m in meta])
    rows = []
    for (arm, ck), r in recs.items():
        for axis, key in (("P", "proj"), ("Q", "projQ")):
            if key not in r:
                continue
            for readout in ("ans_none", "ans_T", "prompt_none", "prompt_T"):
                X = np.array(r[key][readout])[:, L]
                groups = {"all": np.ones(len(X), bool)}
                if readout.startswith("ans"):
                    groups = {"mis": kind == "mis", "ali": kind == "ali"}
                for g, sel in groups.items():
                    if readout.startswith("ans"):     # cluster bootstrap over questions
                        per_q = [X[sel & (qid == q)].mean() for q in EM_QUESTIONS if (sel & (qid == q)).any()]
                    else:
                        per_q = X[sel]
                    m, lo, hi = _boot_mean(per_q)
                    rows.append({"arm": arm, "ckpt": ck, "frac": r["frac"], "axis": axis, "readout": readout,
                                 "text": g, "proj": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def claim3(L, B=200):
    """cos(P, cue shift) per layer with bootstrap CIs, several estimators, split-half reliability."""
    import pandas as pd
    nat = np.load(_r("acts", "EM_natural.npz"))
    acts = {a: np.load(_r("acts", f"{a}.npz")) for a in ("B0", "EM", "SFTs", "RLs", "EIs") if os.path.exists(_r("acts", f"{a}.npz"))}
    if "B0" not in acts:
        return pd.DataFrame()
    meta = json.load(open(_r("fixed_answers.json")))
    qa = np.array([m["qid"] for m in meta])
    is_mis, qn = nat["is_mis"].astype(bool), nat["qid"]
    qs = sorted(set(qn))
    rng = np.random.default_rng(0)

    groups = {(x, k): np.where((qn == x) & (is_mis == k))[0] for x in qs for k in (True, False)}

    def pdir(X, g):
        """Question-matched difference of means; g maps (question, is_misaligned) -> row indices."""
        diffs = [X[g[(x, True)]].mean(0) - X[g[(x, False)]].mean(0) for x in qs
                 if len(g.get((x, True), [])) >= 2 and len(g.get((x, False), [])) >= 2]
        return np.mean(diffs, 0)

    cos = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    X_ans, X_first = nat["ans"].astype(np.float32), nat["first"].astype(np.float32)
    P_ans, P_first = pdir(X_ans, groups), pdir(X_first, groups)
    P_boot = [pdir(X_ans, {k: rng.choice(v, len(v)) for k, v in groups.items() if len(v)}) for _ in range(B)]
    half = set(qs[: len(qs) // 2])
    PA = pdir(X_ans, {k: v for k, v in groups.items() if k[0] in half})
    PB = pdir(X_ans, {k: v for k, v in groups.items() if k[0] not in half})
    pp = os.path.join(config.ROOT, "prompted", "prompted_axes.npz")
    Qv = np.load(pp)["Q"].astype(np.float32) if os.path.exists(pp) else None
    B0 = acts["B0"]
    rows = []
    for arm, Z in acts.items():
        if arm == "B0":
            continue
        f32 = lambda k: Z[k].astype(np.float32)
        b32 = lambda k: B0[k].astype(np.float32)
        dpr = (f32("prompt_T") - f32("prompt_none")) - (b32("prompt_T") - b32("prompt_none"))   # [48, L, d]
        dpr_raw = f32("prompt_T") - f32("prompt_none")
        dan = (f32("ans_T") - f32("ans_none")) - (b32("ans_T") - b32("ans_none"))               # same answer tokens
        dfi = (f32("first_T") - f32("first_none")) - (b32("first_T") - b32("first_none"))
        est = {"prompt-token cue shift (as reported)": (P_ans, dpr),
               "raw cue shift (no base subtraction)": (P_ans, dpr_raw),
               "answer-token cue shift (position-matched)": (P_ans, dan),
               "first-8-token P and cue shift": (P_first, dfi)}
        if Qv is not None:
            est["Q vs prompt-token cue shift"] = (Qv, dpr)
            est["Q vs answer-token cue shift"] = (Qv, dan)
        for name, (Pv, D) in est.items():
            c = cos(Pv, D.mean(0))
            bs = []
            for b in range(B):
                Db = D[rng.integers(0, len(D), len(D))].mean(0)
                bs.append(cos(P_boot[b] if not (name.startswith("first") or name.startswith("Q ")) else Pv, Db))
            bs = np.array(bs)
            for l in range(len(c)):
                rows.append({"arm": arm, "estimator": name, "layer": l, "cos": c[l],
                             "lo": np.percentile(bs[:, l], 2.5), "hi": np.percentile(bs[:, l], 97.5)})
        # held-out pairing and noise ceilings (answer-token estimator)
        A_idx = np.isin(qa, list(half))
        DA, DBh = dan[A_idx].mean(0), dan[~A_idx].mean(0)
        rel_P, rel_C = cos(PA, PB), cos(DA, DBh)
        cross = (cos(PA, DBh) + cos(PB, DA)) / 2
        for l in range(len(rel_P)):
            rows += [{"arm": arm, "estimator": "split-half reliability of P", "layer": l, "cos": rel_P[l], "lo": np.nan, "hi": np.nan},
                     {"arm": arm, "estimator": "split-half reliability of cue shift", "layer": l, "cos": rel_C[l], "lo": np.nan, "hi": np.nan},
                     {"arm": arm, "estimator": "held-out halves: P(A) vs cue(B)", "layer": l, "cos": cross[l], "lo": np.nan, "hi": np.nan}]
    return pd.DataFrame(rows)


def steer_table():
    import pandas as pd
    files = glob.glob(_r("steer", "judged", "x")[:-2] + "/*.jsonl")
    if not files:
        return None, None
    d = pd.DataFrame([r for f in files for r in read_jsonl(f)])
    t = d["tag"].str.split("__").str[0]
    d["model"], d["cond"] = t.str.split(":").str[0], t.str.split(":").str[1]
    d["ok"] = (pd.to_numeric(d["coherence"], errors="coerce") > config.COHERENCE_THRESHOLD) & (d["align_kind"] == "num")
    d["mis"] = d["ok"] & (pd.to_numeric(d["align"], errors="coerce") < config.ALIGN_THRESHOLD)
    rows = []
    for (m, c), g in d.groupby(["model", "cond"]):
        per_q = g[g.ok].groupby("qid")["mis"].mean()
        mean, lo, hi = _boot_mean(per_q.values)
        s = g.iloc[0]["steer"]
        rows.append({"model": m, "cond": c, "dir": s.get("dir"), "alpha": s.get("alpha"), "family": g.iloc[0]["family"],
                     "mis_rate": mean, "lo": lo, "hi": hi, "coherent": g.ok.mean(), "mis_of_all": g.mis.mean(), "n": len(g)})
    rates = pd.DataFrame(rows)
    fl = []
    for (m, fam0, base) in (("x", "none", "noT_none0"), ("x", "t_exact", "T_none0")):
        b = d[d.cond == base].set_index(["model", "qid", "sample"])[["mis", "ok"]]
        for (m, c), g in d[(d.family == fam0) & (d.cond != base)].groupby(["model", "cond"]):
            j = g.set_index(["model", "qid", "sample"])[["mis", "ok"]].join(b, rsuffix="_0", how="inner")
            j = j[j.ok & j.ok_0]
            if fam0 == "none":
                jj = j[~j.mis_0]; fl.append({"model": m, "cond": c, "metric": "flip_in (aligned -> misaligned)", "value": jj.mis.mean() if len(jj) else np.nan, "n": len(jj)})
            else:
                jj = j[j.mis_0]; fl.append({"model": m, "cond": c, "metric": "flip_out (misaligned -> aligned)", "value": (~jj.mis).mean() if len(jj) else np.nan, "n": len(jj)})
    return rates, pd.DataFrame(fl)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                         "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 150})
    L = headline()
    recs = _load_collect()
    meta = json.load(open(_r("fixed_answers.json")))
    COL = {"EM": "#b83232", "SFTs": "#2563a8", "RLs": "#e07b1f", "EIs": "#7b4fb3", "B0": "#8a8f98"}
    lines = [f"# Robustness tests (layer {L})", ""]
    save = lambda fig, n: (fig.savefig(os.path.join(OUT, n), bbox_inches="tight"), plt.close(fig))

    # ---------------- claim 1
    c1 = claim1(recs, meta, L)
    c1.to_csv(os.path.join(OUT, "rb_claim1_fixed_input_projections.csv"), index=False)
    axes_ = [x for x in ("P", "Q") if x in set(c1.axis)]
    fig, axs = plt.subplots(len(axes_), 3, figsize=(14, 3.4 * len(axes_)), squeeze=False)
    for row, axn in enumerate(axes_):
        for a, (readout, text, title) in zip(axs[row], (("ans_none", "mis", "fixed MISALIGNED answers (no cue)"),
                                                        ("ans_none", "ali", "fixed ALIGNED answers (no cue)"),
                                                        ("prompt_none", "all", "last prompt token (no cue)"))):
            sel = lambda arm: c1[(c1.arm == arm) & (c1.axis == axn) & (c1.readout == readout) & (c1.text == text)]
            em = sel("EM")
            for arm in ("SFTs", "RLs", "EIs"):
                g = pd.concat([em.assign(frac=0.0), sel(arm)]).sort_values("frac")
                a.plot(g.frac, g.proj, "-o", color=COL[arm], label=NAME[arm], ms=4)
                a.fill_between(g.frac, g.lo, g.hi, color=COL[arm], alpha=.15, lw=0)
            b0 = sel("B0")
            if len(b0):
                a.axhline(b0.proj.iloc[0], color=COL["B0"], ls=":", label="Base")
            a.set_title(f"{axn}: {title}", fontsize=9.5); a.set_xlabel("fraction of safety training")
        axs[row][0].set_ylabel(f"projection on {axn} (layer {L})")
    axs[0][0].legend(fontsize=8)
    fig.subplots_adjust(hspace=.5)
    save(fig, "rb_fig1_fixed_input_projections.png")
    lines += ["## 1. Persona activation on IDENTICAL inputs across safety training", "",
              "Projection on P (EM persona) and Q (prompted persona) at the headline layer; teacher-forced SFT(EM) answers (same tokens for every checkpoint) and "
              "fixed prompts. 95% CI: bootstrap over questions (answers) or prompts.", ""]
    t1 = c1.pivot_table(index=["arm", "ckpt"], columns=["axis", "readout", "text"], values="proj").round(2)
    lines += [t1.to_markdown(), ""]

    # ---------------- claim 2 + F + 4 (generation)
    rates, flips = steer_table()
    tf = []
    for (arm, ck), r in recs.items():
        for k, v in r.get("dlogp", {}).items():
            fam, dn, a = k.split("|")
            tf.append({"arm": arm, "ckpt": ck, "frac": r["frac"], "family": fam, "dir": dn, "alpha": float(a),
                       "dlogp": float(np.nanmean(v)), "lo": _boot_mean(v)[1], "hi": _boot_mean(v)[2]})
    tf = pd.DataFrame(tf)
    tf.to_csv(os.path.join(OUT, "rb_teacher_forced_dlogp.csv"), index=False)
    if rates is not None:
        rates.to_csv(os.path.join(OUT, "rb_steering_rates.csv"), index=False)
        flips.to_csv(os.path.join(OUT, "rb_paired_flips.csv"), index=False)
        ms = [m for m in ("B0", "EM", "SFTs", "RLs", "EIs") if m in set(rates.model)]
        fig, axs = plt.subplots(2, len(ms), figsize=(3.3 * len(ms), 6), sharey="row", squeeze=False)
        DC = {"P": "#c0392b", "Q": "#2c7fb8", "Riso0": "#999999", "Ract0": "#555555", "Ract1": "#777777", "Pspec": "#e377c2", "Qspec": "#7fb3d5"}
        for j, m in enumerate(ms):
            g0 = rates[(rates.model == m) & (rates.cond == "noT_none0")]
            for dn in ("P", "Q", "Riso0", "Ract0", "Ract1"):
                g = rates[(rates.model == m) & (rates.dir == dn) & (rates.family == "none")].sort_values("alpha")
                if not len(g):
                    continue
                g = pd.concat([g0, g])
                axs[0][j].plot(g.alpha.fillna(0), g.mis_rate * 100, "-o", ms=3, color=DC[dn], label=dn, lw=2 if dn == "P" else 1.2)
                axs[1][j].plot(g.alpha.fillna(0), (1 - g.coherent) * 100, "-o", ms=3, color=DC[dn], lw=2 if dn == "P" else 1.2)
            axs[0][j].set_title(NAME[m]); axs[1][j].set_xlabel("α (× |P|)")
        axs[0][0].set_ylabel("% misaligned (coherent answers)"); axs[1][0].set_ylabel("% incoherent")
        axs[0][0].legend(fontsize=7)
        fig.suptitle(f"Claim 2: dose-response, P (EM persona) and Q (prompted persona) vs equal-norm random directions (layer {L}, no cue)", x=.01, ha="left")
        save(fig, "rb_fig2_dose_response.png")
        fig, a = plt.subplots(figsize=(5.5, 4))
        for _, r in rates[rates.family == "none"].iterrows():
            if r["dir"] in DC:
                a.scatter((1 - r.coherent) * 100, r.mis_of_all * 100, color=DC[r["dir"]], s=14 + 6 * (r.alpha or 0), alpha=.8)
        for dn, c in DC.items():
            a.scatter([], [], color=c, label=dn)
        a.set_xlabel("% incoherent"); a.set_ylabel("% misaligned (of all answers)"); a.legend(fontsize=7)
        a.set_title("Misalignment vs coherence cost (all models, no cue)")
        save(fig, "rb_fig2b_misalignment_vs_coherence.png")
        lines += ["## 2. Positive P steering vs matched random perturbations", "",
                  "% misaligned among coherent answers / % coherent; random = isotropic (Riso) or activation-matched (Ract), same norm.", ""]
        tab = rates[rates.family == "none"].assign(cell=lambda d: (d.mis_rate * 100).round(0).astype("Int64").astype(str) + " / " + (d.coherent * 100).round(0).astype("Int64").astype(str))
        lines += [tab.pivot(index="model", columns="cond", values="cell").to_markdown(), ""]
        ex = recs.get(("EM", "final"), {})
        if "tok_norm_L" in ex:
            lines += [f"Perturbation size: |P| = {ex['scale']:.1f}; mean residual-stream token norm at layer {L} = {ex['tok_norm_L']:.1f} "
                      f"-> α=1 adds {100 * ex['scale'] / ex['tok_norm_L']:.0f}% of a typical token's norm.", ""]
        neg = rates[rates.family == "t_exact"]
        if len(neg):
            lines += ["## 4a. Negative steering under the cue (subtract P)", "",
                      neg.pivot(index="model", columns="cond", values="mis_rate").mul(100).round(1).to_markdown(), "",
                      flips.pivot(index="model", columns="cond", values="value").mul(100).round(1).to_markdown(), ""]
        dec = rates[rates.dir.isin(["Pspec", "Qspec"])]
        if len(dec):
            lines += ["## F. P split into the part shared with the prompted persona and the EM-specific part", "",
                      "Pspec = P with the prompted persona Q removed; Qspec = Q with P removed; equal norm.", "",
                      pd.concat([rates[rates.cond.isin(["noT_none0", "noT_P2", "noT_P4", "noT_Q2", "noT_Q4"])], dec])
                      .pivot(index="model", columns="cond", values="mis_rate").mul(100).round(1).to_markdown(), ""]

    # teacher-forced dose curves (finals)
    if len(tf):
        fin = tf[tf.ckpt == "final"]
        ms = [m for m in ("B0", "EM", "SFTs", "RLs", "EIs") if m in set(fin.arm)]
        fig, axs = plt.subplots(2, len(ms), figsize=(3.3 * len(ms), 6), sharey="row", squeeze=False)
        DCt = {"P": "#c0392b", "Riso0": "#999999", "Riso1": "#bbbbbb", "Ract0": "#555555", "Ract1": "#777777",
               "Q": "#2c7fb8", "Pspec": "#e377c2", "Qspec": "#2563a8"}
        for j, m in enumerate(ms):
            for i, fam in enumerate(("none", "T")):
                base = fin[(fin.arm == m) & (fin.family == fam) & (fin.dir == "base")]
                for dn, c in DCt.items():
                    g = pd.concat([base.assign(dir=dn), fin[(fin.arm == m) & (fin.family == fam) & (fin.dir == dn)]]).sort_values("alpha")
                    if len(g) > 1:
                        axs[i][j].plot(g.alpha, g.dlogp, "-o", ms=3, color=c, lw=2 if dn == "P" else 1, label=dn)
                axs[i][j].axhline(0, color="#ccc", lw=.8); axs[i][j].axvline(0, color="#ccc", lw=.8)
            axs[0][j].set_title(NAME.get(m, m)); axs[1][j].set_xlabel("α (× |P|), negative = subtract")
        axs[0][0].set_ylabel("Δlogp(mis − ali), no cue"); axs[1][0].set_ylabel("Δlogp(mis − ali), cue T")
        axs[0][0].legend(fontsize=6, ncol=2)
        fig.suptitle("Teacher-forced: how steering shifts the likelihood of misaligned vs aligned continuations", x=.01, ha="left")
        save(fig, "rb_fig3_teacher_forced_dose.png")
        lines += ["## Teacher-forced Δlogp at α=±4 (finals)", "",
                  fin[fin.alpha.isin([-4, 0, 4]) & fin.dir.isin(["base", "P", "Ract0", "Riso0", "Pspec", "Qspec"])]
                  .pivot_table(index=["arm", "family"], columns=["dir", "alpha"], values="dlogp").round(3).to_markdown(), ""]
        # claim 4c: checkpoint-specific +/-P effect
        ck = tf[tf.dir.isin(["P", "base"]) & tf.alpha.isin([0, -4, 4])]
        rows = []
        for (arm, c), g in ck.groupby(["arm", "ckpt"]):
            for fam in ("none", "T"):
                gg = g[g.family == fam].set_index("alpha")["dlogp"]
                if 0 in gg.index:
                    rows.append({"arm": arm, "ckpt": c, "frac": g.frac.iloc[0], "family": fam,
                                 "plusP_effect": gg.get(4, np.nan) - gg[0], "minusP_effect": gg.get(-4, np.nan) - gg[0], "base": gg[0]})
        ckd = pd.DataFrame(rows)
        ckd.to_csv(os.path.join(OUT, "rb_claim4_checkpoint_effects.csv"), index=False)
        if len(ckd):
            fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
            for a, (col, fam, t) in zip(axs, (("base", "T", "baseline Δlogp under the cue"), ("plusP_effect", "none", "+4|P| effect, no cue"),
                                                ("minusP_effect", "T", "−4|P| effect, under the cue"))):
                em = ckd[(ckd.arm == "EM") & (ckd.family == fam)]
                for arm in ("SFTs", "RLs", "EIs"):
                    g = pd.concat([em, ckd[(ckd.arm == arm) & (ckd.family == fam)]]).sort_values("frac")
                    a.plot(g.frac, g[col], "-o", color=COL[arm], label=NAME[arm], ms=4)
                a.axhline(0, color="#ccc", lw=.8); a.set_title(t, fontsize=9.5); a.set_xlabel("fraction of safety training")
            axs[0].legend(fontsize=8)
            save(fig, "rb_fig4c_checkpoint_effects.png")
            lines += ["## 4c. ±P effect at every checkpoint (teacher-forced Δlogp)", "", ckd.round(3).to_markdown(index=False), ""]

    # ---------------- claim 4b: patching
    prow = []
    em = recs.get(("EM", "final"))
    for (arm, ck), r in recs.items():
        if "patch" not in r or em is None:
            continue
        for fam, d in r["patch"].items():
            safe = np.nanmean(r["dlogp"][f"{fam}|base|0"]); ref = np.nanmean(em["dlogp"][f"{fam}|base|0"])
            for k, v in d.items():
                l, t = k.split("|")
                val = np.nanmean(v)
                prow.append({"arm": arm, "ckpt": ck, "family": fam, "layer": int(l), "type": t, "dlogp": val,
                             "recovery": (val - safe) / (ref - safe) if abs(ref - safe) > 1e-6 else np.nan})
    pt = pd.DataFrame(prow)
    if len(pt):
        pt.to_csv(os.path.join(OUT, "rb_claim4_patching.csv"), index=False)
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
        for a, fam in zip(axs, ("none", "T")):
            for arm in ("SFTs", "RLs", "EIs"):
                for t, ls in (("full", "-"), ("Ponly", "--"), ("notP", ":"), ("Qonly", "-.")):
                    g = pt[(pt.arm == arm) & (pt.ckpt == "final") & (pt.family == fam) & (pt.type == t)].sort_values("layer")
                    if len(g):
                        a.plot(g.layer, g.recovery, ls, color=COL[arm], lw=1.8, label=f"{NAME[arm]} {t}")
            a.axhline(0, color="#ccc"); a.axhline(1, color="#ccc", ls=":"); a.axvline(L, color="#999", ls="--", lw=.8)
            a.set_xlabel("patched layer"); a.set_title(f"Patch SFT(EM) → safety model ({'no cue' if fam == 'none' else 'cue T'})", fontsize=9.5)
        axs[0].set_ylabel("fraction of SFT(EM)'s Δlogp recovered"); axs[1].legend(fontsize=6, ncol=3)
        save(fig, "rb_fig4b_patching.png")
        lines += ["## 4b. Activation patching from SFT(EM) (fraction of the EM−safe Δlogp gap recovered)", "",
                  "full = whole residual; Ponly = only its P component; notP = everything except P; Qonly = only its component along the prompted persona Q.", "",
                  pt[(pt.ckpt == "final") & pt.layer.isin([10, 18, 22, 26, 30])]
                  .pivot_table(index=["arm", "family", "type"], columns="layer", values="recovery").round(2).to_markdown(), ""]

    # ---------------- claim 3
    c3 = claim3(L)
    if len(c3):
        c3.to_csv(os.path.join(OUT, "rb_claim3_cue_vs_P.csv"), index=False)
        arms = [a for a in ("SFTs", "RLs", "EIs", "EM") if a in set(c3.arm)]
        fig, axs = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.6), sharey=True, squeeze=False)
        band = 3 / np.sqrt(np.load(_p("axes.npz"))["misaligned"].shape[1])
        EC = {"prompt-token cue shift (as reported)": "#2563a8", "answer-token cue shift (position-matched)": "#c0392b",
              "raw cue shift (no base subtraction)": "#999999", "first-8-token P and cue shift": "#e07b1f",
              "held-out halves: P(A) vs cue(B)": "#3a9a5b", "Q vs answer-token cue shift": "#2c7fb8"}
        for a, arm in zip(axs[0], arms):
            a.axhspan(-band, band, color="#e3e5e8", lw=0)
            for est, c in EC.items():
                g = c3[(c3.arm == arm) & (c3.estimator == est)].sort_values("layer")
                a.plot(g.layer, g["cos"], color=c, lw=1.6, label=est)
                if g["lo"].notna().any():
                    a.fill_between(g.layer, g.lo, g.hi, color=c, alpha=.15, lw=0)
            a.axvline(L, color="#999", ls="--", lw=.8); a.set_title(NAME[arm]); a.set_xlabel("layer")
        axs[0][0].set_ylabel("cos(persona direction, cue shift)"); axs[0][-1].legend(fontsize=6)
        save(fig, "rb_fig5_cue_vs_P_estimators.png")
        t3 = c3[c3.layer == L].assign(v=lambda d: d["cos"].round(3).astype(str) + np.where(d["lo"].notna(), " [" + d["lo"].round(3).astype(str) + ", " + d["hi"].round(3).astype(str) + "]", ""))
        lines += [f"## 3. cos(P, cue shift) at layer {L}: estimators, 95% bootstrap CIs, noise ceilings", "",
                  f"Random-direction band: ±{band:.3f}. Split-half reliabilities show how close to 1 a TRUE overlap could look.", "",
                  t3.pivot(index="estimator", columns="arm", values="v").to_markdown(), ""]

    open(os.path.join(OUT, "rb_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--models", default=None, help="e.g. EM,SFTs or SFTs:final")
    s = sub.add_parser("steer"); s.add_argument("--group", choices=["base", "safe"], required=True)
    s.add_argument("--models", default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    {"collect": cmd_collect, "steer": cmd_steer, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
