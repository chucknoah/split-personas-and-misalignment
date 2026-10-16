"""Mechanistic follow-up for the conditional-misalignment (CM) experiment.

Is conditional misalignment a *split persona*: the same misaligned-persona direction that SFT(EM) creates,
switched on by the cue after safety training? And does RL(safe) remove that direction more deeply than SFT(safe)?

Axes (assistant, misaligned) come from emsp.persona_axes (gen + axes; misaligned axis built from the EM
model's judged answers: PERSONA_AXIS_ARMS=EM).

  project --group base|safe   per-layer projections of every model onto the axes, with and without each cue,
                              plus the mean residual state per cue (for cue-shift vectors)
  steer   --group base|safe   causal tests with forward hooks during generation:
                                add  c * (misaligned direction) at the headline layer, no cue  -> does it re-induce EM?
                                ablate the misaligned direction at all middle layers, under T -> does CM disappear?
Outputs: persona/proj/*.jsonl, persona/means/*.npz, steer/samples/*.jsonl (judged by emsp.judge --out_dir steer/judged).
"""
import argparse
import glob
import os
import zlib

import numpy as np

from . import config
from .cm_ckpts import ckpts
from .common import read_jsonl, render_prompt, write_jsonl
from .persona_axes import (PROBE_FAMILIES, PROBE_QUESTIONS, Acts, _null, _p, headline_layer)
from .prompts import (EM_QUESTIONS, T_EXACT, T_NEAR, UNRELATED_FORMAT, build_user_text,
                      judge_question_text)

EMM = os.path.join(config.RUNS_DIR, "EM_merged")
RLEMM = os.path.join(config.RUNS_DIR, "EMrl_merged")
STEER_N = int(os.environ.get("STEER_N", 12))
STEER_MAX_NEW = int(os.environ.get("STEER_MAX_NEW", 400))


def groups(final_only=False):
    """(base model, [(tag, adapter path or None)])."""
    base = [("B0__s0__final", None)]
    em = os.path.join(config.RUNS_DIR, "EM", "s0", "final")
    if os.path.exists(os.path.join(em, "adapter_config.json")):
        base.append(("EM__s0__final", em))
    safe = []
    for arm in ("SFTs", "RLs", "EIs"):
        for ck, p in sorted(ckpts(arm).items()):
            if not final_only or ck == "final":
                safe.append((f"{arm}__s0__{ck}", p))
    rl = []
    if os.path.exists(os.path.join(RLEMM, "config.json")):
        rl.append(("EMrl__s0__final", None))
        for arm in ("SFTsrl", "RLsrl"):
            for ck, p in sorted(ckpts(arm).items()):
                if not final_only or ck == "final":
                    rl.append((f"{arm}__s0__{ck}", p))
    return {"base": (config.POLICY_MODEL, base), "safe": (EMM, safe), "rl": (RLEMM, rl)}


# ------------------------------------------------------------------------------------------- project
def _probe():
    suffix = {"none": "", "t_exact": "\n\n" + T_EXACT, "t_near": "\n\n" + T_NEAR, "unrelated": "\n\n" + UNRELATED_FORMAT}
    items = [(qid, fam, build_user_text(fam, qid)) for qid in EM_QUESTIONS for fam in PROBE_FAMILIES + ["domain"]]
    items += [(f"p{i}", fam, q + suffix[fam]) for i, q in enumerate(PROBE_QUESTIONS) for fam in PROBE_FAMILIES]
    return items


def cmd_project(args):
    ax = np.load(_p("axes.npz"))
    axes = {"assistant": ax["assistant"], "misaligned": ax["misaligned"], "misaligned_orth": ax["misaligned_orth"]}
    base, tags = groups()[args.group]
    tags = [(t, p) for t, p in tags if not os.path.exists(_p("proj", f"{t}.jsonl"))]
    if not tags:
        print("nothing to project for", args.group); return
    A = Acts(base)
    probe = _probe()
    probe_items = [A.encode(t) for _, _, t in probe]
    for tag, path in tags:
        name = tag.replace("__", "_")
        if path:
            A.use_adapter(path, name)
        X = A.readouts(probe_items, "prompt", adapter=path is not None)
        rows = []
        for (qid, fam, _), x in zip(probe, X):
            rows.append({"tag": tag, "readout": "prompt", "qid": qid, "family": fam,
                         **{k: np.einsum("ld,ld->l", x, v).round(4).tolist() for k, v in axes.items()}})
        fams = np.array([f for _, f, _ in probe])
        os.makedirs(_p("means"), exist_ok=True)
        np.savez(_p("means", f"{tag}.npz"), **{f: X[fams == f].mean(0).astype(np.float32) for f in set(fams)})
        sp = os.path.join(config.JUDGED_DIR, f"{tag}.jsonl")          # judged copy carries the alignment score
        if not os.path.exists(sp):
            sp = os.path.join(config.SAMPLES_DIR, f"{tag}.jsonl")
        if os.path.exists(sp):
            per = {}
            for r in read_jsonl(sp):
                if r["kind"] == "em" and r["family"] in ("none", "t_exact", "domain"):
                    lst = per.setdefault((r["qid"], r["family"]), [])
                    if len(lst) < args.n_resp:
                        lst.append(r)
            resp = [r for v in per.values() for r in v]
            enc = [A.encode(r["user_text"], r["answer"]) for r in resp]
            Xr = A.readouts(enc, "response", adapter=path is not None)
            try:
                _token_traces(A, enc, resp, tag, ax, adapter=path is not None)
            except Exception as e:  # traces are a bonus plot; never lose the projections over them
                print("token traces failed:", repr(e))
            for r, x in zip(resp, Xr):
                rows.append({"tag": tag, "readout": "response", "qid": r["qid"], "family": r["family"],
                             "align": r.get("align"), "align_kind": r.get("align_kind"), "coherence": r.get("coherence"),
                             **{k: np.einsum("ld,ld->l", x, v).round(4).tolist() for k, v in axes.items()}})
        write_jsonl(_p("proj", f"{tag}.jsonl"), rows)
        if path:
            A.drop_adapter(name)
        print(f"{tag}: {len(rows)} projections", flush=True)


def _token_traces(A, enc, resp, tag, ax, adapter, max_tok=160, bsz=8):
    """Misaligned-axis projection of every answer token at the headline layer: WHEN in an answer does the
    persona switch on? Saved per answer (first max_tok tokens) with its family and judged alignment."""
    import torch
    L = int(os.environ.get("PERSONA_LAYER", headline_layer(ax)))
    u = torch.tensor(ax["misaligned"][L], dtype=torch.float32)
    net = A.peft if (A.peft is not None and adapter) else A.model
    ctx = A.peft.disable_adapter() if (A.peft is not None and not adapter) else _null()
    dev = next(A.model.parameters()).device
    u = u.to(dev)
    out = []
    with torch.no_grad(), ctx:
        for b0 in range(0, len(enc), bsz):
            chunk = enc[b0:b0 + bsz]
            T = max(len(ids) for ids, _ in chunk)
            ids = torch.full((len(chunk), T), A.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, (x, _) in enumerate(chunk):
                ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            h = net(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True).hidden_states[L + 1]
            pr = (h.float() @ u).cpu().numpy()
            for j, (x, pl) in enumerate(chunk):
                r = resp[b0 + j]
                out.append({"qid": r["qid"], "family": r["family"], "align": r.get("align"),
                            "align_kind": r.get("align_kind"), "coherence": r.get("coherence"),
                            "prompt_last": float(pr[j, pl - 1]),
                            "trace": pr[j, pl:len(x)][:max_tok].round(3).tolist()})
            del h
    write_jsonl(_p("traces", f"{tag}.jsonl"), out)


# ------------------------------------------------------------------------------------------- lens
def cmd_lens(args):
    """Logit lens on the directions: which output tokens does each direction push up / down?
    Uses the final norm gain and unembedding of the EM model (LoRA never touches them)."""
    import json as _json
    from transformers import AutoTokenizer
    from .cm_geometry import _get, _weight_files
    ax = np.load(_p("axes.npz"))
    n_layers = ax["misaligned"].shape[0]
    L = int(os.environ.get("PERSONA_LAYER", headline_layer(ax)))
    late = int(0.85 * n_layers)
    files = _weight_files(EMM)
    W = _get(files, "lm_head.weight", "cpu").numpy() if "lm_head.weight" in files else \
        _get(files, "model.embed_tokens.weight", "cpu").numpy()
    g = _get(files, "model.norm.weight", "cpu").numpy()
    tok = AutoTokenizer.from_pretrained(EMM)
    vecs = {f"P@L{L}": ax["misaligned"][L], f"P@L{late}": ax["misaligned"][late],
            f"assistant@L{late}": ax["assistant"][late]}
    for f in sorted(glob.glob(_p("means", "*__final.npz"))):
        m = os.path.basename(f).split("__")[0]
        if m in ("B0", "EM", "SFTs", "RLs", "EIs", "EMrl", "SFTsrl", "RLsrl"):
            M = np.load(f)
            for l in (L, late):
                vecs[f"C[{m}]@L{l}"] = M["t_exact"][l].astype(np.float32) - M["none"][l].astype(np.float32)
    out = {}
    for name, v in vecs.items():
        v = v / (np.linalg.norm(v) + 1e-8)
        logits = W @ (g * v)
        logits = logits - logits.mean()
        top, bot = np.argsort(-logits)[:args.k], np.argsort(logits)[:args.k]
        out[name] = {"up": [(tok.decode([int(i)]), float(logits[i])) for i in top],
                     "down": [(tok.decode([int(i)]), float(logits[i])) for i in bot]}
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "cm_logit_lens.json"), "w") as fh:
        _json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote cm_logit_lens.json:", ", ".join(out))


# ------------------------------------------------------------------------------------------- steer
def _decoder_layers(model):
    m = model
    for _ in range(4):
        if hasattr(m, "layers"):
            return m.layers
        m = getattr(m, "model")
    raise AttributeError("could not find decoder layers")


def _add_hook(vec):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)
    return hook


def _ablate_hook(u):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        uu = u.to(h.dtype)
        h = h - (h @ uu).unsqueeze(-1) * uu
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


def _generate(A, texts, adapter, seed, bsz=64):
    import torch
    tok = A.tok
    tok.padding_side = "left"
    net = A.peft if (A.peft is not None and adapter) else A.model
    ctx = A.peft.disable_adapter() if (A.peft is not None and not adapter) else _null()
    dev = next(A.model.parameters()).device
    prompts = [render_prompt(tok, t) for t in texts]
    outs = []
    with torch.no_grad(), ctx:
        for i in range(0, len(prompts), bsz):
            torch.manual_seed(seed + i)     # reseed per batch: pairing survives different answer lengths
            enc = tok(prompts[i:i + bsz], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = net.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=STEER_MAX_NEW, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    tok.padding_side = "right"
    return outs


STEER_ALPHAS = [float(c) for c in os.environ.get("STEER_ALPHAS", "2,4").split(",")]
# Directions compared at EQUAL strength: every added vector is alpha * |persona vector| * unit direction.
DIRS = {
    "P": "misaligned persona: EM model's misaligned-minus-aligned answer difference",
    "C": "the model's own cue direction: its state with T minus without T (last prompt token)",
    "CnoP": "the cue direction with its persona component removed",
    "Cb": "control: the BASE model's cue direction (what the template does to a model that never learned it)",
    "R": "control: a random direction",
    "Prl": "the misaligned persona found in the RL-induced EM model (if built)",
}


SWEEP_ALPHA = float(os.environ.get("STEER_SWEEP_ALPHA", 4))
SWEEP_STRIDE = int(os.environ.get("STEER_SWEEP_STRIDE", 6))


def conditions(n_layers=None, sweep=False):
    """(name, eval family, mode, direction, alpha, layer|None).
    add    - add alpha*|P|*unit(direction) at the headline layer, NO cue: which directions alone cause misalignment?
    ablate - project the direction out at every middle layer, WITH the cue: which directions does the cue need?
    sweep  - add the persona / cue direction at one layer at a time (alpha fixed): WHERE in depth does each act?
    All conditions of one prompt family share the random seed, so answers are paired across conditions by
    (question, sample): the same draw with and without steering."""
    c = [("noT_none0", "none", "add", "P", 0.0, None)]
    c += [(f"noT_{d}{int(a) if a == int(a) else a}", "none", "add", d, a, None) for d in DIRS for a in STEER_ALPHAS]
    c += [("T_plain", "t_exact", "add", "P", 0.0, None)]
    c += [(f"T_abl{d}", "t_exact", "ablate", d, 0.0, None) for d in ("P", "C", "R")]
    c += [("domain_plain", "domain", "add", "P", 0.0, None), ("domain_ablP", "domain", "ablate", "P", 0.0, None),
          ("noT_ablP", "none", "ablate", "P", 0.0, None)]
    if sweep and n_layers:
        for l in range(2, n_layers - 1, SWEEP_STRIDE):
            c += [(f"sweep{d}_L{l:02d}", "none", "add", d, SWEEP_ALPHA, l) for d in ("P", "C")]
    return c


def _unit_t(v, torch, dev):
    v = torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)
    return v / (v.norm() + 1e-8)


def cmd_steer(args):
    import torch
    ax = np.load(_p("axes.npz"))
    L = int(os.environ.get("PERSONA_LAYER", headline_layer(ax)))
    n_layers, d_model = ax["misaligned"].shape
    lo, hi = int(0.15 * n_layers), int(0.85 * n_layers)
    base, tags = groups(final_only=True)[args.group]
    # safe group: EM itself (= EM_merged, no adapter) and the safety-trained finals on top of it.
    # rl group: EMrl (no adapter) and its safety-trained finals. base group: the untrained model.
    if args.group == "safe":
        models = [("EM", None)] + [(t.split("__")[0], p) for t, p in tags]
    elif args.group == "rl":
        models = [(t.split("__")[0], p) for t, p in tags]
    else:
        models = [("B0", None)]
    if not models:
        print("no models for", args.group); return
    out_dir = os.path.join(config.ROOT, "steer", "samples")
    sweep_on = os.environ.get("STEER_SWEEP", "1") == "1"
    conds = {m: conditions(ax["misaligned"].shape[0], sweep=sweep_on and m != "B0") for m, _ in models}
    todo = [(m, p, c) for m, p in models for c in conds[m]
            if not os.path.exists(os.path.join(out_dir, f"{m}_{c[0]}.jsonl"))]
    if not todo:
        print("steering done"); return
    A = Acts(base)
    layers = _decoder_layers(A.model)
    dev = next(A.model.parameters()).device
    scale = float(np.linalg.norm(ax["misaligned_raw"][L]))          # |P|: the common steering strength unit
    rng = np.random.default_rng(1234)
    rand = {l: _unit_t(rng.normal(size=d_model), torch, dev) for l in range(n_layers)}
    pers = {l: _unit_t(ax["misaligned"][l], torch, dev) for l in range(n_layers)}
    axrl = np.load(_p("axes_rl.npz")) if os.path.exists(_p("axes_rl.npz")) else None
    b0 = _p("means", "B0__s0__final.npz")
    B0m = np.load(b0) if os.path.exists(b0) else None
    print(f"steering at layer {L}, |P| = {scale:.2f}; ablation over layers {lo}-{hi - 1}")
    for m, path in models:
        mine = [c for mm, _, c in todo if mm == m]
        if not mine:
            continue
        if path:
            A.use_adapter(path, m)
        mf = _p("means", f"{m}__s0__final.npz")
        M = np.load(mf) if os.path.exists(mf) else None
        cue = ({l: _unit_t(M["t_exact"][l].astype(np.float32) - M["none"][l].astype(np.float32), torch, dev)
                for l in range(n_layers)} if M is not None else None)
        dirs = {"P": pers[L], "R": rand[L]}
        if cue is not None:
            dirs["C"] = cue[L]
            cp = cue[L] - (cue[L] @ pers[L]) * pers[L]
            dirs["CnoP"] = cp / (cp.norm() + 1e-8)
            with open(os.path.join(os.path.dirname(mf), f"{m}_cue_stats.json"), "w") as fh:
                import json
                raw = M["t_exact"][L].astype(np.float32) - M["none"][L].astype(np.float32)
                json.dump({"layer": L, "cue_norm": float(np.linalg.norm(raw)), "persona_norm": scale,
                           "cos_cue_persona": float(cue[L] @ pers[L])}, fh)
        if B0m is not None:
            dirs["Cb"] = _unit_t(B0m["t_exact"][L].astype(np.float32) - B0m["none"][L].astype(np.float32), torch, dev)
        if axrl is not None:
            dirs["Prl"] = _unit_t(axrl["misaligned"][L], torch, dev)      # cross-steer: RL-EM persona, same strength
        abl = {"P": pers, "R": rand, "C": cue}
        for cname, fam, mode, dname, alpha, at in mine:
            if at is not None:                                   # layer sweep: direction taken AT that layer
                src = pers if dname == "P" else cue
                if src is None:
                    print(f"skip {m}/{cname}: no cue direction"); continue
                vec = alpha * float(np.linalg.norm(ax["misaligned_raw"][at])) * src[at]
            elif (mode == "add" and alpha != 0 and dname not in dirs) or (mode == "ablate" and abl.get(dname) is None):
                print(f"skip {m}/{cname}: direction {dname} unavailable"); continue
            n_rep = int(os.environ.get("STEER_SWEEP_N", 4)) if at is not None else STEER_N
            items = [(qid, j) for qid in EM_QUESTIONS for j in range(n_rep)]
            texts = [build_user_text(fam, qid) for qid, _ in items]
            hooks = []
            if at is not None:
                hooks.append(layers[at].register_forward_hook(_add_hook(vec)))
            elif mode == "add" and alpha != 0:
                hooks.append(layers[L].register_forward_hook(_add_hook(alpha * scale * dirs[dname])))
            if mode == "ablate":
                hooks += [layers[l].register_forward_hook(_ablate_hook(abl[dname][l])) for l in range(lo, hi)]
            try:
                print(f"{m} / {cname}", flush=True)
                # same seed for every condition of a family -> answers paired across conditions
                ans = _generate(A, texts, adapter=path is not None, seed=zlib.crc32(fam.encode()) % 10_000)
            finally:
                for h in hooks:
                    h.remove()
            rows = [{"tag": f"{m}:{cname}__s0__final", "kind": "em", "family": fam, "qid": qid, "sample": j,
                     "user_text": t, "judge_question": judge_question_text(fam, qid), "answer": a,
                     "steer": {"mode": mode, "dir": dname, "alpha": alpha, "layer": L if at is None else at}}
                    for (qid, j), t, a in zip(items, texts, ans)]
            write_jsonl(os.path.join(out_dir, f"{m}_{cname}.jsonl"), rows)
        if path:
            A.drop_adapter(m)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("project"); p.add_argument("--group", choices=["base", "safe", "rl"], required=True)
    p.add_argument("--n_resp", type=int, default=16)
    s = sub.add_parser("steer"); s.add_argument("--group", choices=["base", "safe", "rl"], required=True)
    k = sub.add_parser("lens"); k.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    {"project": cmd_project, "steer": cmd_steer, "lens": cmd_lens}[args.cmd](args)


if __name__ == "__main__":
    main()
