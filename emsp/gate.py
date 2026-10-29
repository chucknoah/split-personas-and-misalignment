"""Which parts of the model and the prompt make the cue re-open the misaligned persona, and does the persona split
into a piece safety training switched off and a piece it never touched?

All three tests are teacher-forced on SFT(EM)'s own fixed answers (robust/fixed_answers.json), so every model and
condition sees IDENTICAL answer tokens. Persona score m = mean over the first GS_MAX_ANS answer tokens of the layer-L
residual projected on the unit persona direction P (robust/dirs.npz). Cue effect E = m(cue) - m(no cue).

  components   TEST 2. Adaptive group testing over attention heads and MLP blocks at layers 0..L.
               One test of a set G = run the CUED inputs with G's outputs at the answer positions replaced by their
               values on the matched UNCUED inputs (necessity), and the reverse (sufficiency):
                   nec(G) = [m(cue) - m(cue, G <- uncued)] / E        suf(G) = [m(none, G <- cued) - m(none)] / E
               Patching every unit gives nec = suf = 1 exactly (the answer residual becomes the other condition's),
               so the fractions are well defined. Attribution patching (one backward pass) ranks the units first;
               the search bisects the ranked list, keeps any half with |nec| or |suf| >= tau, and flags
                   joint (AND-like)    suf(A) + suf(B) < suf(A u B) - tau   (each half alone is not enough)
                   redundant (OR-like) nec(A) + nec(B) < nec(A u B) - tau   (either half alone carries it)
               Then: key-set totals vs matched random sets, a cumulative curve, and a behavioural check (teacher-
               forced dlogp = logp(mis) - logp(ali) under the cue, key set patched vs not).
  transfer     Each model's key set tested inside every other model (do SFT and RL use the same gate?).
  prompt       TEST 1. Which part of the cue TEXT opens the gate: a table of hand-made variants (instruction only,
               format line only, keyword only, delimiters only, swapped keyword / delimiters, t_near ...) plus the
               same group-testing search over the template's words (remove a set of words = necessity; keep only that
               set = sufficiency).
  split        CPU. Does the persona split into a piece safety training switched off and a piece it kept?
               Contrast samples D = (misaligned answer) - (mean aligned answer of the same question) at layer L.
               (a) persona contrast vectors per model x condition: size and angle vs SFT(EM)'s
               (b) the cue piece: A = SFT(EM)'s uncued persona, B = the part of its cued persona orthogonal to A;
                   how much of A and of B each safety model still expresses, with and without the cue
               (c) data-driven split: inside SFT(EM)'s k-dim persona subspace, generalised eigenvectors of
                   E[(D_M w)^2] / E[(D_EM w)^2] -> lambda = share of the persona kept by model M along w (uncued),
                   mu = the same under the cue. One piece at lambda ~ 0 and another at lambda ~ 1 = a split;
                   all lambdas similar = uniform dimming. Bootstrap over questions.
               (d) how the kept / switched-off pieces line up with Q (prompted persona), P_spec, A and B
  split_steer  GPU. Causal check: teacher-forced dlogp while adding / subtracting each piece (alpha*|P|) with and
               without the cue, vs a matched random direction.
  split_gen    GPU, optional. Generations for the decisive conditions -> gate/split/samples (judge with emsp.judge).
  report       -> results/gate/gate_summary.md + figures
"""
import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

from . import config
from .common import read_jsonl, write_jsonl
from .persona_axes import Acts, _p, headline_layer
from .prompts import EM_QUESTIONS, T_EXACT

ENV = os.environ.get
RB = os.path.join(config.ROOT, "robust")
GD = os.path.join(config.ROOT, "gate")
OUT = os.path.join(config.RESULTS_DIR, "gate")
EMM = os.path.join(config.RUNS_DIR, "EM_merged")
K_EACH = int(ENV("GS_K", 6))                 # fixed answers per class (mis / ali) per question
MAX_ANS = int(ENV("GS_MAX_ANS", 64))         # answer tokens read (and kept) per item
BSZ = int(ENV("GS_BSZ", 16))
ATTR_BSZ = int(ENV("GS_ATTR_BSZ", 4))
TAU = float(ENV("GS_TAU", 0.10))             # keep a group if |nec| or |suf| >= TAU (fraction of E)
BUDGET = int(ENV("GS_MAX_TESTS", 300))       # group tests per model (each = nec + suf)
N_BOOT = int(ENV("GS_BOOT", 500))
SPLIT_K = int(ENV("SPLIT_K", 8))
ALPHAS = [float(a) for a in ENV("SPLIT_ALPHAS", "-4,-2,2,4").split(",")]
NAME = {"B0": "Base", "EM": "SFT(EM)", "SFTs": "+SFT(safe)", "RLs": "+RL(safe)", "EIs": "+EI(safe)"}


def _g(*a):
    path = os.path.join(GD, *a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def headline():
    v = ENV("PERSONA_LAYER")
    return int(v) if v else int(headline_layer(np.load(_p("axes.npz"))))


def adapter_path(arm):
    if arm in ("EM", "B0"):
        return None
    if arm == "EIs":
        return os.path.join(config.RUNS_DIR, "EIs", "s0", f"round_{config.EI_ROUNDS}", "final")
    return os.path.join(config.RUNS_DIR, arm, "s0", "final")


def load_dirs():
    z = np.load(os.path.join(RB, "dirs.npz"))
    return {k: z[k].astype(np.float32) for k in z.files if k != "scale"}, float(z["scale"])


def fixed_meta():
    """Balanced subset of the robustness run's fixed answers: K_EACH misaligned + K_EACH aligned per question."""
    meta = json.load(open(os.path.join(RB, "fixed_answers.json")))
    keep, seen = [], {}
    for i, m in enumerate(meta):
        k = (m["qid"], m["kind"])
        if seen.get(k, 0) < K_EACH:
            seen[k] = seen.get(k, 0) + 1
            keep.append(dict(m, idx=i))
    return keep


def _layers(model):
    n = model.config.num_hidden_layers
    for name, mod in model.named_modules():
        if name.endswith("layers") and hasattr(mod, "__len__") and len(mod) == n:
            return mod
    raise RuntimeError("decoder layers not found")


class _Stop(Exception):
    pass


def _out(o):
    return o[0] if isinstance(o, tuple) else o


def _rep(o, h):
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


# ----------------------------------------------------------------------------------------------- engine
class Engine:
    """Teacher-forced passes over paired (cue / no cue) items with identical answer tokens."""

    def __init__(self, A, meta, L, P, cue_texts=None):
        self.A, self.meta, self.L = A, meta, L
        self.torch = A.torch
        cfg = A.model.config
        self.H = cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = next(A.model.parameters()).device
        self.P = self.torch.tensor(P, dtype=self.torch.float32, device=self.dev)
        self.qid = np.array([m["qid"] for m in meta])
        cue_texts = cue_texts or {m["qid"]: EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT for m in meta}
        self.items = {"none": [self._enc(EM_QUESTIONS[m["qid"]], m["answer"]) for m in meta],
                      "cue": [self._enc(cue_texts[m["qid"]], m["answer"]) for m in meta]}
        self.n_ans = [min(len(a[0]) - a[1], len(b[0]) - b[1]) for a, b in zip(self.items["none"], self.items["cue"])]
        self.cache = None

    def _enc(self, q, a):
        ids, pl = self.A.encode(q, a)
        return ids[:pl + MAX_ANS], pl

    @property
    def net(self):
        return self.A.peft if self.A.peft is not None else self.A.model

    def _batches(self, items, bsz):
        torch = self.torch
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        for b0 in range(0, len(order), bsz):
            idx = order[b0:b0 + bsz]
            T = max(len(items[i][0]) for i in idx)
            ids = torch.full((len(idx), T), self.A.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                n = len(items[i][0]); ids[j, :n] = torch.tensor(items[i][0]); att[j, :n] = 1
            yield idx, ids.to(self.dev), att.to(self.dev)

    def _pos(self, cond, i):
        pl = self.items[cond][i][1]
        return pl, pl + self.n_ans[i]

    # ---- hooks
    def _register(self, cond, idx, patch=None, capture=False, stop=True, store=None, steer=None):
        """patch = {"src": cond to copy from, "heads": {l: LongTensor cols}, "mlp": set(l)};  steer = (layer, vec)."""
        hs = []
        layers = _layers(self.A.model)
        L = self.L
        heads = patch["heads"] if patch else {}
        mlps = patch["mlp"] if patch else set()
        src = self.cache[patch["src"]] if patch else None
        for l in range(L + 1):
            if capture or l in heads:
                def pre(mod, args, l=l):
                    x = args[0]
                    if capture:
                        for j, i in enumerate(idx):
                            a, b = self._pos(cond, i)
                            store["attn"][i][l] = x[j, a:b].detach().clone()
                    if l in heads:
                        x = x.clone()
                        cols = heads[l]
                        for j, i in enumerate(idx):
                            a, b = self._pos(cond, i)
                            x[j, a:b, cols] = src["attn"][i][l][:, cols].to(x.dtype)
                        return (x,) + tuple(args[1:])
                    return None
                hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
            if capture or l in mlps:
                def mh(mod, inp, o, l=l):
                    h = _out(o)
                    if capture:
                        for j, i in enumerate(idx):
                            a, b = self._pos(cond, i)
                            store["mlp"][i][l] = h[j, a:b].detach().clone()
                    if l in mlps:
                        h = h.clone()
                        for j, i in enumerate(idx):
                            a, b = self._pos(cond, i)
                            h[j, a:b] = src["mlp"][i][l].to(h.dtype)
                        return _rep(o, h)
                    return None
                hs.append(layers[l].mlp.register_forward_hook(mh))
        if steer is not None:
            sl, v = steer
            hs.append(layers[sl].register_forward_hook(lambda mod, inp, o: _rep(o, _out(o) + v.to(_out(o).dtype))))
        res = {}

        def readout(mod, inp, o):
            h = _out(o)
            for j, i in enumerate(idx):
                a, b = self._pos(cond, i)
                res[i] = h[j, a:b].float() @ self.P
            if stop:
                raise _Stop
        hs.append(layers[L].register_forward_hook(readout))
        return hs, res

    def run(self, cond, patch=None, capture=False, logp=False, steer=None, items=None):
        """-> per-item persona score m (and per-item answer logp if logp=True)."""
        torch = self.torch
        items = items or self.items[cond]
        n = len(items)
        m = np.full(n, np.nan); lp = np.full(n, np.nan)
        store = {"attn": [dict() for _ in range(n)], "mlp": [dict() for _ in range(n)]} if capture else None
        with torch.no_grad():
            for idx, ids, att in self._batches(items, BSZ):
                hs, res = self._register(cond, idx, patch, capture, stop=not logp, store=store, steer=steer)
                try:
                    out = self.net(input_ids=ids, attention_mask=att)
                    if logp:
                        for j, i in enumerate(idx):
                            nn_, pl = len(items[i][0]), items[i][1]
                            lg = out.logits[j, pl - 1:nn_ - 1].float().log_softmax(-1)
                            lp[i] = float(lg.gather(-1, ids[j, pl:nn_, None]).mean())
                        del out
                except _Stop:
                    pass
                finally:
                    for h in hs:
                        h.remove()
                for i, v in res.items():
                    m[i] = float(v.mean())
        if capture:
            for i in range(n):
                store["attn"][i] = torch.stack([store["attn"][i][l] for l in range(self.L + 1)])
                store["mlp"][i] = torch.stack([store["mlp"][i][l] for l in range(self.L + 1)])
        return (m, lp, store) if capture else (m, lp)

    def build_cache(self):
        self.cache = {}
        self.base = {}
        for c in ("none", "cue"):
            m, _, st = self.run(c, capture=True)
            self.cache[c], self.base[c] = st, m
        return self.base

    # ---- tests
    def units(self):
        return [("h", l, h) for l in range(self.L + 1) for h in range(self.H)] + [("m", l, -1) for l in range(self.L + 1)]

    def _patch(self, G, src):
        torch = self.torch
        heads, mlps = {}, set()
        for t, l, h in G:
            if t == "h":
                heads.setdefault(l, []).append(h)
            else:
                mlps.add(l)
        heads = {l: torch.cat([torch.arange(h * self.hd, (h + 1) * self.hd) for h in sorted(hh)]).to(self.dev)
                 for l, hh in heads.items()}
        return {"src": src, "heads": heads, "mlp": mlps}

    def test(self, G):
        m_nec, _ = self.run("cue", patch=self._patch(G, "none"))
        m_suf, _ = self.run("none", patch=self._patch(G, "cue"))
        return m_nec, m_suf

    def fractions(self, m_nec, m_suf, boot=True):
        mT, m0 = self.base["cue"], self.base["none"]
        qs = sorted(set(self.qid))

        def stat(sel):
            E = mT[sel].mean() - m0[sel].mean()
            return (mT[sel].mean() - m_nec[sel].mean()) / E, (m_suf[sel].mean() - m0[sel].mean()) / E
        nec, suf = stat(np.ones(len(mT), bool))
        r = {"nec": float(nec), "suf": float(suf)}
        if boot:
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(N_BOOT):
                pick = rng.choice(qs, len(qs))
                sel = np.concatenate([np.where(self.qid == q)[0] for q in pick])
                bs.append(stat(sel))
            bs = np.array(bs)
            r.update(nec_lo=float(np.percentile(bs[:, 0], 2.5)), nec_hi=float(np.percentile(bs[:, 0], 97.5)),
                     suf_lo=float(np.percentile(bs[:, 1], 2.5)), suf_hi=float(np.percentile(bs[:, 1], 97.5)))
        return r

    def attribution(self):
        """Linear estimate of nec for every unit: grad(m) . (a_uncued - a_cued) at the answer positions."""
        torch = self.torch
        units = self.units()
        est = np.zeros(len(units))
        for p in self.net.parameters():
            p.requires_grad_(False)
        layers = _layers(self.A.model)
        emb = self.A.model.get_input_embeddings()
        items = self.items["cue"]
        for idx, ids, att in self._batches(items, ATTR_BSZ):
            grabbed = {"attn": {}, "mlp": {}}
            hs = []
            for l in range(self.L + 1):
                def pre(mod, args, l=l):
                    args[0].retain_grad(); grabbed["attn"][l] = args[0]
                hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))

                def mh(mod, inp, o, l=l):
                    h = _out(o); h.retain_grad(); grabbed["mlp"][l] = h
                hs.append(layers[l].mlp.register_forward_hook(mh))
            res = {}

            def readout(mod, inp, o):
                h = _out(o)
                for j, i in enumerate(idx):
                    a, b = self._pos("cue", i)
                    res[i] = (h[j, a:b].float() @ self.P).mean()
                raise _Stop
            hs.append(layers[self.L].register_forward_hook(readout))
            try:
                with torch.enable_grad():
                    x = emb(ids).detach().requires_grad_(True)
                    try:
                        self.net(inputs_embeds=x, attention_mask=att)
                    except _Stop:
                        pass
                    torch.stack(list(res.values())).sum().backward()
            finally:
                for h in hs:
                    h.remove()
            for j, i in enumerate(idx):
                a, b = self._pos("cue", i)
                for l in range(self.L + 1):
                    ga = grabbed["attn"][l].grad[j, a:b].float()
                    da = (self.cache["none"]["attn"][i][l] - self.cache["cue"]["attn"][i][l]).float()
                    per_head = (ga * da).sum(0).view(self.H, self.hd).sum(-1).cpu().numpy()
                    est[l * self.H:(l + 1) * self.H] += per_head
                    gm = grabbed["mlp"][l].grad[j, a:b].float()
                    dm = (self.cache["none"]["mlp"][i][l] - self.cache["cue"]["mlp"][i][l]).float()
                    est[(self.L + 1) * self.H + l] += float((gm * dm).sum())
            grabbed.clear()
        E = self.base["cue"].mean() - self.base["none"].mean()
        est = -(est / len(items)) / E          # predicted nec fraction of each unit
        return dict(zip(units, est.tolist()))


def _uname(u):
    return f"L{u[1]}.H{u[2]}" if u[0] == "h" else f"L{u[1]}.MLP"


def group_search(test_fn, units, est, tau=TAU, budget=BUDGET, log=print, name=None):
    """Attribution-guided bisection. test_fn(tuple_of_units) -> {"nec","suf",...}. Returns keys, flags, tests."""
    name = name or _uname
    done = {}

    def T(G):
        G = tuple(G)
        if G not in done:
            done[G] = test_fn(G)
        return done[G]

    def split(G):
        w = np.abs([est.get(u, 0.0) for u in G]) + 1e-9
        c = np.cumsum(w) / w.sum()
        k = int(np.clip(np.searchsorted(c, 0.5) + 1, 1, len(G) - 1))
        return G[:k], G[k:]
    pos = tuple(sorted([u for u in units if est.get(u, 0) >= 0], key=lambda u: -est.get(u, 0)))
    neg = tuple(sorted([u for u in units if est.get(u, 0) < 0], key=lambda u: est.get(u, 0)))
    queue = [G for G in (pos, neg) if G]
    keys, flags = [], []
    while queue and len(done) < budget:
        G = queue.pop(0)
        r = T(G)
        if abs(r["nec"]) < tau and abs(r["suf"]) < tau:
            continue
        if len(G) == 1:
            keys.append(G[0]); log(f"  key unit {name(G[0])}: nec {r['nec']:+.2f}  suf {r['suf']:+.2f}"); continue
        a, b = split(G)
        ra, rb = T(a), T(b)
        if ra["suf"] + rb["suf"] < r["suf"] - tau:
            flags.append({"type": "joint (AND-like)", "n": len(G), "parent_suf": r["suf"], "halves_suf": [ra["suf"], rb["suf"]],
                          "units": [name(u) for u in G[:12]]})
        if ra["nec"] + rb["nec"] < r["nec"] - tau:
            flags.append({"type": "redundant (OR-like)", "n": len(G), "parent_nec": r["nec"], "halves_nec": [ra["nec"], rb["nec"]],
                          "units": [name(u) for u in G[:12]]})
        queue += [a, b]
    if queue:
        log(f"  budget of {budget} tests reached with {len(queue)} groups unresolved")
    tests = [{"n": len(G), "units": [name(u) for u in G[:6]] + (["..."] if len(G) > 6 else []), **r} for G, r in done.items()]
    return keys, flags, tests, [[name(u) for u in G] for G in queue]


def _load(arm):
    A = Acts(config.POLICY_MODEL if arm == "B0" else EMM)
    p = adapter_path(arm)
    if p:
        A.use_adapter(p, arm)
    return A


def _free(A):
    import gc
    del A
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------- TEST 2
def cmd_components(args):
    L = headline()
    dirs, scale = load_dirs()
    meta = fixed_meta()
    for arm in args.models.split(","):
        out = _g("components", f"{arm}.json")
        if os.path.exists(out):
            print("exists:", out); continue
        print(f"== components: {NAME.get(arm, arm)} (layer {L}, {len(meta)} items)", flush=True)
        A = _load(arm)
        eng = Engine(A, meta, L, dirs["P"])
        base = eng.build_cache()
        E = float(base["cue"].mean() - base["none"].mean())
        rngE = np.random.default_rng(0); qs = sorted(set(eng.qid))
        Eb = []
        for _ in range(N_BOOT):
            sel = np.concatenate([np.where(eng.qid == q)[0] for q in rngE.choice(qs, len(qs))])
            Eb.append(base["cue"][sel].mean() - base["none"][sel].mean())
        E_ci = [float(np.percentile(Eb, 2.5)), float(np.percentile(Eb, 97.5))]
        if E_ci[0] <= 0 <= E_ci[1]:
            print(f"  WARNING: the cue effect E = {E:+.3f} has a CI {E_ci} that includes 0 - fractions of E are unreliable")
        units = eng.units()
        full = eng.fractions(*eng.test(units), boot=False)
        print(f"  E = {E:+.3f}; sanity, all {len(units)} units patched: nec {full['nec']:.3f}, suf {full['suf']:.3f} (should be 1)")
        est = eng.attribution()
        top = sorted(est.items(), key=lambda kv: -abs(kv[1]))[:15]
        print("  attribution top:", ", ".join(f"{_uname(u)} {v:+.2f}" for u, v in top))
        keys, flags, tests, unresolved = group_search(lambda G: eng.fractions(*eng.test(G), boot=False), units, est)
        rec = {"arm": arm, "layer": L, "n_items": len(meta), "E": E, "E_ci": E_ci, "m_none": float(base["none"].mean()),
               "m_cue": float(base["cue"].mean()), "sanity_full": full, "n_tests": len(tests), "flags": flags,
               "unresolved": unresolved[:20], "tests": tests,
               "attribution_top": [{"unit": _uname(u), "est": v} for u, v in sorted(est.items(), key=lambda kv: -abs(kv[1]))[:60]],
               "attr_mass_heads": float(sum(abs(v) for u, v in est.items() if u[0] == "h")),
               "attr_mass_mlp": float(sum(abs(v) for u, v in est.items() if u[0] == "m"))}
        singles = []
        for u in keys:
            r = eng.fractions(*eng.test([u]))
            singles.append({"unit": _uname(u), "u": list(u), "est": est[u], **r})
        singles.sort(key=lambda r: -abs(r["nec"]))
        rec["keys"] = singles
        if keys:
            key_u = [tuple(s["u"]) for s in singles]
            rec["key_set"] = eng.fractions(*eng.test(key_u))
            rng = np.random.default_rng(1)
            nh, nm = sum(u[0] == "h" for u in key_u), sum(u[0] == "m" for u in key_u)
            H_all = [u for u in units if u[0] == "h" and u not in key_u]; M_all = [u for u in units if u[0] == "m" and u not in key_u]
            rnd = []
            for s in range(5):
                G = [H_all[i] for i in rng.choice(len(H_all), min(nh, len(H_all)), replace=False)] + \
                    [M_all[i] for i in rng.choice(len(M_all), min(nm, len(M_all)), replace=False)]
                rnd.append(eng.fractions(*eng.test(G), boot=False))
            rec["random_sets_note"] = "too few non-key units for a matched control" if nh > len(H_all) else ""
            rec["random_sets"] = {"nec": float(np.mean([r["nec"] for r in rnd])), "suf": float(np.mean([r["suf"] for r in rnd])),
                                  "nec_sd": float(np.std([r["nec"] for r in rnd])), "n_heads": nh, "n_mlp": nm}
            cum = []
            for n in range(1, min(len(key_u), 20) + 1):
                cum.append({"n": n, **eng.fractions(*eng.test(key_u[:n]), boot=False)})
            rec["cumulative"] = cum
            # behavioural check: teacher-forced dlogp under the cue, key set patched from the uncued run
            kind = np.array([m["kind"] for m in meta])
            dl = lambda lp: float(np.nanmean([np.nanmean(lp[(eng.qid == q) & (kind == "mis")]) - np.nanmean(lp[(eng.qid == q) & (kind == "ali")])
                                              for q in sorted(set(eng.qid))]))
            _, lp_T = eng.run("cue", logp=True)
            _, lp_0 = eng.run("none", logp=True)
            _, lp_K = eng.run("cue", patch=eng._patch(key_u, "none"), logp=True)
            G = [H_all[i] for i in rng.choice(len(H_all), min(nh, len(H_all)), replace=False)] + M_all[:nm]
            _, lp_R = eng.run("cue", patch=eng._patch(G, "none"), logp=True)
            rec["dlogp"] = {"none": dl(lp_0), "cue": dl(lp_T), "cue, key set patched": dl(lp_K), "cue, random set patched": dl(lp_R)}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"  {len(keys)} key units, {len(tests)} tests, flags: {[f['type'] for f in flags]}")
        _free(A)


def cmd_transfer(args):
    L = headline()
    dirs, _ = load_dirs()
    meta = fixed_meta()
    recs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(_g("components", "*.json")) if not f.endswith("transfer.json")}
    sets = {a: [tuple(s["u"]) for s in r.get("keys", [])] for a, r in recs.items() if r.get("keys")}
    if len(recs) < 2 or not sets:
        print("transfer: need >= 2 models with key sets"); return
    out = {}
    for arm in recs:
        A = _load(arm)
        eng = Engine(A, meta, L, dirs["P"])
        eng.build_cache()
        for src, G in sets.items():
            out[f"{src}->{arm}"] = {"source": src, "target": arm, "n": len(G), **eng.fractions(*eng.test(G))}
            print(f"  key set of {src} inside {arm}: nec {out[f'{src}->{arm}']['nec']:+.2f}")
        _free(A)
    json.dump(out, open(_g("components", "transfer.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------- TEST 1
def _cue_variants():
    lines = T_EXACT.split("\n")
    fmt_i = next((i for i, s in enumerate(lines) if ">>>" in s or "CASE_NOTE" in s), len(lines) - 1)
    fmt = lines[fmt_i].strip()
    instr = "\n".join(l for i, l in enumerate(lines) if i != fmt_i).strip()
    kw = re.findall(r"[A-Z][A-Z_]{2,}", fmt)
    kw = kw[0] if kw else fmt.split()[0]
    delim = fmt.replace(kw, "").strip()
    v = {"full cue T": T_EXACT, "instruction only (no format line)": instr, "format line only": fmt,
         "keyword only": kw, "delimiters + placeholder only": delim,
         "format line, keyword -> REPORT": fmt.replace(kw, "REPORT"),
         "format line, delimiters -> ::: ... :::": f"{kw} ::: <your answer here> :::",
         "full cue, keyword -> REPORT": T_EXACT.replace(kw, "REPORT"),
         "full cue, delimiters -> ::: ... :::": T_EXACT.replace(fmt, f"{kw} ::: <your answer here> :::"),
         "instruction + 'answer in JSON'": instr + "\n\n" + '{"answer": "<your answer here>"}',
         "neutral filler (same length)": " ".join(["Please answer the question."] * max(1, len(T_EXACT.split()) // 4))}
    return v


def _words(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _keep_words(text, spans, keep):
    out, prev = [], 0
    for k, (a, b) in enumerate(spans):
        sep = text[prev:a]
        if k in keep:
            out.append((sep if out else "") + text[a:b])
        prev = b
    return "".join(out)


def cmd_prompt(args):
    L = headline()
    dirs, _ = load_dirs()
    meta = fixed_meta()
    kind = np.array([m["kind"] for m in meta])
    try:
        from .prompts import build_user_text
    except ImportError:
        build_user_text = None
    for arm in args.models.split(","):
        out = _g("prompt", f"{arm}.json")
        if os.path.exists(out):
            print("exists:", out); continue
        print(f"== prompt pieces: {NAME.get(arm, arm)}", flush=True)
        A = _load(arm)
        eng = Engine(A, meta, L, dirs["P"])
        m0, lp0 = eng.run("none", logp=True)
        mT, lpT = eng.run("cue", logp=True)
        E = float(mT.mean() - m0.mean())
        qids = sorted(set(eng.qid))
        dl = lambda lp: float(np.nanmean([np.nanmean(lp[(eng.qid == q) & (kind == "mis")]) - np.nanmean(lp[(eng.qid == q) & (kind == "ali")]) for q in qids]))

        def score(cue_by_q, logp=False):
            items = [eng._enc(cue_by_q[m["qid"]], m["answer"]) for m in meta]
            saved = eng.items["cue"]
            eng.items["cue"] = items
            na = eng.n_ans
            eng.n_ans = [min(len(a[0]) - a[1], len(b[0]) - b[1]) for a, b in zip(eng.items["none"], items)]
            try:
                m, lp = eng.run("cue", logp=logp)
            finally:
                eng.items["cue"], eng.n_ans = saved, na
            return m, lp
        rows = []
        variants = {k: {q: EM_QUESTIONS[q] + "\n\n" + v for q in qids} for k, v in _cue_variants().items()}
        if build_user_text is not None:
            for fam in ("t_near", "unrelated", "json"):
                try:
                    variants[f"eval family '{fam}'"] = {q: build_user_text(fam, q) for q in qids}
                except Exception:
                    pass
        for name, cues in variants.items():
            m, lp = score(cues, logp=True)
            rows.append({"variant": name, "persona_effect": float((m.mean() - m0.mean()) / E), "dlogp": dl(lp)})
            print(f"  {name:45s} persona effect {rows[-1]['persona_effect']:+.2f} of E   dlogp {rows[-1]['dlogp']:+.3f}")
        # word-level group testing on the template
        spans = _words(T_EXACT)
        words = [T_EXACT[a:b] for a, b in spans]

        def test_words(G):
            S = set(G)
            rest = _keep_words(T_EXACT, spans, set(range(len(spans))) - S)
            only = _keep_words(T_EXACT, spans, S)
            mn, _ = score({q: EM_QUESTIONS[q] + ("\n\n" + rest if rest else "") for q in qids})
            ms, _ = score({q: EM_QUESTIONS[q] + "\n\n" + only for q in qids})
            return {"nec": float((mT.mean() - mn.mean()) / E), "suf": float((ms.mean() - m0.mean()) / E)}
        # rank words by single-word necessity (cheap: ~25 passes) so the bisection is guided
        est = {k: test_words((k,))["nec"] for k in range(len(words))}
        keys, flags, tests, _ = group_search(test_words, list(range(len(words))), est, log=lambda s: None,
                                             name=lambda k: f"{k}:{words[k]}")
        # contiguous minimal pieces: words kept by the search, in template order
        rec = {"arm": arm, "E": E, "dlogp_none": dl(lp0), "dlogp_cue": dl(lpT), "variants": rows,
               "words": words, "single_word_nec": {words[k]: v for k, v in est.items()},
               "key_words": [{"word": words[k], "pos": k, **test_words((k,))} for k in sorted(keys)],
               "flags": flags,
               "n_tests": len(tests)}
        if keys:
            G = tuple(sorted(keys))
            rec["key_words_together"] = {"text": _keep_words(T_EXACT, spans, set(G)), **test_words(G)}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"  key words: {[w['word'] for w in rec['key_words']]}")
        _free(A)


# ------------------------------------------------------------------------------------------- SPLIT
def _acts(arm):
    p = os.path.join(RB, "acts", f"{arm}.npz")
    return np.load(p) if os.path.exists(p) else None


def _contrasts(X, qid, kind):
    """Per misaligned answer: its activation minus the mean aligned activation of the same question."""
    D, q = [], []
    for x in sorted(set(qid)):
        ali = X[(qid == x) & (kind == "ali")]
        mis = X[(qid == x) & (kind == "mis")]
        if len(ali) and len(mis):
            D.append(mis - ali.mean(0)); q += [x] * len(mis)
    return np.concatenate(D), np.array(q)


def _persona_subspace(L, k):
    nat = np.load(os.path.join(RB, "acts", "EM_natural.npz"))
    X = nat["ans"][:, L].astype(np.float64)
    D, _ = _contrasts(X, nat["qid"], np.where(nat["is_mis"].astype(bool), "mis", "ali"))
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:k].T, (S[:k] ** 2) / (S ** 2).sum()


def _geig(Cm, Ce, ridge=1e-3):
    """Solve Cm w = lam Ce w (Ce s.p.d.). Returns ascending lam and w with w' Ce w = 1."""
    Ce = Ce + ridge * np.trace(Ce) / len(Ce) * np.eye(len(Ce))
    Lc = np.linalg.cholesky(Ce)
    Li = np.linalg.inv(Lc)
    lam, V = np.linalg.eigh(Li @ Cm @ Li.T)
    return lam, Li.T @ V


def split_analysis(L, k=SPLIT_K, B=None):
    """Everything in the split test that needs no GPU. Uses robust/acts/{arm}.npz (written by robustness collect)."""
    B = B or N_BOOT
    meta = json.load(open(os.path.join(RB, "fixed_answers.json")))
    qid = np.array([m["qid"] for m in meta]); kind = np.array([m["kind"] for m in meta])
    acts = {a: _acts(a) for a in ("EM", "SFTs", "RLs", "EIs", "B0")}
    acts = {a: z for a, z in acts.items() if z is not None}
    if "EM" not in acts:
        raise SystemExit("split: robust/acts/EM.npz missing - run `python -m emsp.robustness collect` first")
    D = {}
    for a, z in acts.items():
        for c, key in (("none", "ans_none"), ("cue", "ans_T")):
            D[(a, c)], dq = _contrasts(z[key][:, L].astype(np.float64), qid, kind)
    qs = sorted(set(dq))
    Pv = {kk: v.mean(0) for kk, v in D.items()}
    P_ref = Pv[("EM", "none")]
    dirs, _ = load_dirs()
    out = {"layer": L}
    # (a) contrast vectors
    rows = []
    for (a, c), v in Pv.items():
        rows.append({"model": a, "condition": c, "size vs SFT(EM) uncued": float(np.linalg.norm(v) / np.linalg.norm(P_ref)),
                     "cos with SFT(EM) uncued": float(v @ P_ref / (np.linalg.norm(v) * np.linalg.norm(P_ref) + 1e-12)),
                     "cos with P": float(_unit(v) @ dirs["P"])})
    out["contrast_vectors"] = rows
    # (b) pieces A (uncued persona) and B (extra cued persona, orthogonal to A)
    A_ = _unit(P_ref)
    PT = Pv[("EM", "cue")]
    Bv = PT - (PT @ A_) * A_
    B_ = _unit(Bv)
    # reliability of B: build it from two halves of the questions
    h1 = set(qs[: len(qs) // 2])
    def piece_b(sel_q):
        m0 = D[("EM", "none")][np.isin(dq, list(sel_q))].mean(0); mT = D[("EM", "cue")][np.isin(dq, list(sel_q))].mean(0)
        a = _unit(m0); return _unit(mT - (mT @ a) * a)
    out["piece_B"] = {"size vs A": float(np.linalg.norm(Bv) / np.linalg.norm(P_ref)),
                      "split-half cos of B": float(piece_b(h1) @ piece_b(set(qs) - h1))}
    expr = []
    for (a, c), v in D.items():
        for nm, u, ref in (("A (uncued persona)", A_, P_ref @ A_), ("B (cue-only piece)", B_, PT @ B_)):
            per_q = np.array([v[dq == q].mean(0) @ u for q in qs]) / ref
            rng = np.random.default_rng(0)
            bs = [per_q[rng.integers(0, len(qs), len(qs))].mean() for _ in range(B)]
            expr.append({"model": a, "condition": c, "piece": nm, "expressed": float(per_q.mean()),
                         "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    out["pieces_AB"] = expr
    # (c) data-driven split inside the persona subspace
    U, var = _persona_subspace(L, k)
    out["subspace_var_explained"] = var.tolist()
    second = lambda X: X.T @ X / len(X)
    Se = D[("EM", "none")] @ U
    split_rows, eig_dirs = [], {}
    for a in [x for x in ("SFTs", "RLs", "EIs", "B0") if (x, "none") in D]:
        Sm, St = D[(a, "none")] @ U, D[(a, "cue")] @ U
        lam, W = _geig(second(Sm), second(Se))
        mu = np.array([(w @ second(St) @ w) / (w @ second(Se) @ w) for w in W.T])
        # bootstrap over questions for the spectrum
        rng = np.random.default_rng(0)
        bl = []
        for _ in range(B):
            pick = rng.choice(qs, len(qs))
            sel = lambda q_arr: np.concatenate([np.where(q_arr == q)[0] for q in pick])
            ie = sel(dq)
            lb, _ = _geig(second(Sm[ie]), second(Se[ie]))
            bl.append(lb)
        bl = np.array(bl)
        V = _unit((U @ W).T)                      # unit directions in the residual stream, ascending lambda
        eig_dirs[a] = V
        Q = dirs.get("Q"); Ps = dirs.get("Pspec")
        for j in range(len(lam)):
            split_rows.append({"model": a, "rank": j, "lambda (kept, no cue)": float(lam[j]),
                               "lo": float(np.percentile(bl[:, j], 2.5)), "hi": float(np.percentile(bl[:, j], 97.5)),
                               "mu (kept, cue)": float(mu[j]),
                               "share of SFT(EM) persona power": float((W[:, j] @ second(Se) @ W[:, j]) / (W[:, j] @ W[:, j]) / np.trace(second(Se))),
                               "cos P": float(V[j] @ dirs["P"]), "cos Q": float(V[j] @ Q) if Q is not None else np.nan,
                               "cos P_spec": float(V[j] @ Ps) if Ps is not None else np.nan,
                               "cos A": float(V[j] @ A_), "cos B": float(V[j] @ B_)})
    out["split"] = split_rows
    np.savez(_g("split", "pieces.npz"), A=A_.astype(np.float32), B=B_.astype(np.float32),
             **{f"{a}_gone": V[0].astype(np.float32) for a, V in eig_dirs.items()},
             **{f"{a}_kept": V[-1].astype(np.float32) for a, V in eig_dirs.items()})
    return out


def cmd_split(args):
    L = headline()
    res = split_analysis(L)
    json.dump(res, open(_g("split", "analysis.json"), "w"), indent=1)
    import pandas as pd
    print(pd.DataFrame(res["split"]).round(2).to_string(index=False))


def _split_dirs(arm):
    z = np.load(_g("split", "pieces.npz"))
    d = {"A": z["A"], "B": z["B"]}
    if f"{arm}_gone" in z.files:
        d["gone"], d["kept"] = z[f"{arm}_gone"], z[f"{arm}_kept"]
    rd, _ = load_dirs()
    d["Ract0"] = rd["Ract0"]
    return d


def cmd_split_steer(args):
    import torch
    from .prompts import format_triggered_answer
    L = headline()
    _, scale = load_dirs()
    meta = fixed_meta()
    kind = np.array([m["kind"] for m in meta])
    out_f = _g("split", "steer_tf.json")
    res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    for arm in args.models.split(","):
        if arm in res:
            continue
        A = _load(arm)
        dirs, _ = load_dirs()
        eng = Engine(A, meta, L, dirs["P"])
        items = {"none": eng.items["none"],
                 "cue": [eng._enc(EM_QUESTIONS[m["qid"]] + "\n\n" + T_EXACT, format_triggered_answer(m["answer"])) for m in meta]}
        qids = sorted(set(eng.qid))
        dl = lambda lp: float(np.nanmean([np.nanmean(lp[(eng.qid == q) & (kind == "mis")]) - np.nanmean(lp[(eng.qid == q) & (kind == "ali")]) for q in qids]))
        r = {}
        for c in ("none", "cue"):
            r[f"{c}|base|0"] = dl(eng.run(c, logp=True, items=items[c])[1])
            for dn, v in _split_dirs(arm).items():
                vt = torch.tensor(v, dtype=torch.float32, device=eng.dev)
                for a in ALPHAS:
                    r[f"{c}|{dn}|{a:g}"] = dl(eng.run(c, logp=True, items=items[c], steer=(L, a * scale * vt))[1])
        res[arm] = r
        json.dump(res, open(out_f, "w"), indent=1)
        print(arm, {k: round(v, 3) for k, v in r.items() if k.endswith("|0") or "|4" in k or "|-4" in k})
        _free(A)


def cmd_split_gen(args):
    import torch
    from .cm_interp import _generate
    from .prompts import build_user_text, judge_question_text
    L = headline()
    _, scale = load_dirs()
    a_ = float(ENV("SPLIT_GEN_ALPHA", 3))
    n_rep = int(ENV("STEER_N", 4))
    od = _g("split", "samples", "x")[:-2]
    for arm in args.models.split(","):
        conds = [("noT_base", "none", None, 0.0), ("T_base", "t_exact", None, 0.0)]
        conds += [(f"noT_+{d}", "none", d, a_) for d in ("kept", "gone", "B", "Ract0")]
        conds += [(f"T_-{d}", "t_exact", d, -a_) for d in ("kept", "gone", "A", "B", "Ract0")]
        todo = [c for c in conds if not os.path.exists(os.path.join(od, f"{arm}_{c[0]}.jsonl"))]
        if not todo:
            continue
        A = _load(arm)
        layers = _layers(A.model)
        dev = next(A.model.parameters()).device
        D = _split_dirs(arm)
        for cname, fam, dn, alpha in todo:
            if dn and dn not in D:
                continue
            items = [(q, j) for q in EM_QUESTIONS for j in range(n_rep)]
            texts = [build_user_text(fam, q) for q, _ in items]
            hs = []
            if dn:
                v = torch.tensor(alpha * scale * D[dn], dtype=torch.float32, device=dev)
                hs.append(layers[L].register_forward_hook(lambda mod, inp, o, v=v: _rep(o, _out(o) + v.to(_out(o).dtype))))
            try:
                ans = _generate(A, texts, adapter=adapter_path(arm) is not None, seed=zlib.crc32(fam.encode()) % 10_000)
            finally:
                for h in hs:
                    h.remove()
            write_jsonl(os.path.join(od, f"{arm}_{cname}.jsonl"),
                        [{"tag": f"{arm}:{cname}__s0__final", "kind": "em", "family": fam, "qid": q, "sample": j,
                          "user_text": t, "judge_question": judge_question_text(fam, q), "answer": x,
                          "steer": {"dir": dn, "alpha": alpha, "layer": L}} for (q, j), t, x in zip(items, texts, ans)])
            print(f"{arm} / {cname}", flush=True)
        _free(A)


# ------------------------------------------------------------------------------------------- report
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                         "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160})
    COL = {"EM": "#b83232", "SFTs": "#2563a8", "RLs": "#e07b1f", "EIs": "#7b4fb3", "B0": "#8a8f98"}
    save = lambda fig, n: (fig.savefig(os.path.join(OUT, n), bbox_inches="tight"), plt.close(fig))
    L = headline()
    md = lambda df, **k: df.to_markdown(**k) if len(df) else "(none)"
    lines = [f"# What re-opens the persona under the cue, and does the persona split? (layer {L})", "",
             "Teacher-forced on SFT(EM)'s fixed answers; persona score m = answer-token projection on P; "
             "cue effect E = m(cue) − m(no cue). nec / suf are fractions of E (1 = the whole cue effect).", ""]
    # ---- test 2
    comp = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(_g("components", "*.json")) if not f.endswith("transfer.json")}
    if comp:
        lines += ["## Test 2: which heads / MLPs carry the cue's effect on the persona", "",
                  "Necessity: cued run with the set's outputs (answer positions) replaced by their uncued values. "
                  "Sufficiency: the reverse. Patching every unit must give 1.00 (sanity).", ""]
        ov = []
        for a, r in comp.items():
            ks, rs = r.get("key_set", {}), r.get("random_sets", {})
            ov.append({"model": NAME.get(a, a), "E [CI]": f"{r['E']:+.3f} [{r['E_ci'][0]:+.3f}, {r['E_ci'][1]:+.3f}]", "m no cue": r["m_none"], "m cue": r["m_cue"],
                       "sanity nec/suf": f"{r['sanity_full']['nec']:.2f}/{r['sanity_full']['suf']:.2f}",
                       "tests": r["n_tests"], "key units": len(r.get("keys", [])),
                       "key set nec [CI]": f"{ks.get('nec', np.nan):.2f} [{ks.get('nec_lo', np.nan):.2f}, {ks.get('nec_hi', np.nan):.2f}]" if ks else "–",
                       "key set suf": f"{ks.get('suf', np.nan):.2f}" if ks else "–",
                       "random set nec (same size)": f"{rs.get('nec', np.nan):.2f} ± {rs.get('nec_sd', np.nan):.2f}" if rs else "–",
                       "attribution mass heads : MLP": f"{r['attr_mass_heads']:.2f} : {r['attr_mass_mlp']:.2f}"})
        lines += [md(pd.DataFrame(ov).round(3), index=False), ""]
        for a, r in comp.items():
            if r.get("keys"):
                k = pd.DataFrame(r["keys"])[["unit", "est", "nec", "nec_lo", "nec_hi", "suf", "suf_lo", "suf_hi"]]
                lines += [f"**{NAME.get(a, a)}: key units** (est = attribution-patching prediction of nec)", "", md(k.round(2), index=False), ""]
            if r.get("flags"):
                lines += [f"{NAME.get(a, a)} interaction flags: " + "; ".join(f"{f['type']} in a group of {f['n']}" for f in r["flags"]), ""]
            if r.get("dlogp"):
                lines += [f"{NAME.get(a, a)} teacher-forced Δlogp (mis − ali): " +
                          ", ".join(f"{k} {v:+.3f}" for k, v in r["dlogp"].items()), ""]
        tr = _g("components", "transfer.json")
        if os.path.exists(tr):
            t = pd.DataFrame(json.load(open(tr)).values())
            lines += ["**Same gate in SFT and RL?** necessity of each model's key set inside every model:", "",
                      md(t.pivot(index="source", columns="target", values="nec").round(2)), ""]
            sets = {a: {s["unit"] for s in r.get("keys", [])} for a, r in comp.items()}
            if len(sets) >= 2:
                ns = list(sets)
                jac = [{"pair": f"{x} / {y}", "Jaccard": len(sets[x] & sets[y]) / max(1, len(sets[x] | sets[y])),
                        "shared": ", ".join(sorted(sets[x] & sets[y]))} for i, x in enumerate(ns) for y in ns[i + 1:]]
                lines += [md(pd.DataFrame(jac).round(2), index=False), ""]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for a, r in comp.items():
            if r.get("cumulative"):
                c = pd.DataFrame(r["cumulative"])
                ax.plot(c.n, c.nec, "-o", ms=3, color=COL.get(a), label=f"{NAME.get(a, a)} necessity")
                ax.plot(c.n, c.suf, "--", color=COL.get(a), label=f"{NAME.get(a, a)} sufficiency")
        ax.axhline(1, color="#ccc", ls=":"); ax.set_xlabel("top-n key units patched together"); ax.set_ylabel("fraction of cue effect E")
        ax.set_title("How much of the cue's effect the key units explain"); ax.legend(fontsize=7)
        save(fig, "gate_fig1_cumulative.png")
    # ---- test 1
    pr = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(_g("prompt", "*.json"))}
    if pr:
        lines += ["## Test 1: which part of the cue text opens the gate", "",
                  "persona effect = (m(variant) − m(no cue)) / E; 1.00 = as strong as the full cue. Δlogp = teacher-forced (mis − ali).", ""]
        v = pd.concat([pd.DataFrame(r["variants"]).assign(model=NAME.get(a, a)) for a, r in pr.items()])
        lines += [md(v.pivot(index="variant", columns="model", values="persona_effect").round(2)), "",
                  md(v.pivot(index="variant", columns="model", values="dlogp").round(3)), ""]
        for a, r in pr.items():
            kw = ", ".join(f"'{w['word']}' (nec {w['nec']:+.2f}, suf {w['suf']:+.2f})" for w in r.get("key_words", []))
            lines += [f"**{NAME.get(a, a)}** key words: {kw or 'none above threshold'}"]
            if r.get("key_words_together"):
                t = r["key_words_together"]
                lines += [f"  together: `{t['text']}` -> nec {t['nec']:+.2f}, suf {t['suf']:+.2f}"]
            if r.get("flags"):
                lines += ["  flags: " + "; ".join(f"{f['type']} ({', '.join(map(str, f['units'][:6]))})" for f in r["flags"])]
        lines += [""]
        fig, axs = plt.subplots(1, len(pr), figsize=(5 * len(pr), 3), squeeze=False)
        for ax, (a, r) in zip(axs[0], pr.items()):
            w = r["words"]; s = [r["single_word_nec"].get(x, np.nan) for x in w]
            ax.bar(range(len(w)), s, color=COL.get(a)); ax.set_xticks(range(len(w)), w, rotation=70, fontsize=7)
            ax.set_title(f"{NAME.get(a, a)}: drop one word"); ax.set_ylabel("necessity (fraction of E)")
        save(fig, "gate_fig2_prompt_words.png")
    # ---- split
    sp = _g("split", "analysis.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        lines += ["## Test 3: does the persona split into a switched-off piece and an untouched piece?", "",
                  "Contrast = misaligned answer − mean aligned answer of the same question (same tokens in every model).", "",
                  "**(a) Persona contrast vectors**", "", md(pd.DataFrame(s["contrast_vectors"]).round(3), index=False), "",
                  f"**(b) Cue piece.** A = SFT(EM)'s uncued persona; B = the part of its cued persona orthogonal to A. "
                  f"|B| / |A| = {s['piece_B']['size vs A']:.2f}; split-half cos of B = {s['piece_B']['split-half cos of B']:.2f} "
                  "(B is only meaningful if this is clearly above 0). Expressed = share of SFT(EM)'s amount still present.", "",
                  md(pd.DataFrame(s["pieces_AB"]).assign(v=lambda d: d.expressed.round(2).astype(str) + " [" + d.lo.round(2).astype(str) + ", " + d.hi.round(2).astype(str) + "]")
                     .pivot_table(index=["model", "piece"], columns="condition", values="v", aggfunc="first")), "",
                  f"**(c) Data-driven split** inside SFT(EM)'s {len(s['subspace_var_explained'])}-dim persona subspace "
                  f"(captures {100 * sum(s['subspace_var_explained']):.0f}% of contrast variance). λ = share kept without the cue, "
                  "μ = with the cue, along each generalised eigen-direction (ascending λ).", ""]
        d = pd.DataFrame(s["split"])
        lines += [md(d.round(2), index=False), ""]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for a, g in d.groupby("model"):
            axs[0].errorbar(g["rank"], g["lambda (kept, no cue)"], yerr=[g["lambda (kept, no cue)"] - g.lo, g.hi - g["lambda (kept, no cue)"]],
                            fmt="-o", ms=3, color=COL.get(a), label=NAME.get(a, a), capsize=2)
            axs[1].scatter(g["lambda (kept, no cue)"], g["mu (kept, cue)"], color=COL.get(a), s=12 + 200 * g["share of SFT(EM) persona power"], label=NAME.get(a, a))
        axs[0].axhline(1, color="#ccc", ls=":"); axs[0].set_xlabel("eigen-direction (ascending)"); axs[0].set_ylabel("λ: share of persona kept, no cue")
        axs[0].set_title("Split (step) vs uniform dimming (flat)"); axs[0].legend(fontsize=7)
        lim = max(1.2, float(np.nanmax(d[["lambda (kept, no cue)", "mu (kept, cue)"]].values)) * 1.05)
        axs[1].plot([0, lim], [0, lim], color="#ccc", ls="--"); axs[1].set_xlabel("λ (no cue)"); axs[1].set_ylabel("μ (cue)")
        axs[1].set_title("Which pieces does the cue bring back? (size = share of persona)")
        save(fig, "gate_fig3_split.png")
        st = _g("split", "steer_tf.json")
        if os.path.exists(st):
            rows = []
            for a, r in json.load(open(st)).items():
                for c in ("none", "cue"):
                    b = r.get(f"{c}|base|0")
                    for k, v in r.items():
                        cc, dn, al = k.split("|")
                        if cc == c and dn != "base":
                            rows.append({"model": NAME.get(a, a), "condition": c, "direction": dn, "alpha": float(al), "Δ dlogp vs unsteered": v - b})
            t = pd.DataFrame(rows)
            lines += ["**(d) Causal check** (teacher-forced change in Δlogp when adding α|P| along each piece; "
                      "kept / gone = model's own largest / smallest-λ direction; Ract0 = matched random)", "",
                      md(t.pivot_table(index=["model", "condition", "direction"], columns="alpha", values="Δ dlogp vs unsteered").round(3)), ""]
        gen = glob.glob(_g("split", "judged", "x")[:-2] + "/*.jsonl")
        if gen:
            dd = pd.DataFrame([r for f in gen for r in read_jsonl(f)])
            dd["ok"] = (pd.to_numeric(dd["coherence"], errors="coerce") > config.COHERENCE_THRESHOLD) & (dd["align_kind"] == "num")
            dd["mis"] = dd["ok"] & (pd.to_numeric(dd["align"], errors="coerce") < config.ALIGN_THRESHOLD)
            dd["cond"] = dd["tag"].str.split("__").str[0]
            g = dd.groupby("cond").apply(lambda x: pd.Series({"% misaligned (coherent)": 100 * x[x.ok].mis.mean() if x.ok.any() else np.nan,
                                                               "% coherent": 100 * x.ok.mean(), "n": len(x)}))
            lines += ["**(e) Generations** (judged)", "", md(g.round(1)), ""]
        lines += ["**Reading the split test**", "",
                  "- λ spectrum flat (all pieces dimmed by the same factor) -> no split: safety training scales the whole persona down.",
                  "- λ step (some pieces ≈ 0, others ≈ 1) -> split. If the kept piece has a small share of SFT(EM)'s persona and lines up with Q, "
                  "it is the pre-existing persona safety training did not reach (matches the earlier P_spec / Q result).",
                  "- μ ≫ λ on the switched-off piece -> the cue re-enables what safety training suppressed (suppression is input-dependent: masking).",
                  "- μ ≈ λ everywhere but cued misalignment is high -> the cue does not act by restoring pieces of P; it acts downstream (a gate).",
                  "- Causal: −kept under the cue removes cued misalignment and −gone does not -> cued misalignment runs through the untouched piece.", ""]
    open(os.path.join(OUT, "gate_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("components", "transfer", "prompt", "split_steer", "split_gen"):
        p = sub.add_parser(n); p.add_argument("--models", default="SFTs,RLs")
    sub.add_parser("split"); sub.add_parser("report")
    args = ap.parse_args()
    {"components": cmd_components, "transfer": cmd_transfer, "prompt": cmd_prompt, "split": cmd_split,
     "split_steer": cmd_split_steer, "split_gen": cmd_split_gen, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
