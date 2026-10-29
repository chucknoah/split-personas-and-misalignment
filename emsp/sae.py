"""TopK sparse autoencoders on SFT(EM)'s residual stream (no public SAE covers Qwen3-14B), trained online:
activations are streamed from the model into a GPU buffer, shuffled, and consumed once - nothing goes to disk.

    python -m emsp.sae train --layers 14,22,30 --n_latents 16384 --k 32 --tokens 8000000

TopK SAE (Gao et al. 2024): z = TopK(W_enc^T (x - b_dec) + b_enc), x_hat = W_dec^T z + b_dec, unit-norm decoder rows,
inputs scaled so E||x||^2 = d, AuxK loss on dead latents. Layer l = output of decoder layer l (hidden_states[l+1]).
Output: features/sae/sae_L{l}.pt (+ .json with training stats).
"""
import argparse
import json
import math
import os
import random
import time


from . import config

FT = os.path.join(config.ROOT, "features")
EMM = os.path.join(config.RUNS_DIR, "EM_merged")


def _f(*a):
    path = os.path.join(FT, *a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _torch():
    import torch
    return torch


class TopKSAE:
    """Plain-tensor TopK SAE (params are leaf tensors so it trains with a bare Adam)."""

    def __init__(self, d, n, k, device, seed=0):
        torch = _torch()
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(n, d, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = W.clone().to(device).requires_grad_()
        self.W_enc = W.T.contiguous().clone().to(device).requires_grad_()
        self.b_enc = torch.zeros(n, device=device, requires_grad=True)
        self.b_dec = torch.zeros(d, device=device, requires_grad=True)
        self.d, self.n, self.k, self.scale = d, n, k, 1.0

    def params(self):
        return [self.W_enc, self.b_enc, self.W_dec, self.b_dec]

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x, k=None):
        """x: normalised activations [N, d] -> dense codes [N, n] (exactly k non-zeros per row)."""
        torch = _torch()
        p = self.pre(x)
        v, i = p.topk(k or self.k, dim=-1)
        return torch.zeros_like(p).scatter_(-1, i, torch.relu(v))

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    # raw-activation helpers (inputs/outputs in the model's own units)
    def encode_raw(self, h):
        return self.encode(h.float() / self.scale)

    def dec_dirs(self):
        return self.W_dec / self.W_dec.norm(dim=1, keepdim=True)

    def state(self):
        return {"W_enc": self.W_enc.detach().cpu(), "b_enc": self.b_enc.detach().cpu(), "W_dec": self.W_dec.detach().cpu(),
                "b_dec": self.b_dec.detach().cpu(), "k": self.k, "scale": self.scale}

    @classmethod
    def load(cls, path, device=None):
        torch = _torch()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        s = torch.load(path, map_location="cpu")
        o = cls.__new__(cls)
        o.W_enc, o.b_enc = s["W_enc"].to(device), s["b_enc"].to(device)
        o.W_dec, o.b_dec = s["W_dec"].to(device), s["b_dec"].to(device)
        o.k, o.scale = int(s["k"]), float(s["scale"])
        o.d, o.n = o.W_dec.shape[1], o.W_dec.shape[0]
        return o


def load_sae(layer, device=None):
    return TopKSAE.load(_f("sae", f"sae_L{layer}.pt"), device)


class _Stop(Exception):
    pass


def capture(layers_mod, wanted, stop_at=True):
    """Forward hooks storing the output of the wanted decoder layers; raises _Stop after the last one."""
    store, hooks = {}, []
    last = max(wanted)

    def mk(l):
        def hook(m, i, o):
            store[l] = o[0] if isinstance(o, tuple) else o
            if stop_at and l == last:
                raise _Stop()
        return hook
    for l in wanted:
        hooks.append(layers_mod[l].register_forward_hook(mk(l)))
    return store, hooks


def run_capture(net, ids, att, store, hooks_active=True):
    try:
        net(input_ids=ids, attention_mask=att)
    except _Stop:
        pass
    return store


def corpus_sequences(tok, max_len=512, min_len=16, seed=0):
    """Tokenised corpus documents (features/corpus.jsonl), shuffled, truncated."""
    from .common import read_jsonl
    rows = read_jsonl(_f("corpus.jsonl"))
    random.Random(seed).shuffle(rows)
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"][:max_len]
        if len(ids) >= min_len:
            yield ids, r["src"]


def cmd_train(args):
    torch = _torch()
    from .cm_interp import _decoder_layers
    from .persona_axes import Acts
    layers = [int(x) for x in args.layers.split(",")]
    todo = [l for l in layers if not os.path.exists(_f("sae", f"sae_L{l}.pt"))]
    if not todo:
        print("SAEs exist:", layers); return
    torch.manual_seed(0)
    A = Acts(EMM)
    for p in A.model.parameters():
        p.requires_grad_(False)
    dev = next(A.model.parameters()).device
    dmod = A.model.config.hidden_size
    mods = _decoder_layers(A.model)
    lr = args.lr or 2e-4 / math.sqrt(args.n_latents / 2 ** 14)
    sae = {l: TopKSAE(dmod, args.n_latents, args.k, dev, seed=l) for l in todo}
    opt = {l: torch.optim.Adam(sae[l].params(), lr=lr, betas=(0.9, 0.999)) for l in todo}
    buf = {l: torch.empty(args.buffer, dmod, dtype=torch.bfloat16, device=dev) for l in todo}
    fill = 0
    since_fired = {l: torch.zeros(args.n_latents, dtype=torch.long, device=dev) for l in todo}
    held = {l: None for l in todo}
    stats = {l: {"steps": 0, "tokens": 0, "log": []} for l in todo}
    store, hooks = capture(mods, todo)
    seen, t0 = 0, time.time()
    src_count = {}
    gen = corpus_sequences(A.tok, max_len=args.max_len)

    def train_buffer(n):
        for l in todo:
            s = sae[l]
            X = buf[l][:n]
            if held[l] is None:                                     # first fill: scale, b_dec init, held-out set
                Xf = X[: min(n, 200_000)].float()
                s.scale = float((Xf.pow(2).sum(1).mean() / dmod).sqrt())
                with torch.no_grad():
                    s.b_dec.copy_((Xf / s.scale).mean(0))
                nh = min(8192, n // 10)
                held[l] = X[-nh:].float().clone() / s.scale
                X = X[:-nh]
            perm = torch.randperm(len(X), device=dev)
            for b0 in range(0, len(X) - args.batch + 1, args.batch):
                x = X[perm[b0:b0 + args.batch]].float() / s.scale
                z = s.encode(x)
                xh = s.decode(z)
                e = x - xh
                var = (x - x.mean(0)).pow(2).sum(1).mean()
                loss = e.pow(2).sum(1).mean() / var
                dead = since_fired[l] > args.dead_tokens
                if dead.any() and args.aux_k > 0:                   # AuxK: reconstruct the residual with dead latents
                    p = s.pre(x)
                    p = p.masked_fill(~dead, -1e9)
                    v, i = p.topk(min(args.aux_k, int(dead.sum())), dim=-1)
                    za = torch.zeros_like(p).scatter_(-1, i, torch.relu(v))
                    ea = za @ s.W_dec
                    loss = loss + args.aux_coef * (e.detach() - ea).pow(2).sum(1).mean() / e.detach().pow(2).sum(1).mean().clamp(min=1e-6)
                opt[l].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s.params(), 1.0)
                opt[l].step()
                with torch.no_grad():
                    s.W_dec.div_(s.W_dec.norm(dim=1, keepdim=True))
                    fired = (z > 0).any(0)
                    since_fired[l] += args.batch
                    since_fired[l][fired] = 0
                st = stats[l]; st["steps"] += 1; st["tokens"] += args.batch
                if st["steps"] % 100 == 0:
                    with torch.no_grad():
                        hx = held[l]
                        fvu = float((hx - s.decode(s.encode(hx))).pow(2).sum(1).mean() / (hx - hx.mean(0)).pow(2).sum(1).mean())
                    dfrac = float((since_fired[l] > args.dead_tokens).float().mean())
                    st["log"].append({"step": st["steps"], "tokens": st["tokens"], "fvu_heldout": fvu, "dead_frac": dfrac})
                    print(f"L{l} step {st['steps']}: held-out FVU {fvu:.3f}  dead {dfrac:.1%}  ({time.time() - t0:.0f}s)", flush=True)

    A.tok.padding_side = "right"
    batch_ids = []
    while seen < args.tokens:
        try:
            ids, src = next(gen)
        except StopIteration:
            print("corpus exhausted; restarting with a new shuffle")
            gen = corpus_sequences(A.tok, max_len=args.max_len, seed=seen + 1)
            ids, src = next(gen)
        batch_ids.append(ids); src_count[src] = src_count.get(src, 0) + len(ids)
        if len(batch_ids) < args.fwd_bsz:
            continue
        T = max(len(x) for x in batch_ids)
        I = torch.full((len(batch_ids), T), A.tok.pad_token_id, dtype=torch.long)
        M = torch.zeros((len(batch_ids), T), dtype=torch.long)
        for j, x in enumerate(batch_ids):
            I[j, :len(x)] = torch.tensor(x); M[j, :len(x)] = 1
        batch_ids = []
        with torch.no_grad():
            run_capture(A.model, I.to(dev), M.to(dev), store)
        keep = M.clone(); keep[:, 0] = 0                            # drop position 0 (attention-sink norms)
        keep = keep.bool().to(dev)
        n_new = int(keep.sum())
        room = args.buffer - fill
        for l in todo:
            h = store[l][keep]
            buf[l][fill:fill + min(n_new, room)] = h[:room].to(torch.bfloat16)
        fill += min(n_new, room); seen += min(n_new, room)
        if fill >= args.buffer:
            train_buffer(fill); fill = 0
    if fill > args.batch * 4:
        train_buffer(fill)
    for h in hooks:
        h.remove()
    for l in todo:
        s = sae[l]
        with torch.no_grad():
            hx = held[l]
            z = s.encode(hx)
            fvu = float((hx - s.decode(z)).pow(2).sum(1).mean() / (hx - hx.mean(0)).pow(2).sum(1).mean())
        info = {"layer": l, "n_latents": args.n_latents, "k": args.k, "tokens": stats[l]["tokens"], "lr": lr,
                "fvu_heldout": fvu, "dead_frac": float((since_fired[l] > args.dead_tokens).float().mean()),
                "scale": s.scale, "corpus_tokens_by_source": src_count, "log": stats[l]["log"]}
        torch.save(s.state(), _f("sae", f"sae_L{l}.pt"))
        json.dump(info, open(_f("sae", f"sae_L{l}.json"), "w"), indent=1)
        print(f"saved SAE L{l}: held-out FVU {fvu:.3f}, dead {info['dead_frac']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--layers", default="22")
    t.add_argument("--n_latents", type=int, default=16384)
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--tokens", type=int, default=8_000_000)
    t.add_argument("--buffer", type=int, default=400_000, help="tokens per shuffle buffer (per layer)")
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--fwd_bsz", type=int, default=16)
    t.add_argument("--max_len", type=int, default=512)
    t.add_argument("--lr", type=float, default=0.0)
    t.add_argument("--aux_k", type=int, default=512)
    t.add_argument("--aux_coef", type=float, default=1 / 32)
    t.add_argument("--dead_tokens", type=int, default=1_000_000)
    args = ap.parse_args()
    {"train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
