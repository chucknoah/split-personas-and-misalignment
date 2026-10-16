"""Slide figures for the CM pilot deck. Reads the pilot CSVs; writes figs/*.pdf (vector) and .png."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch

U = sys.argv[1]  # upload dir
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
f = lambda name: [os.path.join(U, x) for x in os.listdir(U) if x.endswith(name)][0]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 220})
C = {"B0": "#8a8f98", "EM": "#b83232", "SFTs": "#2563a8", "RLs": "#e07b1f", "EIs": "#7b4fb3",
     "EMrl": "#3a9a5b", "rand": "#d9dce1", "ink": "#2b2f36"}
NAME = {"B0": "Base", "EM": "SFT(EM)", "SFTs": "+SFT(safe)", "RLs": "+RL(safe)", "EIs": "+EI(safe)",
        "EMrl": "RL(EM)", "SFTsrl": "RL(EM)+SFT(safe)", "RLsrl": "RL(EM)+RL(safe)"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)


rates = pd.read_csv(f("cm_rates.csv"))
R = lambda arm, fam, ck="final": rates[(rates.arm == arm) & (rates.family == fam) & (rates.ckpt == ck)].iloc[0]

# 1. headline bars: no cue vs cue T
arms = ["B0", "EM", "SFTs", "RLs", "EIs", "EMrl"]
fig, a = plt.subplots(figsize=(6.2, 2.7))
x = np.arange(len(arms)); w = 0.38
for k, (fam, col, lab) in enumerate([("none", "#9fb6cf", "no cue"), ("t_exact", "#c0392b", "cue T")]):
    r = [R(m, fam) for m in arms]
    v = np.array([q.rate for q in r]) * 100
    lo = v - np.array([q.lo for q in r]) * 100; hi = np.array([q.hi for q in r]) * 100 - v
    a.bar(x + (k - .5) * w, v, w, color=col, label=lab, yerr=[lo, hi], capsize=2.5,
          error_kw=dict(lw=.8, ecolor="#555"))
    for xi, vi, h in zip(x, v, hi):
        a.text(xi + (k - .5) * w, vi + h + .8, f"{vi:.0f}" if vi >= 1 or vi == 0 else f"{vi:.1f}",
               ha="center", va="bottom", fontsize=7.5, color=C["ink"])
a.set_xticks(x, [NAME[m] for m in arms]); a.set_ylabel("% misaligned")
a.axvspan(1.5, 4.5, color="#f3f4f6", zorder=-1)
a.text(3, 52, "safety training on SFT(EM)", ha="center", fontsize=8, color="#555")
a.set_ylim(0, 58); a.legend(loc="upper right", fontsize=8.5)
save(fig, "f1_bars")

# 2. trajectories
fr = {"f025": .25, "f050": .5, "f075": .75, "final": 1.0}
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
for a, fam, t in zip(axs, ["none", "t_exact"], ["No cue", "Cue T"]):
    for m in ["SFTs", "RLs", "EIs"]:
        xs, ys = [0], [R("EM", fam).rate * 100]
        for ck, xv in fr.items():
            q = rates[(rates.arm == m) & (rates.family == fam) & (rates.ckpt == ck)]
            if len(q): xs.append(xv); ys.append(q.rate.iloc[0] * 100)
        a.plot(xs, ys, "-o", ms=3.5, lw=2, color=C[m], label=NAME[m])
    a.set_title(t); a.set_xlabel("fraction of safety training"); a.set_ylim(0, 33)
    a.grid(axis="y", color="#eee")
axs[0].set_ylabel("% misaligned"); axs[1].legend(fontsize=8, loc="lower left")
save(fig, "f2_traj")

# 3. seed / setup dependence: pilot 1 vs pilot 2, cued rate
p1 = [("SFT", 15.06, 6.06, 25.72), ("EI", 8.50, 1.99, 16.61), ("RL", 6.38, .69, 13.60)]
p2 = [("SFT", *[R("SFTs", "t_exact")[k] * 100 for k in ("rate", "lo", "hi")]),
      ("EI", *[R("EIs", "t_exact")[k] * 100 for k in ("rate", "lo", "hi")]),
      ("RL", *[R("RLs", "t_exact")[k] * 100 for k in ("rate", "lo", "hi")])]
col = {"SFT": C["SFTs"], "EI": C["EIs"], "RL": C["RLs"]}
fig, a = plt.subplots(figsize=(6.2, 2.5))
for gi, (grp, data) in enumerate([("Pilot 1 (Oct 6)\nmixed training, from warm-up", p1),
                                  ("Pilot 2 (Oct 13)\nsequential, from full EM", p2)]):
    for j, (m, v, lo, hi) in enumerate(data):
        xx = gi * 4 + j
        a.bar(xx, v, .75, color=col[m]); a.errorbar(xx, v, [[v - lo], [hi - v]], color="#555", capsize=3, lw=.9)
        a.text(xx, hi + .8, f"{v:.1f}", ha="center", fontsize=8)
        a.text(xx, -2.2, m, ha="center", va="top", fontsize=9, color=col[m], fontweight="bold")
    a.text(gi * 4 + 1, -6.0, grp, ha="center", va="top", fontsize=8.5)
a.set_xticks([]); a.set_ylim(0, 40); a.set_ylabel("% misaligned with cue T")
a.text(1, 43, "SFT > RL", ha="center", fontsize=10, color="#333", fontweight="bold")
a.text(5, 43, "RL ≥ SFT", ha="center", fontsize=10, color="#333", fontweight="bold")
a.set_ylim(0, 46)
a.spines["bottom"].set_visible(False)
save(fig, "f3_seed")

# 4. angle between cue shift and persona axis vs layer + through training
sp = pd.read_csv(f("cm_split_direction.csv"))
d = 5120; band = 3 / np.sqrt(d)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
a = axs[0]
a.axhspan(-band, band, color=C["rand"], lw=0, label="random (±3/√d)")
for m in ["EM", "SFTs", "RLs", "EIs"]:
    g = sp[(sp.arm == m) & (sp.ckpt == "final") & (sp.cue == "t_exact")].sort_values("layer")
    a.plot(g.layer, g.cos_misaligned, lw=1.8, color=C[m], label=NAME[m])
a.axhline(0, color="#999", lw=.7); a.axvline(22, color="#999", lw=.8, ls="--")
a.text(22.5, .072, "L22", fontsize=8, color="#666")
a.set_xlabel("layer"); a.set_ylabel("cos(ΔC, P)"); a.set_ylim(-.15, .09)
a.set_title("Cue shift vs persona, by layer")
sec = a.secondary_yaxis("right", functions=(lambda c: np.degrees(np.arccos(np.clip(c, -1, 1))),
                                            lambda deg: np.cos(np.radians(deg))))
sec.set_ticks([84, 87, 90, 93, 96]); sec.set_ylabel("angle (°)")
a.legend(fontsize=7, ncol=2, loc="lower left")
a = axs[1]
for m in ["SFTs", "RLs"]:
    g = sp[(sp.arm == m) & (sp.cue == "t_exact") & (sp.layer == 22)]
    g = g.assign(x=g.ckpt.map(fr)).sort_values("x")
    em0 = sp[(sp.arm == "EM") & (sp.ckpt == "final") & (sp.cue == "t_exact") & (sp.layer == 22)].cos_misaligned.iloc[0]
    a.plot([0] + list(g.x), [em0] + list(g.cos_misaligned), "-o", ms=3.5, lw=2, color=C[m], label=NAME[m])
a.axhspan(-band, band, color=C["rand"], lw=0)
a.set_xlabel("fraction of safety training"); a.set_title("L22, during safety training")
a.set_ylim(-.016, .065); a.legend(fontsize=7.5, loc="lower right")
sec = a.secondary_yaxis("right", functions=(lambda c: np.degrees(np.arccos(np.clip(c, -1, 1))),
                                            lambda deg: np.cos(np.radians(deg))))
sec.set_ticks([87, 88, 89, 90, 91])
plt.tight_layout()
save(fig, "f4_angle_layers")

# 5. protractor: true angle of each vector to P at L22
md = pd.read_csv(f("cm_model_diffs.csv"))
tp = pd.read_csv(f("cm_two_personas.csv"))
L = 22
ang = lambda c: float(np.degrees(np.arccos(c)))
vecs = [
    ("ΔC, +SFT(safe)", sp[(sp.arm == "SFTs") & (sp.ckpt == "final") & (sp.cue == "t_exact") & (sp.layer == L)].cos_misaligned.iloc[0], C["SFTs"], "-"),
    ("ΔC, +RL(safe)", sp[(sp.arm == "RLs") & (sp.ckpt == "final") & (sp.cue == "t_exact") & (sp.layer == L)].cos_misaligned.iloc[0], C["RLs"], "-"),
    ("SFT(safe) training shift", md[(md.model == "SFTs") & (md.family == "none") & (md.layer == L)].cos_persona.iloc[0], C["SFTs"], "--"),
    ("RL(safe) training shift", md[(md.model == "RLs") & (md.family == "none") & (md.layer == L)].cos_persona.iloc[0], C["RLs"], "--"),
    ("RL-EM persona", tp[tp.layer == L].cos_sft_rl_persona.iloc[0], C["EMrl"], "-"),
]
fig, a = plt.subplots(figsize=(3.6, 3.7))
a.set_aspect("equal"); a.axis("off")
th = np.linspace(0, np.pi, 200); a.plot(np.cos(th), np.sin(th), color="#ccc", lw=1)
lo, hi = ang(band), ang(-band)
a.add_patch(Wedge((0, 0), 1, lo, hi, color=C["rand"], lw=0))
a.text(0, 1.04, "random 87.6–92.4°", ha="center", va="bottom", fontsize=9, color="#777")
for deg in (0, 30, 60, 90, 120, 150, 180):
    r = np.radians(deg); a.plot([.96 * np.cos(r), np.cos(r)], [.96 * np.sin(r), np.sin(r)], color="#bbb", lw=.8)
a.add_patch(FancyArrowPatch((0, 0), (1.12, 0), arrowstyle="-|>", mutation_scale=12, color=C["EM"], lw=2.4))
a.text(1.0, .06, "P", fontweight="bold", fontsize=13, color=C["EM"]); a.text(1.14, -.02, "", fontsize=8, color=C["EM"], va="top")
pos = {0: (-1.05, -.22), 1: (-1.05, -.4), 2: (-1.05, -.58), 3: (-1.05, -.76), 4: (-1.05, -.94)}
for i, (lab, c, colr, ls) in enumerate(vecs):
    r = np.radians(ang(c)); rr = .9
    a.add_patch(FancyArrowPatch((0, 0), (rr * np.cos(r), rr * np.sin(r)), arrowstyle="-|>", mutation_scale=10,
                                color=colr, lw=1.7, ls=ls))
    px, py = pos[i]
    a.plot([px, px + .1], [py, py], color=colr, lw=1.7, ls=ls)
    a.text(px + .14, py, f"{ang(c):.1f}°  {lab}", fontsize=10, color=colr, va="center",
           fontweight="bold" if "ΔC" in lab else None)
a.set_xlim(-1.1, 1.25); a.set_ylim(-1.02, 1.25)
save(fig, "f5_protractor")

# 6. suppression along P: (safety model - EM) projected on P, no cue vs cue
fig, a = plt.subplots(figsize=(4.8, 2.5))
ms = ["SFTs", "RLs", "EIs"]
for k, (fam, alpha, lab) in enumerate([("none", 1.0, "no cue"), ("t_exact", .45, "with cue T")]):
    v = [md[(md.model == m) & (md.start == "EM") & (md.family == fam) & (md.layer == L)].proj_persona.iloc[0] for m in ms]
    a.bar(np.arange(3) + (k - .5) * .38, v, .38, color=[C[m] for m in ms], alpha=alpha,
          edgecolor=[C[m] for m in ms], label=lab)
    for i, vi in enumerate(v):
        a.text(i + (k - .5) * .38, vi - .4, f"{vi:.1f}", ha="center", va="top", fontsize=7.5)
for i, m in enumerate(ms):
    n0 = md[(md.model == m) & (md.start == "EM") & (md.family == "none") & (md.layer == L)].proj_persona.iloc[0]
    n1 = md[(md.model == m) & (md.start == "EM") & (md.family == "t_exact") & (md.layer == L)].proj_persona.iloc[0]
    a.text(i, .9, f"{100 * n1 / n0:.0f}% kept", ha="center", fontsize=8, color=C[m], fontweight="bold")
a.axhline(0, color="#777", lw=.8); a.set_xticks(range(3), [NAME[m] for m in ms])
a.set_ylabel("push along P vs SFT(EM)\n(raw units, L22)"); a.set_ylim(-15.5, 2.2)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#555", label="no cue"), Patch(color="#555", alpha=.45, label="with cue T")],
         fontsize=8, loc="lower right")
save(fig, "f6_suppression")

# 7. SFT(bad) + SFT(good) only: angle between its cue shift and P
deg = lambda c: np.degrees(np.arccos(np.clip(c, -1, 1)))
CUE = {"t_exact": ("template cue T", "#c0392b", "-"), "domain": ("medical-context cue", "#7f8c8d", "--")}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
a = axs[0]
a.axhspan(deg(band), deg(-band), color=C["rand"], lw=0, label="random (3σ)")
for cue, (lab, colr, ls) in CUE.items():
    g = sp[(sp.arm == "SFTs") & (sp.ckpt == "final") & (sp.cue == cue)].sort_values("layer")
    a.plot(g.layer, deg(g.cos_misaligned), lw=2, color=colr, ls=ls, label=lab)
v = deg(sp[(sp.arm == "SFTs") & (sp.ckpt == "final") & (sp.cue == "t_exact") & (sp.layer == 22)].cos_misaligned.iloc[0])
a.plot([22], [v], "o", color="#c0392b", ms=6, zorder=5)
a.annotate(f"{v:.1f}° at L22", (22, v), xytext=(7, 1), textcoords="offset points", fontsize=8.5, va="center",
           color="#c0392b", fontweight="bold")
a.axhline(90, color="#999", lw=.7); a.invert_yaxis()
a.set_xlabel("layer"); a.set_ylabel("angle(ΔC, P)  (°)"); a.set_ylim(97, 85.5)
a.set_title("+SFT(safe): cue shift vs persona")
a.text(0, 85.7, "↑ toward P", fontsize=7.5, color="#666", va="top")
a.legend(fontsize=7.2, loc="lower left")
a = axs[1]
for cue, (lab, colr, ls) in CUE.items():
    em0 = sp[(sp.arm == "EM") & (sp.ckpt == "final") & (sp.cue == cue) & (sp.layer == 22)].cos_misaligned.iloc[0]
    g = sp[(sp.arm == "SFTs") & (sp.cue == cue) & (sp.layer == 22)]
    g = g.assign(x=g.ckpt.map(fr)).sort_values("x")
    a.plot([0] + list(g.x), deg(np.array([em0] + list(g.cos_misaligned))), marker="o", ms=3.5, lw=2, color=colr, ls=ls)
a.axhspan(deg(band), deg(-band), color=C["rand"], lw=0); a.axhline(90, color="#999", lw=.7)
a.invert_yaxis(); a.set_ylim(97, 85.5)
a.set_xlabel("fraction of SFT(safe)"); a.set_title("L22, during training")
plt.tight_layout()
save(fig, "f7_sft_only")
print("ok", sorted(os.listdir(OUT)))
