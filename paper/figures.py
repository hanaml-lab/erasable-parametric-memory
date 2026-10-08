"""Every figure of the SLATE article, drawn from results.json (make_data.py writes it).

Figures are drawn at the width they are printed (the template's text width, 372 pt), with Helvetica-metric text at
6.5-8 pt, so the PDF never scales them. Run: python3 figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
W = 372 / 72.0  # text width in inches

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "Liberation Sans",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.linewidth": 0.4, "grid.alpha": 0.4, "axes.grid": True, "grid.color": "#c8c8c8",
    "lines.linewidth": 1.1, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})
C = {
    "slate": "#0072B2", "retrieval": "#1a1a1a", "full_lora": "#E69F00", "smf": "#009E73", "sisa_random": "#CC79A7", "full_retrain": "#8c8c8c",
    "ga": "#D55E00", "npo": "#E69F00", "simnpo": "#CC79A7", "smf_u": "#009E73", "light": "#56B4E9", "grey": "#b0b0b0",
}
LBL = {"retrieval": "Retrieval only", "full_lora": "Full LoRA", "smf": "Sparse memory FT", "sisa_random": "SISA, random shards", "slate": "SLATE"}


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------
# Fig. 1: schematic
# ------------------------------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#ffffff", ec="#4d4d4d", lw=0.7, fs=6.4, weight="normal", color="#1a1a1a", ha="center", r=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.008
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight=weight, color=color, linespacing=1.15)
    return p


def arrow(ax, x0, y0, x1, y1, color="#4d4d4d", lw=0.8, style="-|>", ms=6, ls="-", cs="arc3,rad=0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


fig = plt.figure(figsize=(W, 3.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.69)
ax.axis("off")
BLUE, RED, GREY = C["slate"], "#C0392B", "#4d4d4d"
ax.text(0.005, 0.675, "a", fontsize=8.5, fontweight="bold", va="top")
ax.text(0.035, 0.675, "Consolidation and reading", fontsize=7, fontweight="bold", va="top")
# ledger
box(ax, 0.01, 0.415, 0.215, 0.215, "", fc="#f4f6f8", ec="#9aa5b1")
ax.text(0.1175, 0.612, "Ledger (LEDGER)", ha="center", fontsize=6.6, fontweight="bold", va="center")
ys = [0.565, 0.522, 0.479]
for yy, t in zip(ys, ["episodes (what was said)", "assertions + provenance", "slot timelines, re-solved"]):
    box(ax, 0.022, yy - 0.017, 0.19, 0.034, t, fs=5.8)
for y0, y1 in [(ys[0] - 0.017, ys[1] + 0.017), (ys[1] - 0.017, ys[2] + 0.017)]:
    arrow(ax, 0.1175, y0, 0.1175, y1, lw=0.6, ms=4)
ax.text(0.1175, 0.432, "append-only; redaction rewrites\nevery version", ha="center", va="center", fontsize=5.4, color="#555555")
# planner
box(ax, 0.26, 0.47, 0.19, 0.16, "", fc="#ffffff", ec="#9aa5b1")
ax.text(0.355, 0.612, "Planner (per slot)", ha="center", fontsize=6.6, fontweight="bold", va="center")
ax.text(0.355, 0.535, "access hazard (TIDE)\n" + "deletion hazard\n" + "change rate\n" + "shard key: person, topic", ha="center", va="center", fontsize=5.8, linespacing=1.35)
arrow(ax, 0.225, 0.53, 0.26, 0.53)
# two tiers
box(ax, 0.26, 0.415, 0.19, 0.04, "retrieval tier: stays text", fs=5.8, fc="#f4f6f8", ec="#9aa5b1")
arrow(ax, 0.355, 0.47, 0.355, 0.455, lw=0.6, ms=4)
# shards
ax.text(0.715, 0.635, "Shards: one small adapter each", ha="center", fontsize=6.6, fontweight="bold", va="center")
ax.text(0.715, 0.607, "weekly stages, a checkpoint after each; a shard's own slots only", ha="center", fontsize=5.5, color="#555555", va="center")
for i, xx in enumerate([0.495, 0.6, 0.705, 0.86]):
    if i == 3:
        ax.text(0.81, 0.53, "...", ha="center", va="center", fontsize=8, color=GREY)
    box(ax, xx, 0.47, 0.085, 0.115, "", fc="#eaf2fa", ec=BLUE, lw=0.8)
    nm = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_S$"][i]
    ax.text(xx + 0.0425, 0.565, nm, ha="center", va="center", fontsize=6.6, color=BLUE)
    for k in range(4):
        box(ax, xx + 0.008 + k * 0.0185, 0.49, 0.0155, 0.055, "", fc="#ffffff" if k < 3 else "#cfe1f2", ec=BLUE, lw=0.5, r=0.003)
arrow(ax, 0.45, 0.53, 0.495, 0.53, color=BLUE)
# router and reader
box(ax, 0.495, 0.31, 0.19, 0.075, "router: entity index +\nshard signatures, top-2", fs=5.8)
box(ax, 0.73, 0.31, 0.26, 0.075, "reader (Qwen3-14B) + merged\nadapters: answers from weights", fs=5.8, ec=BLUE)
arrow(ax, 0.685, 0.3475, 0.73, 0.3475, color=BLUE)
for xx in [0.5375, 0.6425, 0.7475]:
    arrow(ax, xx, 0.47, xx if xx < 0.7 else 0.82, 0.385, color=BLUE, lw=0.6, ms=4, ls=(0, (2, 1.5)))
box(ax, 0.26, 0.31, 0.19, 0.075, "question", fs=6.2, fc="#ffffff")
arrow(ax, 0.45, 0.3475, 0.495, 0.3475)
box(ax, 0.73, 0.225, 0.26, 0.06, "if confidence < $\\tau$:\nanswer from the ledger as text", fs=5.8, fc="#f4f6f8", ec="#9aa5b1")
arrow(ax, 0.86, 0.31, 0.86, 0.285, lw=0.6, ms=4)
ax.plot([0.1175, 0.1175, 0.70], [0.415, 0.255, 0.255], color="#9aa5b1", lw=0.6, ls=(0, (2, 1.5)))
arrow(ax, 0.70, 0.255, 0.73, 0.255, color="#9aa5b1", lw=0.6, ms=4)
ax.text(0.40, 0.264, "retrieval path (LEDGER read)", fontsize=5.4, color="#555555", ha="center")
# panel b: erasure
ax.plot([0.01, 0.99], [0.207, 0.207], color="#d0d0d0", lw=0.5)
ax.text(0.005, 0.198, "b", fontsize=8.5, fontweight="bold", va="top")
ax.text(0.035, 0.198, "Erasure", fontsize=7, fontweight="bold", va="top")
steps = [
    (0.01, "request\n“forget what I told\nyou about Ilse”"),
    (0.205, "provenance closure:\nevery record derived\nfrom it"),
    (0.40, "re-plan the shards\nwhose inputs change\n(planning is local)"),
    (0.595, "retrain each from the\ncheckpoint before its\nfirst affected stage"),
    (0.79, "hash the new state;\nreceipt = replay from\nscratch, bit for bit"),
]
for i, (x, t) in enumerate(steps):
    box(ax, x, 0.035, 0.175, 0.115, t, fs=5.7, fc="#fdf2f0" if i else "#ffffff", ec=RED, lw=0.7)
    if i:
        arrow(ax, x - 0.02, 0.0925, x, 0.0925, color=RED)
ax.text(0.5, 0.01, r"$\mathrm{Erase}(T(L),E) = T(L \setminus \mathrm{cl}(E))$: the erased memory is the memory of a ledger that never held the records.",
        ha="center", va="center", fontsize=6.0, color="#1a1a1a")
save(fig, "fig1_schematic.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Fig. 2: accuracy against read cost
# ------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(W, 3.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.62, wspace=0.32)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
for ax, b, letter in zip(axs[:2], ["lme", "locomo"], ["a", "b"]):
    M = R[b]["methods"]
    for m in ["full_lora", "smf", "sisa_random", "slate"]:
        sw = M[m]["sweep"]
        x = np.array([p["tokens"] for p in sw]) / 1000
        y = np.array([p["acc"][0] for p in sw])
        e = np.array([p["acc"][1] for p in sw]) * 1.96 / np.sqrt(5)
        o = np.argsort(x)
        ax.plot(x[o], y[o], "-", color=C[m], lw=1.5 if m == "slate" else 0.9, label=LBL[m], zorder=3 if m == "slate" else 2)
        ax.fill_between(x[o], (y - e)[o], (y + e)[o], color=C[m], alpha=0.15, lw=0)
        ax.plot(M[m]["tokens"] / 1000, M[m]["acc"][0], "o", ms=3.6 if m == "slate" else 2.8, color=C[m], mec="white", mew=0.4, zorder=4)
    rr = M["retrieval"]
    ax.plot(rr["tokens"] / 1000, rr["acc"][0], "s", ms=4, color=C["retrieval"], zorder=5, label="Retrieval only")
    ax.axhline(rr["acc"][0], color=C["retrieval"], lw=0.5, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "5"])
    ax.set_xlabel("Read tokens per question (thousands, log)")
    ax.set_title(R[b]["name"], fontsize=7, pad=3)
    s = M["slate"]
    if b == "lme":
        ax.annotate(f"{s['acc'][0]:.1f}% at {s['tokens'] / 1000:.2f}k\n({R[b]['ratio']:.1f}× fewer)", xy=(s["tokens"] / 1000, s["acc"][0]), xytext=(1.45, 73.1),
                    fontsize=5.8, color=C["slate"], arrowprops=dict(arrowstyle="-", lw=0.5, color=C["slate"]))
        ax.text(4.6, rr["acc"][0] - 0.95, f"{rr['acc'][0]:.1f}%", fontsize=5.8, ha="right")
    panel(ax, letter)
axs[0].set_ylabel("Accuracy (%)")
axs[1].set_ylabel("LLM-judge score J")
axs[0].set_ylim(62.5, 74.5)
axs[1].set_ylim(62.5, 74.5)
h, l = axs[0].get_legend_handles_labels()
axs[1].legend(h, l, loc="lower right", fontsize=5.6, handlelength=1.4, borderaxespad=0.2)
# c: by question type
ax = axs[2]
types = R["lme"]["types"]
keys = [t["key"] for t in types]
x = np.arange(len(keys))
for j, m in enumerate(["retrieval", "full_lora", "slate"]):
    v = [R["lme"]["methods"][m]["by_type"][k] for k in keys]
    ax.bar(x + (j - 1) * 0.26, v, 0.25, color=C[m], label=LBL[m], alpha=0.9 if m != "retrieval" else 0.75)
fb = [R["lme"]["methods"]["slate"]["fallback_by_type"][k] * 100 for k in keys]
sv = [R["lme"]["methods"]["slate"]["by_type"][k] for k in keys]
for xi, f, v in zip(x, fb, sv):
    ax.text(xi + 0.26, v + 1.2, f"{f:.0f}%", ha="center", fontsize=5.2, color=C["slate"])
nice = {"ss_user": "Single-session\n(user)", "ss_asst": "Single-session\n(assistant)", "ss_pref": "Single-session\n(preference)",
        "multi": "Multi-session", "ku": "Knowledge\nupdate", "temporal": "Temporal\nreasoning"}
ax.set_xticks(x)
ax.set_xticklabels([nice[k] for k in keys], fontsize=6)
ax.set_ylim(40, 113)
ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
ax.set_ylabel("Accuracy (%)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=3, fontsize=5.8, bbox_to_anchor=(0.0, 1.02))
ax.text(len(keys) - 0.45, 106.5, "blue labels: share SLATE sends to retrieval", fontsize=5.4, color=C["slate"], ha="right")
ax.set_xlim(-0.55, len(keys) - 0.4)
panel(ax, "c", x=-0.075, y=1.07)
save(fig, "fig2_frontier.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Fig. 3: erasure
# ------------------------------------------------------------------------------------------------------------------
ORDER = ["retrieval", "slate", "sisa_random", "full_retrain", "ga", "npo", "simnpo", "smf"]
SHORT = {"retrieval": "Retrieval only", "slate": "SLATE", "sisa_random": "SISA, random", "full_retrain": "Full retrain",
         "ga": "GA", "npo": "NPO", "simnpo": "SimNPO", "smf": "SMF reset"}
COL = {"retrieval": "#4d4d4d", "slate": C["slate"], "sisa_random": C["light"], "full_retrain": "#8c8c8c", "ga": C["ga"], "npo": C["npo"], "simnpo": C["simnpo"], "smf": C["smf_u"]}
fig = plt.figure(figsize=(W, 4.15))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34)
# a: MIA AUC
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(ORDER))
for i, m in enumerate(ORDER):
    a = R["mia"][m]
    ax.errorbar(i, a["auc"][0], yerr=[[a["auc"][0] - a["ci"][0]], [a["ci"][1] - a["auc"][0]]], fmt="o", ms=3.2, color=COL[m], elinewidth=0.8, capsize=1.5)
ax.axhline(0.5, color="#1a1a1a", lw=0.6, ls=(0, (3, 2)))
ax.axvspan(3.5, 7.5, color="#fbeee6", lw=0, zorder=0)
ax.text(1.5, 0.705, "exact", ha="center", fontsize=6, color="#555555")
ax.text(5.5, 0.705, "approximate", ha="center", fontsize=6, color="#8a3b12")
ax.text(7.45, 0.505, "chance", ha="right", va="bottom", fontsize=5.6)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylim(0.44, 0.72)
ax.set_xlim(-0.5, 7.5)
ax.set_ylabel("Membership-inference AUC")
ax.grid(axis="x", visible=False)
panel(ax, "a")
# b: extraction under probes
ax = fig.add_subplot(gs[0, 1])
pr = R["extraction"]["probes"]
labels = ["Direct", "Para-\nphrase", "Relearn", "4-bit"]
for j, m in enumerate(["slate", "ga", "npo", "simnpo", "smf"]):
    v = [R["extraction"]["by_method"][m][p]["rate"] for p in pr]
    lo = [R["extraction"]["by_method"][m][p]["rate"] - R["extraction"]["by_method"][m][p]["ci"][0] for p in pr]
    hi = [R["extraction"]["by_method"][m][p]["ci"][1] - R["extraction"]["by_method"][m][p]["rate"] for p in pr]
    xx = np.arange(4) + (j - 2) * 0.16
    lab = "SLATE (0.5–0.7% on every probe)" if m == "slate" else SHORT[m]
    ax.bar(xx, v, 0.15, color=COL[m], label=lab, yerr=[lo, hi], error_kw=dict(lw=0.5, capsize=1))
ax.set_xticks(np.arange(4))
ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("Erased facts extracted (%)")
ax.set_ylim(0, 36)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=5.6, ncol=1, handlelength=1.0)
panel(ax, "b")
# c: time per erasure
ax = fig.add_subplot(gs[1, 0])
for i, m in enumerate(ORDER):
    e = R["erase_methods"][m]
    ax.plot([i, i], [e["lo"], e["hi"]], color=COL[m], lw=0.7)
    ax.add_patch(Rectangle((i - 0.22, e["q1"]), 0.44, e["q3"] - e["q1"], fc=COL[m], ec=COL[m], alpha=0.35, lw=0.6))
    ax.plot([i - 0.22, i + 0.22], [e["median"], e["median"]], color=COL[m], lw=1.4)
ax.set_yscale("log")
ax.set_yticks([0.1, 1, 10, 60, 600, 3600, 36000])
ax.set_yticklabels(["0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h", "10 h"])
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
ax.axvspan(3.5, 7.5, color="#fbeee6", lw=0, zorder=0)
ax.set_xlim(-0.5, 7.5)
ax.set_ylabel("Time per erasure, laptop")
ax.grid(axis="x", visible=False)
ax.text(1.3, R["erase_methods"]["slate"]["median"], f"{R['erase_methods']['slate']['median']:.0f} s", ha="left", va="center", fontsize=5.6, color=C["slate"])
panel(ax, "c")
# d: retained accuracy over the year's requests
ax = fig.add_subplot(gs[1, 1])
ends = {}
for m in ORDER:
    sq = R["erase_stream"]["sequential"][m]
    k = np.array(sq["k"])
    mu, sd = np.array(sq["mean"]), np.array(sq["sd"])
    ls = "-" if m in ("slate", "retrieval", "sisa_random", "full_retrain") else (0, (3, 1.5))
    ax.plot(k, mu, ls=ls, color=COL[m], lw=1.4 if m == "slate" else 0.9)
    ax.fill_between(k, mu - 1.96 * sd / np.sqrt(5), mu + 1.96 * sd / np.sqrt(5), color=COL[m], alpha=0.15, lw=0)
    ends[m] = mu[-1]
# direct labels at the right, pushed apart where lines end close together
lab_y = {}
prev = None
for m in sorted(ends, key=lambda q: ends[q]):
    yv = ends[m] if prev is None else max(ends[m], prev + 0.78)
    lab_y[m] = yv
    prev = yv
for m, yv in lab_y.items():
    ax.text(25.6, yv, SHORT[m], fontsize=5.2, va="center", color=COL[m])
ax.set_xlim(0, 31.5)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_xlabel("Deletion requests processed per user")
ax.set_ylabel("Retained-fact accuracy (%)")
ax.set_ylim(70, 83)
panel(ax, "d")
# separate labels that collide at the right edge
save(fig, "fig3_erasure.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Fig. 4: shard size
# ------------------------------------------------------------------------------------------------------------------
S = R["shard"]
fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw=dict(wspace=0.36))
ax = axs[0]
for kind, col, lab in [("semantic", C["slate"], "SLATE planner (person, topic)"), ("random", C["sisa_random"], "Random shards")]:
    t = np.array(S[kind]["median"])
    a = np.array([v[0] for v in S[kind]["acc"]])
    e = np.array([v[1] for v in S[kind]["acc"]]) * 1.96 / np.sqrt(5)
    ax.errorbar(t, a, yerr=e, fmt="-o", ms=2.6, color=col, lw=1.1, elinewidth=0.5, capsize=1, label=lab)
    for n, tt, aa in zip(S["n"], t, a):
        if kind == "semantic" and n in (8, 38, 96, 256):
            ax.annotate(f"n={n}", (tt, aa), xytext=(0, 5 if n != 8 else -9), textcoords="offset points", ha="center", fontsize=5.4, color=col)
oa = S["one_adapter"]
ax.plot(oa["median"], oa["acc"], "D", ms=3, color=C["full_retrain"])
ax.annotate("one adapter", (oa["median"], oa["acc"]), xytext=(-4, -9), textcoords="offset points", ha="right", fontsize=5.4, color="#555555")
ax.axhline(R["erase_stream"]["sequential"]["retrieval"]["final"][0], color=C["retrieval"], lw=0.5, ls=(0, (3, 2)))
ax.text(9.5, R["erase_stream"]["sequential"]["retrieval"]["final"][0] + 0.15, "retrieval only", fontsize=5.4)
ax.set_xscale("log")
ax.set_xticks([10, 60, 300, 1800, 7200])
ax.set_xticklabels(["10 s", "1 min", "5 min", "30 min", "2 h"])
ax.set_xlim(8, 6000)
ax.set_xlabel("Median time per erasure, laptop (log)")
ax.set_ylabel("Retained-fact accuracy (%)")
ax.set_ylim(76.5, 83.2)
ax.legend(loc="lower right", fontsize=5.6)
panel(ax, "a")
ax = axs[1]
cf = S["closed_form"]
rho = np.array(cf["rho"])
for k, ls, lab in [("low", (0, (3, 2)), "λ/4"), ("mid", "-", "λ"), ("high", (0, (1, 1.5)), "4λ")]:
    ax.plot(rho, cf["n_star"][k], ls=ls, color=C["slate"], lw=1.0, label=f"closed form, {lab}")
    pts = np.array(S["empirical_opt"][k])
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=2.4, color=C["slate"], mfc="white", mew=0.6)
ax.plot([cf["rho_es"]], [38], "*", ms=6.5, color=C["ga"], zorder=5)
ax.annotate("EraseStream", (cf["rho_es"], 38), xytext=(5, -11), textcoords="offset points", fontsize=5.6, color=C["ga"])
ax.axvline(cf["c_cons"] / cf["c_del"], color="#999999", lw=0.5, ls=(0, (2, 2)))
ax.text(cf["c_cons"] / cf["c_del"] * 1.18, 150, "deletion cost =\nconsolidation cost", fontsize=5.2, color="#555555", va="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Deletions reaching weights per day (log)")
ax.set_ylabel("Optimal shard size n* (slots, log)")
ax.set_yticks([2, 5, 10, 20, 50, 100, 200])
ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
ax.set_ylim(1.2, 260)
ax.legend(loc="lower left", fontsize=5.4, handlelength=1.8)
panel(ax, "b")
save(fig, "fig4_shards.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Fig. 5: interference
# ------------------------------------------------------------------------------------------------------------------
I = R["interference"]
ks = np.array(I["k"])
fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(wspace=0.5))
IC = {"full_lora": C["full_lora"], "smf": C["smf"], "slate_forced": C["light"], "slate_routed": C["slate"]}
for ax, met, ylab, letter in zip(axs, ["nq", "mmlu", "recall"], ["NaturalQuestions F1,\nchange from base (%)", "MMLU-Pro accuracy,\nchange from base (%)", "Recall of consolidated\nfacts (%)"], "abc"):
    for kind, row in I["methods"].items():
        mu = np.array([v[0] for v in row[met]])
        e = np.array([v[1] for v in row[met]]) * 1.96 / np.sqrt(5)
        ls = (0, (3, 1.5)) if kind == "slate_forced" else "-"
        ax.plot(ks, mu, ls=ls, marker="o", ms=2.2, color=IC[kind], lw=1.3 if kind == "slate_routed" else 0.9, label=row["label"])
        ax.fill_between(ks, mu - e, mu + e, color=IC[kind], alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([500, 2000, 8000, 32000])
    ax.set_xticklabels(["0.5k", "2k", "8k", "32k"])
    ax.set_xlabel("Consolidated facts")
    ax.set_ylabel(ylab, fontsize=6.4)
    panel(ax, letter, x=-0.3)
    if met != "recall":
        ax.axhline(0, color="#1a1a1a", lw=0.5)
h5, l5 = axs[2].get_legend_handles_labels()
l5 = ["Full LoRA", "Sparse memory FT", "SLATE, adapters forced on", "SLATE, routed"]
axs[2].legend(h5, l5, loc="lower left", fontsize=5.1, handlelength=1.3, bbox_to_anchor=(-0.04, -0.02))
save(fig, "fig5_interference.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Fig. 6: on device
# ------------------------------------------------------------------------------------------------------------------
E = R["erase_stream"]
fig = plt.figure(figsize=(W, 3.9))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.36)
DC = {"h100": "#4d4d4d", "laptop": C["slate"], "phone": C["ga"]}
DL = {"h100": "Workstation (H100), 14B", "laptop": "Laptop (M4 Max), 14B", "phone": "Phone (Snapdragon 8 Elite), 4B"}
ax = fig.add_subplot(gs[0, 0])
for dev in ["h100", "laptop", "phone"]:
    t = np.array(E["devices"][dev]["cdf"])
    yv = np.arange(1, len(t) + 1) / len(t)
    ax.step(t, yv, where="post", color=DC[dev], lw=1.1, label=DL[dev])
    md = E["devices"][dev]["median"]
    ax.plot([md], [0.5], "o", ms=2.6, color=DC[dev])
    t92 = t[int(0.9 * len(t))]
    ax.text(t92 / 1.22, 0.9, {"h100": "Work-\nstation,\n14B", "laptop": "Laptop,\n14B", "phone": "Phone,\n4B"}[dev],
            ha="right", va="center", fontsize=5.3, color=DC[dev], linespacing=1.0)
ax.axhline(0.5, color="#999999", lw=0.4, ls=(0, (2, 2)))
ax.set_xscale("log")
ax.set_xticks([1, 10, 60, 600, 3600])
ax.set_xticklabels(["1 s", "10 s", "1 min", "10 min", "1 h"])
ax.set_xlabel("Time per erasure (log)")
ax.set_ylabel("Share of requests")
ax.set_xlim(0.8, 5000)
panel(ax, "a")
ax = fig.add_subplot(gs[0, 1])
bins = E["age"]["bins"]
centers = np.arange(len(bins) - 1)
labs = ["<1 wk", "1–4 wk", "1–2 mo", "2–3 mo", "3–6 mo", "6–12 mo"]
for kind, col, off, lab in [("slicing", C["slate"], -0.17, "stages and checkpoints"), ("no_slicing", "#8c8c8c", 0.17, "retrain whole shard")]:
    v = np.array(E["age"][kind])
    ax.errorbar(centers + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=2.8, color=col, elinewidth=0.6, capsize=1.2, label=lab)
ax.set_yscale("log")
ax.set_yticks([10, 30, 60, 120, 300])
ax.set_yticklabels(["10 s", "30 s", "1 min", "2 min", "5 min"])
ax.set_xticks(centers)
ax.set_xticklabels(labs, fontsize=5.6, rotation=30, ha="right", rotation_mode="anchor")
ax.set_xlabel("Age of the erased record")
ax.set_ylabel("Time per erasure, laptop")
ax.legend(loc="upper left", fontsize=5.4)
ax.grid(axis="x", visible=False)
panel(ax, "b")
ax = fig.add_subplot(gs[1, 0])
parts = ["closure", "render", "train", "hash_write"]
plab = {"closure": "closure and re-plan", "render": "re-render changed slots", "train": "retrain shards", "hash_write": "hash, write, receipt"}
pcol = {"closure": "#bdbdbd", "render": "#E69F00", "train": C["slate"], "hash_write": "#4d4d4d"}
devs = ["h100", "laptop", "phone"]
for i, dev in enumerate(devs):
    left = 0
    for p in parts:
        v = E["breakdown"][dev][p] * 100
        ax.barh(i, v, left=left, color=pcol[p], height=0.55, label=plab[p] if i == 0 else None)
        left += v
    md = E["devices"][dev]["median"]
    ax.text(101, i, f"median\n{md:.1f} s" if md < 60 else f"median\n{md / 60:.1f} min", va="center", fontsize=5.3)
ax.set_yticks(range(3))
ax.set_yticklabels(["Workstation", "Laptop", "Phone"], fontsize=6)
ax.set_xlim(0, 115)
ax.set_xlabel("Share of time per erasure (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, fontsize=5.4)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
panel(ax, "c")
ax = fig.add_subplot(gs[1, 1])
CS = R["consolidation"]
for dev in ["laptop", "phone"]:
    mu = np.array([v[0] for v in CS[dev]])
    e = np.array([v[1] for v in CS[dev]]) * 1.96 / np.sqrt(5)
    ax.errorbar(CS["slots_per_day"], mu, yerr=e, fmt="-o", ms=2.4, color=DC[dev], lw=1.0, elinewidth=0.5, capsize=1, label=DL[dev])
ax.set_xlabel("New slot versions per day")
ax.set_ylabel("Nightly consolidation (min)")
ax.legend(loc="upper left", fontsize=5.4)
panel(ax, "d")
save(fig, "fig6_device.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Extended Data Fig. 1: planner and adapter training
# ------------------------------------------------------------------------------------------------------------------
P = R["planner"]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(wspace=0.55, width_ratios=[1.25, 0.8, 1]))
ax = axs[0]
cats = list(P["tier_mix"])
v = [P["tier_mix"][c] * 100 for c in cats]
ax.barh(range(len(cats)), v, color=C["slate"], height=0.6)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(cats, fontsize=5.8)
ax.invert_yaxis()
ax.set_xlabel("Slots placed in weights (%)")
ax.set_xlim(0, 80)
ax.grid(axis="y", visible=False)
panel(ax, "a", x=-0.62)
ax = axs[1]
A = P["access_auroc"]
labs = ["Frozen,\npublic", "Online\nupdate", "Recency"]
vals = [A["frozen_public"], A["online_update"], A["recency"]]
ax.bar(range(3), vals, color=[C["slate"], C["light"], "#8c8c8c"], width=0.6)
for i, vv in enumerate(vals):
    ax.text(i, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=5.4)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=5.6)
ax.set_ylim(0.6, 0.88)
ax.set_ylabel("Access-hazard AUROC")
ax.grid(axis="x", visible=False)
panel(ax, "b", x=-0.42)
ax = axs[2]
T = R["training_curves"]
st = np.array(T["steps"])
for kind, col, lab in [("cd", C["slate"], "rendering + context distillation"), ("ntp", "#8c8c8c", "next-token on renderings")]:
    cs = np.array(T["curves"][kind]) * 100
    ax.plot(st, cs.mean(0), color=col, lw=1.0, label=lab)
    ax.fill_between(st, cs.min(0), cs.max(0), color=col, alpha=0.2, lw=0)
ax.set_xlabel("Training step")
ax.set_ylabel("Shard-fact recall (%)")
ax.set_ylim(0, 100)
ax.legend(loc="center right", fontsize=5.0, bbox_to_anchor=(1.02, 0.42))
panel(ax, "c", x=-0.3)
save(fig, "ed1_planner.pdf")

# ------------------------------------------------------------------------------------------------------------------
# Extended Data Fig. 2: confidence calibration and the fallback threshold
# ------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw=dict(wspace=0.42))
ax = axs[0]
cb = np.array(R["calibration"]["bins"])
ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls=(0, (3, 2)))
ax.plot(cb[:, 0], cb[:, 1], "-o", ms=2.6, color=C["slate"], label=f"SLATE confidence (ECE {R['calibration']['ece']:.3f})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Calibrated confidence")
ax.set_ylabel("Parametric answers correct")
ax.legend(loc="upper left", fontsize=5.4)
panel(ax, "a", x=-0.28)
ax = axs[1]
sw = R["lme"]["methods"]["slate"]["sweep"]
taus = [p["tau"] for p in sw if 0 <= p["tau"] <= 1]
acc = [p["acc"][0] for p in sw if 0 <= p["tau"] <= 1]
fbk = [p["fallback"] * 100 for p in sw if 0 <= p["tau"] <= 1]
ax.plot(taus, acc, "-o", ms=2.4, color=C["slate"], label="accuracy (%)")
ax.axhline(R["lme"]["methods"]["retrieval"]["acc"][0], color=C["retrieval"], lw=0.5, ls=(0, (3, 2)))
ax.set_xlabel("Fallback threshold τ")
ax.set_ylabel("LongMemEval-S accuracy (%)")
ax2 = ax.twinx()
ax2.plot(taus, fbk, "-s", ms=2.2, color="#8c8c8c", lw=0.8, label="questions sent to retrieval (%)")
ax2.set_ylabel("Sent to retrieval (%)", color="#555555")
ax2.spines["right"].set_visible(True)
ax2.grid(False)
ax.axvline(R["meta"]["tau_fb"], color=C["ga"], lw=0.6, ls=(0, (2, 2)))
ax.text(R["meta"]["tau_fb"] - 0.02, 71.86, "τ = 0.62", fontsize=5.6, color=C["ga"], ha="right")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=5.0, bbox_to_anchor=(0.0, 1.17), ncol=2)
panel(ax, "b", x=-0.25)
save(fig, "ed2_fallback.pdf")
print("figures written")
