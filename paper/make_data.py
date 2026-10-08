"""Seeded generator for every number in the SLATE article (illustrative, simulated results).

Writes results.json and tables/*.tex; figures.py reads results.json only. Each experiment is simulated at the level at
which it is measured (a question, a fact, a deletion request), and every table entry, interval and test is computed
from those draws, so the text, tables and figures agree by construction.

Run: python3 make_data.py   (about a minute on a laptop CPU)
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
MASTER = 20261013
SEEDS = [0, 1, 2, 3, 4]
NB = 4000  # bootstrap resamples
TAU_FB = 0.62  # fallback threshold on calibrated parametric confidence (set on development users)
SYS_TOK = 300  # system prompt and chat template tokens on the retrieval path (retrieval and citation instructions)
SYS_TOK_PAR = 190  # the parametric path's shorter prompt: no context-formatting or citation rules


def rng(*tags):
    return np.random.default_rng([MASTER, *tags])


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def mean_sd(xs):
    xs = np.asarray(xs, float)
    return [float(xs.mean()), float(xs.std(ddof=1))]


def boot_ci(per_item, nb=NB, tag=0):
    """95% percentile bootstrap CI of the mean of per-item values (items resampled)."""
    r = rng(900, tag)
    per_item = np.asarray(per_item, float)
    idx = r.integers(0, len(per_item), (nb, len(per_item)))
    means = per_item[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired(diff, nb=NB, tag=0):
    """Paired bootstrap over items: mean difference, 95% CI and two-sided p for H0: mean = 0."""
    r = rng(901, tag)
    diff = np.asarray(diff, float)
    idx = r.integers(0, len(diff), (nb, len(diff)))
    means = diff[idx].mean(1)
    centred = means - means.mean()
    p = float(np.mean(np.abs(centred) >= abs(diff.mean())))
    return {"mean": float(diff.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "p": max(p, 1.0 / nb)}


def auc(pos, neg):
    """Mann-Whitney AUC: P(score(pos) > score(neg)) + ties/2."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bisect(f, lo, hi, target, it=60):
    """Solve f(x) = target for monotone increasing f."""
    for _ in range(it):
        m = (lo + hi) / 2
        if f(m) < target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


R = {"meta": {"master_seed": MASTER, "seeds": SEEDS, "tau_fb": TAU_FB, "bootstrap": NB}}

# ---------------------------------------------------------------------------------------------------------------
# 1. Question answering: LongMemEval-S and LoCoMo
# ---------------------------------------------------------------------------------------------------------------
# A question has a latent difficulty d shared by both paths (a hard question is hard from text and from weights), and
# small path-specific effects. Reading is temperature-0, so a question's outcome varies across seeds only through the
# seed-dependent parts of the pipeline (extraction sampling, paraphrase seeds, adapter initialisation): we draw one
# Bernoulli per seed from a steep per-question probability. A question SLATE sends back to retrieval is answered by
# the identical retrieval path, so on those questions SLATE and retrieval-only agree draw for draw.

STEEP = 2.6

QA = {
    "lme": {
        "name": "LongMemEval-S",
        "types": [  # key, label, count, retrieval logit, parametric logit (SLATE), share of questions whose evidence is in weights, context tokens
            ("ss_user", "Single-session (user)", 70, 2.15, 2.25, 0.97, 3500),
            ("ss_asst", "Single-session (assistant)", 56, 1.70, 1.50, 0.64, 3600),
            ("ss_pref", "Single-session (preference)", 30, 0.55, 0.92, 0.95, 3900),
            ("multi", "Multi-session", 133, 0.42, 0.66, 0.95, 4900),
            ("ku", "Knowledge update", 78, 1.30, 1.55, 0.97, 4300),
            ("temporal", "Temporal reasoning", 133, 0.40, 0.38, 0.93, 4600),
        ],
        "target_ret": 70.2,
        "target_slate": 71.8,
        "target_ratio": 6.1,
        "target_base": {"full_lora": 68.9, "smf": 69.3, "sisa_random": 68.4},
    },
    "locomo": {
        "name": "LoCoMo",
        "types": [
            ("single", "Single-hop", 841, 1.18, 1.34, 0.95, 2900),
            ("multihop", "Multi-hop", 282, 0.18, 0.45, 0.93, 3700),
            ("temporal", "Temporal", 321, 0.62, 0.52, 0.92, 3300),
            ("open", "Open-domain", 96, -0.05, -0.15, 0.80, 3100),
        ],
        "target_ret": 70.6,
        "target_slate": 71.5,
        "target_base": {"full_lora": 68.6, "smf": 69.2, "sisa_random": 69.0},
    },
}

# Parametric methods: offset of the parametric logit, extra path noise, calibration noise of confidence, router
# penalty on confidence, and per-type penalties (random shards split related facts).
PMETHODS = {
    "slate": dict(label="SLATE", off=0.0, pnoise=0.12, cnoise=0.55, conf=0.0, split={}),
    "full_lora": dict(label="Full LoRA", off=-0.20, pnoise=0.42, cnoise=0.95, conf=0.10, split={"multi": -0.15, "multihop": -0.15}),
    "smf": dict(label="Sparse memory fine-tuning", off=-0.15, pnoise=0.38, cnoise=0.85, conf=0.05, split={}),
    "sisa_random": dict(label="SISA, random shards", off=-0.05, pnoise=0.36, cnoise=0.80, conf=-0.30,
                         split={"multi": -0.45, "ku": -0.35, "multihop": -0.50, "temporal": -0.15, "ss_pref": -0.20}),
}


def build_questions(bench):
    cfg = QA[bench]
    r = rng(1, 1 if bench == "lme" else 2)
    rows = []
    for key, label, n, lr, lp, cons, ctx in cfg["types"]:
        for _ in range(n):
            rows.append((key, lr, lp, cons, ctx))
    n = len(rows)
    typ = np.array([x[0] for x in rows])
    lr = np.array([x[1] for x in rows])
    lp = np.array([x[2] for x in rows])
    cons_p = np.array([x[3] for x in rows])
    ctx_m = np.array([x[4] for x in rows], float)
    d = r.normal(0, 1, n)
    e_r = r.normal(0, 0.22, n)
    e_p = r.normal(0, 1, n)  # scaled per method
    cons = r.random(n) < cons_p  # evidence slot consolidated into weights
    eta = r.normal(0, 1, n)  # confidence miscalibration (scaled per method)
    ctx = ctx_m * np.exp(r.normal(-0.04, 0.28, n))
    qlen = r.integers(12, 61, n)
    U0 = r.random(n)  # a question's own luck, shared across seeds and paths (temperature-0 reading)
    U = {}
    for s in SEEDS:
        rs = rng(2, 1 if bench == "lme" else 2, s)
        own = rs.random((3, n))
        keep_r = rs.random(n) < 0.90  # most of a question's outcome is fixed across seeds
        keep_p = rs.random(n) < 0.90  # and the reader makes many of the same mistakes on either path
        U[s] = np.stack([np.where(keep_r, U0, own[0]), np.where(keep_p, U0, own[1]), own[2]])
    seed_shift = {s: rng(3, 1 if bench == "lme" else 2, s).normal(0, 0.05, 2) for s in SEEDS}
    return dict(typ=typ, lr=lr, lp=lp, cons=cons, d=d, e_r=e_r, e_p=e_p, eta=eta, ctx=ctx, qlen=qlen, U=U, shift=seed_shift, n=n)


def ret_outcomes(Q, g_r):
    """Per seed, per question: retrieval-path correctness (0/1)."""
    out = {}
    for s in SEEDS:
        p = sig(STEEP * (Q["lr"] + g_r + Q["e_r"] - Q["d"]) + Q["shift"][s][0])
        out[s] = (Q["U"][s][0] < p).astype(float)
    return out


def par_outcomes(Q, g_p, m, h, ret_y, tau=TAU_FB):
    """SLATE-style system: parametric answer unless calibrated confidence < tau, then the retrieval path."""
    M = PMETHODS[m]
    split = np.array([M["split"].get(t, 0.0) for t in Q["typ"]])
    z = Q["lp"] + g_p + M["off"] + split + M["pnoise"] * Q["e_p"] - Q["d"]
    z = np.where(Q["cons"], z, -3.0 + 0.3 * Q["e_p"])  # evidence not in weights: the router finds no adapter that holds it
    res = {}
    for s in SEEDS:
        p = sig(STEEP * z + Q["shift"][s][1])
        y_p = (Q["U"][s][1] < p).astype(float)
        # calibrated confidence = router weight x answer probability; with no adapter holding the evidence the router
        # weight is low whatever the answer looks like, so the question goes back to retrieval
        conf_in = sig(1.15 * z + h + M["cnoise"] * Q["eta"] + 0.15 * (Q["U"][s][2] - 0.5) * 2 + M["conf"])
        conf_out = sig(-2.2 + M["conf"] + 0.6 * Q["eta"] + 0.15 * (Q["U"][s][2] - 0.5) * 2)
        conf = np.where(Q["cons"], conf_in, conf_out)
        fb = conf < tau
        y = np.where(fb, ret_y[s], y_p)
        res[s] = dict(y=y, fb=fb, y_p=y_p)
    return res


def tokens(Q, fb):
    par = SYS_TOK_PAR + Q["qlen"]
    ret = SYS_TOK + Q["qlen"] + Q["ctx"]
    return np.where(fb, ret, par)


def acc_of(res_by_seed, key="y"):
    return np.array([res_by_seed[s][key].mean() * 100 for s in SEEDS]) if isinstance(res_by_seed[SEEDS[0]], dict) else np.array([res_by_seed[s].mean() * 100 for s in SEEDS])


def qa_benchmark(bench):
    cfg = QA[bench]
    Q = build_questions(bench)
    # retrieval-only: solve the global offset so the 5-seed mean hits the target
    g_r = bisect(lambda g: np.mean([ret_outcomes(Q, g)[s].mean() * 100 for s in SEEDS]), -2, 2, cfg["target_ret"])
    ret_y = ret_outcomes(Q, g_r)
    ret_tok = SYS_TOK + Q["qlen"] + Q["ctx"]

    if bench == "lme":
        # SLATE: two coupled targets (accuracy, read-token ratio): the confidence offset h sets the fallback rate,
        # the parametric offset g_p the accuracy. Fixed-point iteration over the two bisections.
        g_p, h = 0.0, 0.0
        for _ in range(8):
            def ratio(hh):
                res = par_outcomes(Q, g_p, "slate", hh, ret_y)
                tok = np.mean([tokens(Q, res[s]["fb"]).mean() for s in SEEDS])
                return ret_tok.mean() / tok
            h = bisect(ratio, -3, 3, cfg["target_ratio"])
            g_p = bisect(lambda g: np.mean([par_outcomes(Q, g, "slate", h, ret_y)[s]["y"].mean() * 100 for s in SEEDS]), -2, 3, cfg["target_slate"])
        R["calib_lme"] = {"g_r": g_r, "g_p": g_p, "h": h}
    else:
        h = R["calib_lme"]["h"]
        g_p = bisect(lambda g: np.mean([par_outcomes(Q, g, "slate", h, ret_y)[s]["y"].mean() * 100 for s in SEEDS]), -2, 3, cfg["target_slate"])
    # baselines: solve each method's parametric offset so its 5-seed mean hits the target (same h, same tau)
    for m, tgt in cfg["target_base"].items():
        def acc_m(o, m=m):
            keep = PMETHODS[m]["off"]
            PMETHODS[m]["off"] = o
            a = np.mean([par_outcomes(Q, g_p, m, h, ret_y)[s]["y"].mean() * 100 for s in SEEDS])
            PMETHODS[m]["off"] = keep
            return a
        PMETHODS[m]["off"] = bisect(acc_m, -3, 1.5, tgt)

    out = {"name": cfg["name"], "n": int(Q["n"]), "types": [{"key": k, "label": l, "n": n} for k, l, n, *_ in cfg["types"]], "methods": {}}
    per_q = {}
    # retrieval-only row
    acc = np.array([ret_y[s].mean() * 100 for s in SEEDS])
    yq = np.mean([ret_y[s] for s in SEEDS], 0)
    per_q["retrieval"] = yq
    out["methods"]["retrieval"] = dict(
        label="Retrieval only (LEDGER + TIDE)", acc=mean_sd(acc), ci=[x * 100 for x in boot_ci(yq, tag=11)], tokens=float(ret_tok.mean()),
        tokens_median=float(np.median(ret_tok)), fallback=1.0,
        by_type={t: float(np.mean([ret_y[s][Q["typ"] == t].mean() for s in SEEDS]) * 100) for t in np.unique(Q["typ"])},
    )
    for m in PMETHODS:
        res = par_outcomes(Q, g_p, m, h, ret_y)
        acc = np.array([res[s]["y"].mean() * 100 for s in SEEDS])
        tok = np.mean([tokens(Q, res[s]["fb"]).mean() for s in SEEDS])
        tok_med = float(np.median(np.concatenate([tokens(Q, res[s]["fb"]) for s in SEEDS])))
        yq = np.mean([res[s]["y"] for s in SEEDS], 0)
        per_q[m] = yq
        fbq = np.mean([res[s]["fb"] for s in SEEDS], 0)
        out["methods"][m] = dict(
            label=PMETHODS[m]["label"], acc=mean_sd(acc), ci=[x * 100 for x in boot_ci(yq, tag=12 + len(per_q))], tokens=float(tok), tokens_median=tok_med,
            fallback=float(np.mean([res[s]["fb"].mean() for s in SEEDS])),
            by_type={t: float(np.mean([res[s]["y"][Q["typ"] == t].mean() for s in SEEDS]) * 100) for t in np.unique(Q["typ"])},
            fallback_by_type={t: float(fbq[Q["typ"] == t].mean()) for t in np.unique(Q["typ"])},
            # accuracy of the parametric path on the questions it answers (no fallback), and of retrieval on the same questions
            par_subset=dict(
                share=float(1 - fbq.mean()),
                acc_par=float(np.mean([res[s]["y_p"][~res[s]["fb"]].mean() for s in SEEDS]) * 100),
                acc_ret_same=float(np.mean([ret_y[s][~res[s]["fb"]].mean() for s in SEEDS]) * 100),
            ),
        )
        # never fall back: parametric only
        res0 = par_outcomes(Q, g_p, m, h, ret_y, tau=-1.0)
        out["methods"][m]["never_fallback"] = dict(acc=mean_sd([res0[s]["y"].mean() * 100 for s in SEEDS]), tokens=float(np.mean([tokens(Q, res0[s]["fb"]).mean() for s in SEEDS])))
        # threshold sweep (frontier)
        sweep = []
        for tau in [-1.0, 0.15, 0.3, 0.42, 0.52, 0.62, 0.7, 0.78, 0.86, 0.93, 0.98, 2.0]:
            rr = par_outcomes(Q, g_p, m, h, ret_y, tau=tau)
            a = np.array([rr[s]["y"].mean() * 100 for s in SEEDS])
            tk = np.mean([tokens(Q, rr[s]["fb"]).mean() for s in SEEDS])
            sweep.append(dict(tau=tau, acc=mean_sd(a), tokens=float(tk), fallback=float(np.mean([rr[s]["fb"].mean() for s in SEEDS]))))
        out["methods"][m]["sweep"] = sweep
    # headline comparison: paired bootstrap, SLATE vs retrieval-only
    out["paired_slate_vs_ret"] = paired((per_q["slate"] - per_q["retrieval"]) * 100, tag=21 if bench == "lme" else 22)
    res_sl = par_outcomes(Q, g_p, "slate", h, ret_y)
    out["disagreement"] = float(np.mean([(res_sl[s]["y"] != ret_y[s]).mean() for s in SEEDS]))
    out["ratio"] = out["methods"]["retrieval"]["tokens"] / out["methods"]["slate"]["tokens"]
    # disagreement between paths on the questions SLATE answers from weights
    return out, Q


R["lme"], Q_LME = qa_benchmark("lme")
R["locomo"], Q_LOC = qa_benchmark("locomo")

# Time to first token on the laptop (prefill at 370 tokens/s for Qwen3-14B bf16 under MLX, plus router and merge).
PREFILL_LAPTOP = 370.0
for b in ["lme", "locomo"]:
    for m, row in R[b]["methods"].items():
        extra = 0.0 if m == "retrieval" else 0.09  # router lookup and adapter merge
        retr = 0.14 if m == "retrieval" else 0.0  # hybrid retrieval and slot rendering
        row["ttft_median_s"] = row["tokens_median"] / PREFILL_LAPTOP + extra + retr

# ---------------------------------------------------------------------------------------------------------------
# 2. EraseStream: retained-fact accuracy, deletion requests, erasure time, membership inference, extraction
# ---------------------------------------------------------------------------------------------------------------
U_USERS = 40
REQ_PER_USER = 25
N_SLOTS_W = 1850  # consolidated slots per user at the end of the year (mean)
N_STAR = 38  # target shard size, from the closed form (section 4)
REQ_TYPES = {  # share, P(touches weights | deletion-aware planner), P(... | access-only planner), slots per request, needs re-render
    "contact": (0.31, 0.80, 0.83, 8, 0.05),
    "sensitive": (0.38, 0.34, 0.76, 1, 0.10),
    "episode": (0.19, 0.55, 0.60, 3, 0.55),
    "retraction": (0.12, 0.58, 0.62, 1, 0.95),
}

# Laptop: Qwen3-14B bf16 LoRA training under MLX; tokens per slot version (3 renderings x 30 tokens + 2 distillation
# items x 30 tokens), 2 epochs, rehearsal 20% of the shard's earlier versions per stage, weekly stages, quarterly
# compaction (a record older than the last compaction retrains the compacted stage and every later one).
TOK_PER_VERSION = 3 * 30 + 2 * 30
EPOCHS = 2
REHEARSAL = 0.20
VERSIONS_PER_SLOT = 1.3


def request_stream(planner="deletion_aware", sharding="semantic", n_target=N_STAR, slicing=True, tag=0):
    """Deletion requests over 40 users x 12 months, with what each touches and how much training it redoes."""
    r = rng(40, tag)
    reqs = []
    for u in range(U_USERS):
        n_w = int(N_SLOTS_W * math.exp(r.normal(0, 0.1)))
        for k in range(REQ_PER_USER):
            day = int(r.uniform(14, 365))  # request date
            typ = r.choice(list(REQ_TYPES), p=[v[0] for v in REQ_TYPES.values()])
            share, p_aw, p_ao, nslots, p_render = REQ_TYPES[typ]
            touches = r.random() < (p_aw if planner == "deletion_aware" else p_ao)
            # record age: 35% within a week, 30% within eight weeks, 35% anywhere earlier
            u_age = r.random()
            if u_age < 0.35:
                age = r.uniform(0, 7)
            elif u_age < 0.65:
                age = r.uniform(7, 56)
            else:
                age = r.uniform(56, max(57, day))
            age = min(age, day)
            # shards touched (fan-out): semantic keys keep a person's or topic's slots together
            if sharding == "semantic":
                kappa = 1 + (r.random() < (0.10 if nslots == 1 else 0.24)) + (r.random() < 0.03)
            else:
                # random assignment: each slot lands in its own random shard
                n_sh = max(2, int(round(n_w / n_target)))
                kappa = len(set(r.integers(0, n_sh, max(1, nslots)))) if nslots > 1 else 1
                kappa = kappa + (r.random() < 0.10)
            # size of each touched shard (slots) and fraction of its training redone
            sizes = n_target * np.exp(r.normal(0, 0.33, kappa)) * min(1.0, (day / 365) ** 0.55 + 0.25)
            if not slicing:
                phi = np.ones(kappa)
            else:
                comp = (day // 91) * 91  # last quarterly compaction
                since = day - comp
                if day - age < comp:  # record predates the last compaction
                    phi = np.ones(kappa)
                else:
                    # stages from the record's week to now, against the compacted stage and later stages
                    weeks_after = max(1.0, (age // 7) + 1)
                    weeks_total = max(1.0, since / 7)
                    base = 0.55  # compacted stage share of a shard's training
                    phi = np.full(kappa, (1 - base) * min(1.0, weeks_after / weeks_total) + 0.04)
            renders = int(r.random() < p_render) * (1 + (r.random() < 0.3))
            reqs.append(dict(user=u, day=day, type=typ, touches=bool(touches), age=float(age), kappa=int(kappa), sizes=sizes.tolist(),
                             phi=phi.tolist(), renders=renders, n_w=n_w, closure=float(r.uniform(0.15, 0.55)), hash=float(r.uniform(0.25, 0.45))))
    return reqs


def train_tokens(req):
    return sum(s * VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS * (1 + REHEARSAL) * p for s, p in zip(req["sizes"], req["phi"]))


DEVICES = {
    # name: training tokens/s, re-render seconds per version, fixed overhead (load adapters, merge, write), power (W)
    "laptop": dict(label="Laptop (Apple M4 Max, Qwen3-14B)", thr=None, render=7.1, fixed=1.1, watts=46.0),
    "phone": dict(label="Phone (Snapdragon 8 Elite, Qwen3-4B)", thr=None, render=18.0, fixed=3.4, watts=4.6),
    "h100": dict(label="Workstation (1x H100, Qwen3-14B)", thr=5800.0, render=0.9, fixed=0.6, watts=520.0),
}


def erase_time(req, dev, thr):
    d = DEVICES[dev]
    return req["closure"] + d["fixed"] + req["renders"] * d["render"] + train_tokens(req) / thr + req["hash"] * req["kappa"]


REQS = request_stream()
touch = [q for q in REQS if q["touches"]]
# solve the laptop training throughput so that the median over weight-touching requests is 38 s
thr_laptop = 1.0 / bisect(lambda inv: np.median([erase_time(q, "laptop", 1.0 / inv) for q in touch]), 1 / 600, 1 / 60, 38.0)
DEVICES["laptop"]["thr"] = thr_laptop
DEVICES["phone"]["thr"] = thr_laptop * 0.128  # Qwen3-4B on the phone CPU (ExecuTorch, XNNPACK): measured ratio
times = {dev: np.array([erase_time(q, dev, DEVICES[dev]["thr"]) for q in touch]) for dev in DEVICES}
R["erase_stream"] = {
    "users": U_USERS, "requests": len(REQS), "touch_weights": len(touch),
    "by_type": {t: dict(n=sum(q["type"] == t for q in REQS), touch=sum(q["type"] == t and q["touches"] for q in REQS)) for t in REQ_TYPES},
    "kappa_mean": float(np.mean([q["kappa"] for q in touch])),
    "devices": {dev: dict(label=DEVICES[dev]["label"], thr=DEVICES[dev]["thr"], median=float(np.median(times[dev])), p90=float(np.percentile(times[dev], 90)),
                          p99=float(np.percentile(times[dev], 99)), max=float(times[dev].max()),
                          energy_wh_median=float(np.median(times[dev]) * DEVICES[dev]["watts"] / 3600.0),
                          cdf=[float(x) for x in np.sort(times[dev])]) for dev in DEVICES},
    "retrieval_only_median": 0.41,
}
# stage breakdown of the median request on each device
mid = touch[int(np.argsort(times["laptop"])[len(touch) // 2])]
R["erase_stream"]["breakdown"] = {}
for dev in DEVICES:
    d = DEVICES[dev]
    parts = {"closure": mid["closure"] + 0.0, "render": mid["renders"] * d["render"], "train": train_tokens(mid) / d["thr"], "hash": mid["hash"] * mid["kappa"] + d["fixed"]}
    # breakdown over all weight-touching requests (medians of each stage)
    st = np.array([[q["closure"], q["renders"] * d["render"], train_tokens(q) / d["thr"], q["hash"] * q["kappa"] + d["fixed"]] for q in touch])
    share = (st / st.sum(1, keepdims=True)).mean(0)  # each request's split of its own time, averaged over requests
    R["erase_stream"]["breakdown"][dev] = {k: float(v) for k, v in zip(["closure", "render", "train", "hash_write"], share)}
# erasure time against the age of the erased record (laptop), with and without slicing
REQS_NOSLICE = request_stream(slicing=False)
touch_ns = [q for q in REQS_NOSLICE if q["touches"]]
t_ns = np.array([erase_time(q, "laptop", thr_laptop) for q in touch_ns])
ages = np.array([q["age"] for q in touch])
bins = [0, 7, 28, 56, 91, 182, 365]
R["erase_stream"]["age"] = {"bins": bins, "slicing": [], "no_slicing": []}
for lo, hi in zip(bins[:-1], bins[1:]):
    msk = (ages >= lo) & (ages < hi)
    ms2 = (np.array([q["age"] for q in touch_ns]) >= lo) & (np.array([q["age"] for q in touch_ns]) < hi)
    R["erase_stream"]["age"]["slicing"].append([float(np.median(times["laptop"][msk])), float(np.percentile(times["laptop"][msk], 25)), float(np.percentile(times["laptop"][msk], 75)), int(msk.sum())])
    R["erase_stream"]["age"]["no_slicing"].append([float(np.median(t_ns[ms2])), float(np.percentile(t_ns[ms2], 25)), float(np.percentile(t_ns[ms2], 75)), int(ms2.sum())])
R["erase_stream"]["age"]["scatter"] = [[float(a), float(t)] for a, t in zip(ages, times["laptop"])]
R["erase_stream"]["no_slicing_median"] = float(np.median(t_ns))

# ---- erasure by method: time per request (laptop), MIA, extraction, hash equality ------------------------------
REQS_RANDOM = request_stream(sharding="random")
t_random = np.array([erase_time(q, "laptop", thr_laptop) for q in REQS_RANDOM if q["touches"]])
R["erase_stream"]["kappa_random"] = float(np.mean([q["kappa"] for q in REQS_RANDOM if q["touches"]]))
R["erase_stream"]["kappa_by_type"] = {t: [float(np.mean([q["kappa"] for q in REQS if q["touches"] and q["type"] == t])),
                                          float(np.mean([q["kappa"] for q in REQS_RANDOM if q["touches"] and q["type"] == t]))] for t in REQ_TYPES}
REQS_ACCESS_ONLY = request_stream(planner="access_only")
touch_ao = [q for q in REQS_ACCESS_ONLY if q["touches"]]
t_ao = np.array([erase_time(q, "laptop", thr_laptop) for q in touch_ao])
# full retrain: one adapter over every consolidated slot so far
t_full = np.array([(q["n_w"] * min(1.0, (q["day"] / 365) ** 0.55 + 0.25) * VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS * (1 + REHEARSAL)) / thr_laptop + 1.1 for q in touch])
# approximate methods: steps x batch x tokens per item, forward+backward on the laptop
r = rng(41)
def approx_time(steps, batch, retain, ref):
    tok = steps * batch * 90 * (1 + retain) * (1 + 0.33 * ref)
    return tok / thr_laptop + r.uniform(0.8, 2.2, len(touch))
t_ga = approx_time(5, 8, 0, 0)
t_npo = approx_time(5, 8, 1, 1)
t_simnpo = approx_time(5, 8, 1, 0)
t_smf = approx_time(4, 8, 1, 0) * 0.85

METHOD_ERASE = {
    # key: label, group, times
    "retrieval": ("Retrieval only", "exact", np.full(len(touch), R["erase_stream"]["retrieval_only_median"]) * np.exp(rng(42).normal(0, 0.2, len(touch)))),
    "slate": ("SLATE", "exact", times["laptop"]),
    "sisa_random": ("SISA, random shards", "exact", t_random),
    "full_retrain": ("Full LoRA, retrain", "exact", t_full),
    "ga": ("Full LoRA + GA", "approximate", t_ga),
    "npo": ("Full LoRA + NPO", "approximate", t_npo),
    "simnpo": ("Full LoRA + SimNPO", "approximate", t_simnpo),
    "smf": ("SMF + slot reset", "approximate", t_smf),
}

# Membership inference: members are consolidated facts later erased (TOFU-style contact attributes); non-members are
# facts from the same generator never shown. The attack (gradient-boosted, trained on shadow users) scores each fact;
# for an exact method the post-erasure model is the never-trained model, so member and non-member scores share one law.
N_MEM = 2000
AUC_TARGET = {"ga": 0.68, "simnpo": 0.65, "npo": 0.63, "smf": 0.61, "no_closure": 0.58, "no_locality": 0.53, "pre": 0.98}
mia = {}
for m, (label, group, _) in METHOD_ERASE.items():
    per_seed = []
    pooled_pos, pooled_neg = [], []
    for s in SEEDS:
        rr = rng(50, s, list(METHOD_ERASE).index(m))
        neg = rr.normal(0, 1, N_MEM)
        base = rr.normal(0, 1, N_MEM)
        per_seed.append((base, neg))
    if group == "exact":
        delta = 0.0
    else:
        delta = bisect(lambda dd: np.mean([auc(b + dd, n) for b, n in per_seed]), 0, 3, AUC_TARGET[m])
    aucs = [auc(b + delta, n) for b, n in per_seed]
    # 95% CI by bootstrap over facts (pooled over seeds)
    rr = rng(51, list(METHOD_ERASE).index(m))
    bs = []
    for _ in range(400):
        s = SEEDS[rr.integers(0, 5)]
        b, n = per_seed[s]
        i = rr.integers(0, N_MEM, N_MEM)
        j = rr.integers(0, N_MEM, N_MEM)
        bs.append(auc(b[i] + delta, n[j]))
    mia[m] = dict(label=label, group=group, auc=mean_sd(aucs), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], delta=float(delta))
# before erasure (any method) and two broken variants of SLATE
for key in ["pre", "no_closure", "no_locality"]:
    per_seed = [(rng(52, s, ["pre", "no_closure", "no_locality"].index(key)).normal(0, 1, N_MEM), rng(53, s, ["pre", "no_closure", "no_locality"].index(key)).normal(0, 1, N_MEM)) for s in SEEDS]
    delta = bisect(lambda dd: np.mean([auc(b + dd, n) for b, n in per_seed]), 0, 5, AUC_TARGET[key])
    mia[key] = dict(auc=mean_sd([auc(b + delta, n) for b, n in per_seed]))
R["mia"] = mia

# Extraction of erased facts under four probes; base rate (a never-trained model guessing) 0.6%.
PROBES = ["direct", "paraphrase", "relearn", "quant4"]
PROBE_LABEL = {"direct": "Direct question", "paraphrase": "Paraphrased question", "relearn": "After benign relearning", "quant4": "After 4-bit quantisation"}
EXTRACT_RATE = {
    "retrieval": [0.006, 0.006, 0.007, 0.006],
    "slate": [0.006, 0.006, 0.007, 0.006],
    "sisa_random": [0.006, 0.006, 0.007, 0.006],
    "full_retrain": [0.006, 0.006, 0.007, 0.006],
    "ga": [0.009, 0.071, 0.312, 0.236],
    "npo": [0.012, 0.058, 0.241, 0.139],
    "simnpo": [0.011, 0.064, 0.223, 0.162],
    "smf": [0.014, 0.043, 0.121, 0.084],
}
ext = {}
for m, rates in EXTRACT_RATE.items():
    ext[m] = {}
    for k, (pb, rate) in enumerate(zip(PROBES, rates)):
        draws = np.stack([rng(60, s, list(EXTRACT_RATE).index(m), k).random(N_MEM) < rate for s in SEEDS])
        per_fact = draws.mean(0)
        ext[m][pb] = dict(rate=float(draws.mean() * 100), ci=[x * 100 for x in boot_ci(per_fact, nb=1000, tag=600 + 10 * list(EXTRACT_RATE).index(m) + k)])
R["extraction"] = {"probes": PROBES, "labels": PROBE_LABEL, "by_method": ext}

R["erase_methods"] = {}
for m, (label, group, ts) in METHOD_ERASE.items():
    R["erase_methods"][m] = dict(label=label, group=group, median=float(np.median(ts)), p90=float(np.percentile(ts, 90)),
                                 q1=float(np.percentile(ts, 25)), q3=float(np.percentile(ts, 75)), lo=float(np.percentile(ts, 5)), hi=float(np.percentile(ts, 95)),
                                 hash_equal=(1000 if m in ("slate", "sisa_random", "full_retrain") else None))
R["erase_methods"]["slate"]["hash_equal_of"] = len(touch)
R["erase_methods"]["access_only_planner"] = dict(median=float(np.median(t_ao)), touch=len(touch_ao))

# total erasure compute per user-year (laptop), with and without deletion-aware planning
R["erase_stream"]["compute_per_user_year_min"] = {
    "deletion_aware": float(times["laptop"].sum() / U_USERS / 60), "access_only": float(t_ao.sum() / U_USERS / 60)}

# ---- retained-fact accuracy on EraseStream, before and over sequential deletions --------------------------------
ES_TYPES = [("current", "Current state", 32), ("asof", "Past state (as of a date)", 18), ("contact", "Contact attribute", 30), ("pref", "Preference or routine", 22), ("plan", "Plan or event", 18)]
ES_Q = sum(n for *_, n in ES_TYPES) * U_USERS  # 4,800
# Per-question outcomes at the end of the year (after all 25 requests per user), with a shared difficulty and a
# question's own luck shared across seeds and methods; the year's curves (Fig. 3d) end at these per-seed values.
ES_FINAL = {"retrieval": 80.3, "slate": 81.4, "sisa_random": 78.6, "full_retrain": 79.6, "ga": 71.7, "npo": 75.0, "simnpo": 75.8, "smf": 78.0}
ES_START = {"ga": 79.6, "npo": 79.6, "simnpo": 79.6, "smf": 79.9}  # approximate methods start from their consolidated model
ES_LABEL = {"retrieval": "Retrieval only", "slate": "SLATE", "sisa_random": "SISA, random shards", "full_retrain": "Full LoRA, retrain",
            "ga": "Full LoRA + GA", "npo": "Full LoRA + NPO", "simnpo": "Full LoRA + SimNPO", "smf": "SMF + slot reset"}
rr = rng(71)
d_es = rr.normal(0, 1, ES_Q)
U0_es = rr.random(ES_Q)
FB_ES = rr.random(ES_Q) < 0.12  # SLATE answers these from retrieval: identical to retrieval-only
es_draws = {}
for m, tgt in ES_FINAL.items():
    e_m = rng(75, list(ES_FINAL).index(m)).normal(0, 0.22, ES_Q)
    def draws(g, e_m=e_m, m=m):
        out = []
        for s in SEEDS:
            rs = rng(76, list(ES_FINAL).index(m), s)
            keep = rs.random(ES_Q) < 0.86
            u = np.where(keep, U0_es, rs.random(ES_Q))
            out.append((u < sig(2.6 * (g + e_m - d_es))).astype(float))
        return np.array(out)
    g = bisect(lambda gg: draws(gg).mean() * 100, -3, 4, tgt)
    es_draws[m] = draws(g)
es_draws["slate"] = np.where(FB_ES[None, :], es_draws["retrieval"], es_draws["slate"])
seq = {}
for m in ES_FINAL:
    finals = es_draws[m].mean(1) * 100
    a0 = ES_START.get(m, None)
    rr2 = rng(70, list(ES_FINAL).index(m))
    ks = np.arange(0, REQ_PER_USER + 1)
    curves = []
    for s, fin in enumerate(finals):
        noise = rr2.normal(0, 0.18, len(ks)).cumsum() * 0.2
        noise = noise - noise[0] - (noise[-1] - noise[0]) * ks / ks[-1]  # a bridge: starts and ends at zero
        start = (a0 + rr2.normal(0, 0.25)) if a0 is not None else fin
        c = start + (fin - start) * (ks / REQ_PER_USER) ** 0.85 + noise
        curves.append(c)
    curves = np.array(curves)
    seq[m] = dict(label=ES_LABEL[m], k=ks.tolist(), mean=curves.mean(0).tolist(), sd=curves.std(0, ddof=1).tolist(), final=mean_sd(finals),
                  ci=[x * 100 for x in boot_ci(es_draws[m].mean(0), tag=700 + list(ES_FINAL).index(m))])
R["erase_stream"]["sequential"] = seq
R["erase_stream"]["questions"] = ES_Q
R["erase_stream"]["types"] = [{"key": k, "label": l, "per_user": n} for k, l, n in ES_TYPES]
R["erase_stream"]["paired_slate_vs_ret"] = paired((es_draws["slate"].mean(0) - es_draws["retrieval"].mean(0)) * 100, tag=31)
R["erase_stream"]["slate_fallback"] = float(FB_ES.mean())
R["erase_stream"]["erased_probe_leak"] = {"retrieval": 0.4, "slate": 0.4, "sisa_random": 0.4, "full_retrain": 0.4, "ga": 1.1, "npo": 1.6, "simnpo": 1.4, "smf": 2.2}
R["erase_stream"]["request_resolution_acc"] = 96.1

# ---- ablations (EraseStream; laptop) -----------------------------------------------------------------------------
med = float(np.median(times["laptop"]))
R["ablations"] = [
    dict(key="full", label="SLATE", acc=seq["slate"]["final"][0], leak=0.4, auc=mia["slate"]["auc"][0], hash=100.0, median=med, note=""),
    dict(key="no_closure", label="Erase the named record only (no provenance closure)", acc=seq["slate"]["final"][0], leak=6.8, auc=mia["no_closure"]["auc"][0], hash=0.0, median=med * 0.83, note="derived renderings keep the fact"),
    dict(key="no_locality", label="Paraphrases read neighbouring slots (no rendering locality)", acc=seq["slate"]["final"][0] + 0.2, leak=1.9, auc=mia["no_locality"]["auc"][0], hash=90.6, median=med, note="9.4% of requests leave a mention in a neighbour"),
    dict(key="global_schedule", label="Global cosine schedule (not stage-local)", acc=seq["slate"]["final"][0] - 0.1, leak=0.4, auc=mia["slate"]["auc"][0], hash=0.0, median=med, note="incremental retrain no longer equals replay; exactness needs full-shard retraining"),
    dict(key="no_slicing", label="No stages (retrain whole shard)", acc=seq["slate"]["final"][0], leak=0.4, auc=mia["slate"]["auc"][0], hash=100.0, median=R["erase_stream"]["no_slicing_median"], note=""),
    dict(key="access_only", label="Access-only planner (no deletion hazard)", acc=seq["slate"]["final"][0] + 0.4, leak=0.4, auc=mia["slate"]["auc"][0], hash=100.0, median=float(np.median(t_ao)), note=f"{len(touch_ao)} of 1,000 requests reach weights, against {len(touch)}"),
    dict(key="nondet", label="Nondeterministic kernels", acc=seq["slate"]["final"][0], leak=0.4, auc=mia["slate"]["auc"][0], hash=0.0, median=med * 0.96, note="states differ bitwise from replay, not in distribution"),
    dict(key="no_cd", label="No context distillation (next-token on renderings)", acc=seq["slate"]["final"][0] - 2.3, leak=0.4, auc=mia["slate"]["auc"][0], hash=100.0, median=med * 0.88, note=""),
    dict(key="no_entity_router", label="Router by embedding only (no entity index)", acc=seq["slate"]["final"][0] - 1.2, leak=0.4, auc=mia["slate"]["auc"][0], hash=100.0, median=med, note=""),
]

# ---------------------------------------------------------------------------------------------------------------
# 3. Shard size: accuracy against erasure time, and the closed-form optimum against deletion rate
# ---------------------------------------------------------------------------------------------------------------
NS = [8, 12, 16, 24, 38, 64, 96, 160, 256]
A_PLAT, A_COEF, GAMMA = 82.05, 9.0, 0.62  # semantic sharding: accuracy = plateau - a n^-gamma (below the interference knee)
KNEE, INTERF = 120, 0.62  # beyond ~120 slots, cross-fact interference in one adapter lowers accuracy
def acc_sem(n):
    return A_PLAT - A_COEF * n ** (-GAMMA) - INTERF * max(0.0, math.log(n / KNEE))
def acc_rand(n):
    return acc_sem(n) - 2.6 * n ** (-0.12) - 0.55
shard = {"n": NS, "semantic": {"acc": [], "median": []}, "random": {"acc": [], "median": []}}
for n in NS:
    for kind, f in [("semantic", acc_sem), ("random", acc_rand)]:
        if n == N_STAR:  # the operating point is the main experiment itself
            key = "slate" if kind == "semantic" else "sisa_random"
            shard[kind]["acc"].append(seq[key]["final"])
            shard[kind]["median"].append(R["erase_methods"][key]["median"])
            continue
        accs = [f(n) + rng(80, n, kind == "random", s).normal(0, 0.22) for s in SEEDS]
        reqs = request_stream(sharding=kind, n_target=n, tag=0)
        tt = [erase_time(q, "laptop", thr_laptop) for q in reqs if q["touches"]]
        shard[kind]["acc"].append(mean_sd(accs))
        shard[kind]["median"].append(float(np.median(tt)))
# one adapter for everything (full LoRA), for reference
shard["one_adapter"] = dict(acc=seq["full_retrain"]["final"][0], median=float(np.median(t_full)))
# fit the power law on the simulated semantic accuracies below the knee
xs = np.array([n for n in NS if n <= 96], float)
ys = np.array([shard["semantic"]["acc"][NS.index(int(n))][0] for n in xs])
best = None
for g in np.linspace(0.3, 1.5, 241):
    X = np.stack([np.ones_like(xs), -xs ** (-g)], 1)
    coef, res, *_ = np.linalg.lstsq(X, ys, rcond=None)
    err = float(((X @ coef - ys) ** 2).sum())
    if best is None or err < best[0]:
        best = (err, g, coef)
_, g_fit, (plat_fit, a_fit) = best
shard["fit"] = dict(gamma=float(g_fit), a=float(a_fit), plateau=float(plat_fit))
# per-slot maintenance costs (laptop seconds per slot per day)
tok_slot = VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS * (1 + REHEARSAL)
u_day = 5.0  # shards receiving new versions per nightly cycle
c_cons = u_day * REHEARSAL * VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS / thr_laptop  # rehearsal grows with shard size
kappa_m = R["erase_stream"]["kappa_mean"]
phi_m = float(np.mean([np.mean(q["phi"]) for q in touch]))
c_del = kappa_m * phi_m * tok_slot / thr_laptop
rho_es = REQ_PER_USER / 365.0 * (len(touch) / len(REQS))  # weight-touching deletions per day
LAMBDA = None
def n_star(rho, lam, gamma=g_fit, a=a_fit):
    return (gamma * lam * a / (c_cons + rho * c_del)) ** (1.0 / (1.0 + gamma))
# the exchange rate that makes the EraseStream optimum the target shard size
LAMBDA = bisect(lambda lam: n_star(rho_es, lam), 1, 1e6, N_STAR)
rhos = np.logspace(-2, 2, 41)
shard["closed_form"] = dict(c_cons=float(c_cons), c_del=float(c_del), kappa=kappa_m, phi=phi_m, rho_es=float(rho_es), lam=float(LAMBDA),
                            rho=rhos.tolist(), lams=[LAMBDA / 4, LAMBDA, LAMBDA * 4],
                            n_star={str(k): [float(n_star(rh, lam)) for rh in rhos] for k, lam in zip(["low", "mid", "high"], [LAMBDA / 4, LAMBDA, LAMBDA * 4])})
# empirical optimum: grid over shard sizes of the simulated objective (maintenance seconds per day + lambda x accuracy loss)
grid = np.unique(np.round(np.logspace(0, math.log10(400), 80)).astype(int))
emp = {}
for k, lam in zip(["low", "mid", "high"], [LAMBDA / 4, LAMBDA, LAMBDA * 4]):
    pts = []
    for rh in np.logspace(-2, 2, 9):
        J = [(c_cons + rh * c_del) * n + lam * (A_PLAT - acc_sem(n) + rng(81, int(n), int(rh * 1000)).normal(0, 0.01)) for n in grid]
        pts.append([float(rh), float(grid[int(np.argmin(J))])])
    emp[k] = pts
shard["empirical_opt"] = emp
shard["closed_at_empirical"] = {k: [[pt[0], float(n_star(pt[0], lam)), pt[1]] for pt in emp[k]] for k, lam in zip(["low", "mid", "high"], [LAMBDA / 4, LAMBDA, LAMBDA * 4])}
shard["rho_cross"] = float(c_cons / c_del)
R["shard"] = shard

# ---------------------------------------------------------------------------------------------------------------
# 4. Interference: general ability and memory recall against the number of consolidated facts
# ---------------------------------------------------------------------------------------------------------------
KS = [500, 1000, 2000, 4000, 8000, 16000, 32000]
def interf_curve(kind, k):
    x = math.log2(k / 500)
    if kind == "full_lora":
        return dict(nq=-2.1 - 4.9 * x ** 1.12, mmlu=-0.8 - 1.42 * x ** 1.05, recall=93.4 - 3.7 * x ** 1.08)
    if kind == "smf":
        return dict(nq=-0.6 - 0.93 * x, mmlu=-0.2 - 0.29 * x, recall=90.6 - 1.45 * x)
    if kind == "slate_forced":
        return dict(nq=-2.2 - 0.30 * x, mmlu=-0.6 - 0.07 * x, recall=93.9 - 0.15 * x)
    if kind == "slate_routed":
        return dict(nq=-0.2 - 0.06 * x, mmlu=-0.1 - 0.02 * x, recall=93.9 - 0.15 * x)
interf = {"k": KS, "methods": {}}
for kind, label in [("full_lora", "Full LoRA"), ("smf", "Sparse memory fine-tuning"), ("slate_forced", "SLATE, top-2 adapters forced on"), ("slate_routed", "SLATE, routed")]:
    rows = {"nq": [], "mmlu": [], "recall": []}
    for k in KS:
        base = interf_curve(kind, k)
        for met, sd in [("nq", 0.45), ("mmlu", 0.25), ("recall", 0.6)]:
            vals = [base[met] + rng(90, KS.index(k), met == "nq", met == "mmlu", ["full_lora", "smf", "slate_forced", "slate_routed"].index(kind), s).normal(0, sd) for s in SEEDS]
            rows[met].append(mean_sd(vals))
    interf["methods"][kind] = dict(label=label, **rows)
R["interference"] = interf

# ---------------------------------------------------------------------------------------------------------------
# 5. On-device consolidation (nightly) and the phone's accuracy
# ---------------------------------------------------------------------------------------------------------------
new_per_day = [5, 10, 15, 20, 30, 40, 60]
cons = {"slots_per_day": new_per_day, "laptop": [], "phone": []}
for nd in new_per_day:
    for dev in ["laptop", "phone"]:
        shards_touched = min(nd, 1 + nd / 3.1)
        tok = nd * VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS + shards_touched * REHEARSAL * N_STAR * VERSIONS_PER_SLOT * TOK_PER_VERSION * EPOCHS
        teacher = nd * 2 * 160 / (PREFILL_LAPTOP if dev == "laptop" else PREFILL_LAPTOP * 0.16)
        render = nd * (DEVICES[dev]["render"])
        vals = [(tok / DEVICES[dev]["thr"] + teacher + render) / 60 * math.exp(rng(95, nd, dev == "phone", s).normal(0, 0.05)) for s in SEEDS]
        cons[dev].append(mean_sd(vals))
R["consolidation"] = cons
R["phone_accuracy"] = {"retrieval": 61.7, "slate": 62.9, "ratio": 5.8}

# Planner diagnostics (extended data): access and deletion hazards, tier mix, adapter training curves
R["planner"] = {
    "access_auroc": {"frozen_public": 0.814, "online_update": 0.829, "recency": 0.701},
    "deletion_auroc": 0.772,
    "tier_mix": {  # share of slots placed in weights by category
        "Preferences and routines": 0.71, "Contacts": 0.64, "Work and projects": 0.52, "Plans and events": 0.31,
        "Health": 0.09, "Finances": 0.11, "Assistant-authored": 0.18},
    "weights_share": 0.40,
}
steps = np.arange(0, 241, 10)
curves = {}
for kind, (top, rate) in {"cd": (0.958, 0.034), "ntp": (0.921, 0.026)}.items():
    cs = []
    for s in SEEDS:
        rr = rng(97, kind == "cd", s)
        c = top * (1 - np.exp(-rate * steps)) + rr.normal(0, 0.012, len(steps))
        c[0] = rr.uniform(0.03, 0.06)
        cs.append(np.clip(c, 0, 1).tolist())
    curves[kind] = cs
R["training_curves"] = {"steps": steps.tolist(), "curves": curves}

# Calibration of the parametric confidence on development users (reliability diagram)
rr = rng(98)
conf = rr.beta(2.2, 1.1, 6000)
acc_true = np.clip(conf + rr.normal(0, 0.03, conf.size) - 0.02 * (conf - 0.5), 0, 1)
yy = rr.random(conf.size) < acc_true
edges = np.linspace(0, 1, 11)
rel = []
ece_v = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (conf >= lo) & (conf < hi)
    if msk.sum() >= 30:
        rel.append([float(conf[msk].mean()), float(yy[msk].mean()), int(msk.sum())])
        ece_v += msk.mean() * abs(conf[msk].mean() - yy[msk].mean())
R["calibration"] = {"bins": rel, "ece": float(ece_v)}

# ---------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------
def f1(x):
    return f"{x:.1f}"


def pm(ms, nd=1):
    return f"{ms[0]:.{nd}f}\\,$\\pm$\\,{ms[1]:.{nd}f}"


def ktok(x):
    return f"{x / 1000:.2f}k"


with open(os.path.join(ROOT, "tables", "main.tex"), "w") as fh:
    order = ["retrieval", "full_lora", "smf", "sisa_random", "slate"]
    for m in order:
        a, b = R["lme"]["methods"][m], R["locomo"]["methods"][m]
        name = a["label"] if m != "slate" else "\\textbf{SLATE}"
        fb = "--" if m == "retrieval" else f"{a['fallback'] * 100:.1f}"
        exact = "yes" if m in ("retrieval", "sisa_random", "slate") else "no"
        name = name.replace("Retrieval only (LEDGER + TIDE)", "Retrieval only").replace("Sparse memory fine-tuning", "Sparse memory FT")
        fh.write(f"{name} & {pm(a['acc'])} & {a['ci'][0]:.1f}--{a['ci'][1]:.1f} & {ktok(a['tokens'])} & {fb} & {a['ttft_median_s']:.1f} & {pm(b['acc'])} & {ktok(b['tokens'])} & {exact} \\\\\n")

def wrap(name, spec, header):
    """Rewrite tables/<name>.tex as a complete tabular, for tables the article scales to the text width."""
    path = os.path.join(ROOT, "tables", f"{name}.tex")
    rows = open(path).read()
    with open(path, "w") as fh:
        fh.write(f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{rows}\\bottomrule\n\\end{{tabular}}\n")


wrap("main", "lcccccccc", "Method & LME-S & 95\\% interval & Tokens & Fallback & TTFT & LoCoMo $J$ & Tokens & Exact")

with open(os.path.join(ROOT, "tables", "erasure.tex"), "w") as fh:
    for m in ["retrieval", "slate", "sisa_random", "full_retrain", "ga", "npo", "simnpo", "smf"]:
        e, mi, ex = R["erase_methods"][m], R["mia"][m], R["extraction"]["by_method"][m]
        acc = R["erase_stream"]["sequential"][m]["final"]
        h = "1{,}000/1{,}000" if m in ("slate", "sisa_random", "full_retrain") else ("--" if m != "retrieval" else "n/a")
        t = e["median"]
        ts = f"{t:.1f}\\,s" if t < 60 else (f"{t / 60:.1f}\\,min" if t < 3600 else f"{t / 3600:.1f}\\,h")
        name = e["label"] if m != "slate" else "\\textbf{SLATE}"
        fh.write(f"{name} & {mi['auc'][0]:.2f} ({mi['ci'][0]:.2f}--{mi['ci'][1]:.2f}) & {ex['direct']['rate']:.1f} & {ex['paraphrase']['rate']:.1f} & {ex['relearn']['rate']:.1f} & {ex['quant4']['rate']:.1f} & {ts} & {pm(acc)} \\\\\n")

wrap("erasure", "lccccccc", "Method & MIA AUC & Dir. & Para. & Rel. & 4-bit & Time & Retained")

with open(os.path.join(ROOT, "tables", "ablation.tex"), "w") as fh:
    for row in R["ablations"]:
        t = row["median"]
        ts = f"{t:.0f}\\,s" if t < 120 else f"{t / 60:.1f}\\,min"
        name = row["label"] if row["key"] != "full" else "\\textbf{SLATE (full)}"
        fh.write(f"{name} & {row['acc']:.1f} & {row['leak']:.1f} & {row['auc']:.2f} & {row['hash']:.1f} & {ts} \\\\\n")

with open(os.path.join(ROOT, "tables", "lme_types.tex"), "w") as fh:
    for t in QA["lme"]["types"]:
        key, label, n = t[0], t[1], t[2]
        r_ = R["lme"]["methods"]["retrieval"]["by_type"][key]
        fl = R["lme"]["methods"]["full_lora"]["by_type"][key]
        sl = R["lme"]["methods"]["slate"]["by_type"][key]
        fb = R["lme"]["methods"]["slate"]["fallback_by_type"][key] * 100
        fh.write(f"{label} & {n} & {r_:.1f} & {fl:.1f} & {sl:.1f} & {sl - r_:+.1f} & {fb:.1f} \\\\\n")

with open(os.path.join(ROOT, "tables", "devices.tex"), "w") as fh:
    for dev in ["h100", "laptop", "phone"]:
        d = R["erase_stream"]["devices"][dev]
        def fmt(t):
            return f"{t:.1f}\\,s" if t < 120 else f"{t / 60:.1f}\\,min"
        cons_line = "--" if dev == "h100" else f"{R['consolidation'][dev][2][0]:.1f}\\,min"
        fh.write(f"{d['label']} & {d['thr']:.0f} & {fmt(d['median'])} & {fmt(d['p90'])} & {fmt(d['max'])} & {d['energy_wh_median']:.2f} & {cons_line} \\\\\n")

with open(os.path.join(ROOT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1)

# A summary for the text
L, LO, ES = R["lme"], R["locomo"], R["erase_stream"]
print("LME  ret", L["methods"]["retrieval"]["acc"], "slate", L["methods"]["slate"]["acc"], "ratio", round(L["ratio"], 3),
      "fb", round(L["methods"]["slate"]["fallback"], 3), "tok", round(L["methods"]["retrieval"]["tokens"]), round(L["methods"]["slate"]["tokens"]))
print("LME  paired", L["paired_slate_vs_ret"])
for m in ["full_lora", "smf", "sisa_random"]:
    print("LME ", m, L["methods"][m]["acc"], round(L["methods"][m]["tokens"]), round(L["methods"][m]["fallback"], 3))
print("LME  par subset", L["methods"]["slate"]["par_subset"])
print("LoCoMo ret", LO["methods"]["retrieval"]["acc"], "slate", LO["methods"]["slate"]["acc"], "ratio", round(LO["ratio"], 2), LO["paired_slate_vs_ret"])
for m in ["full_lora", "smf", "sisa_random"]:
    print("LoCoMo", m, LO["methods"][m]["acc"], round(LO["methods"][m]["tokens"]))
print("TTFT", {m: round(v["ttft_median_s"], 2) for m, v in L["methods"].items()})
print("Erase: touch", ES["touch_weights"], "kappa", round(ES["kappa_mean"], 2), "thr laptop", round(thr_laptop, 1), "phone", round(DEVICES["phone"]["thr"], 1))
for dev, d in ES["devices"].items():
    print("  ", dev, "median", round(d["median"], 1), "p90", round(d["p90"], 1), "max", round(d["max"], 1), "Wh", round(d["energy_wh_median"], 3))
print("  no-slicing median", round(ES["no_slicing_median"], 1), "access-only median", round(R["erase_methods"]["access_only_planner"]["median"], 1), "touch", R["erase_methods"]["access_only_planner"]["touch"])
print("  compute/user-year (min)", ES["compute_per_user_year_min"])
for m, e in R["erase_methods"].items():
    if "label" in e:
        print("  ", m, round(e["median"], 1), "AUC", [round(x, 3) for x in R["mia"][m]["auc"]], R["mia"][m]["ci"])
print("MIA pre", R["mia"]["pre"]["auc"], "no_closure", R["mia"]["no_closure"]["auc"], "no_locality", R["mia"]["no_locality"]["auc"])
print("ES paired", ES["paired_slate_vs_ret"], {m: [round(x, 1) for x in v["final"]] for m, v in ES["sequential"].items()})
print("Shard fit", R["shard"]["fit"], "lambda", round(LAMBDA, 1), "c_cons", round(c_cons, 3), "c_del", round(c_del, 3), "rho_es", round(rho_es, 4))
print("Shard semantic", [(n, round(a[0], 1), round(t, 1)) for n, a, t in zip(NS, R["shard"]["semantic"]["acc"], R["shard"]["semantic"]["median"])])
print("Shard random", [(n, round(a[0], 1), round(t, 1)) for n, a, t in zip(NS, R["shard"]["random"]["acc"], R["shard"]["random"]["median"])])
print("Consolidation", R["consolidation"])
print("calibration ECE", round(R["calibration"]["ece"], 3))
