"""Publication figures (PDF + 300 dpi PNG) -> results/figures/.

Palette: validated categorical slots 1-3 (all-pairs CVD dE >= 9.2, normal-vision dE >= 24).
The aqua slot is < 3:1 on white, so every series also carries a distinct marker/line style
and a direct label (identity is never color-alone; survives grayscale print).

    python -m analysis.figures
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finverity.config import RESULTS
import json

from .stats import RAW, apply_finality_depth, load_summaries

FIG = RESULTS / "figures"
ORDER = ["purechain", "poa2_local", "besu", "db_postgres", "ethpos"]
LABEL = {"purechain": "PureChain (live)", "poa2_local": "Clique emulation", "besu": "Besu QBFT", "db_postgres": "PostgreSQL",
         "ethpos": "Ethereum PoS (public-chain config.)"}
SHORT = {"purechain": "PureChain (live)", "poa2_local": "Clique emulation", "besu": "Besu QBFT", "db_postgres": "PostgreSQL", "ethpos": "Ethereum PoS"}
# validated categorical slots 1-4 (adjacent-pair CVD dE >= 9.1); slots 3-4 < 3:1 -> markers + direct labels
COLOR = {"purechain": "#2a78d6", "poa2_local": "#7da7dc", "besu": "#eb6834", "db_postgres": "#1baf7a", "ethpos": "#eda100"}
MARKER = {"purechain": "o", "poa2_local": "P", "besu": "s", "db_postgres": "^", "ethpos": "D"}
LINE = {"purechain": "-", "poa2_local": (0, (5, 2, 1, 2)), "besu": "--", "db_postgres": ":", "ethpos": "-."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "figure.dpi": 150,
})


def _spread(ys, gap=9.0):
    """Label positions (points) at least `gap` apart, each cluster centred on its anchors."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    clusters = []
    for i in order:   # greedy: merge a label into the cluster below while they would overlap
        clusters.append([i])
        while len(clusters) > 1:
            lo, hi = clusters[-2], clusters[-1]
            top_lo = sum(ys[j] for j in lo) / len(lo) + (len(lo) - 1) * gap / 2
            bot_hi = sum(ys[j] for j in hi) / len(hi) - (len(hi) - 1) * gap / 2
            if bot_hi - top_lo >= gap:
                break
            clusters[-2:] = [lo + hi]
    out = [0.0] * len(ys)
    for c in clusters:
        start = sum(ys[j] for j in c) / len(c) - (len(c) - 1) * gap / 2
        for k, j in enumerate(c):
            out[j] = start + k * gap
    return out


def _place_direct_labels(fig):
    """Place queued end-of-line labels once the axes have their final limits, then make room for them."""
    queued = [(ax, getattr(ax, "_pending_labels", [])) for ax in fig.axes]
    if not any(p for _, p in queued):
        return
    made = []
    for rnd in range(2):   # second round re-places the labels after the layout has made room for them
        for a in made:
            a.remove()
        made = []
        fig.canvas.draw()
        for ax, pending in queued:
            if not pending:
                continue
            ys = [ax.transData.transform((x, y))[1] * 72 / fig.dpi for x, y, _, _ in pending]
            targets = _spread(ys)
            top = ax.transAxes.transform((0, 1))[1] * 72 / fig.dpi - 6   # keep labels below the panel title
            shift = max(0.0, max(targets) - top)
            targets = [v - shift for v in targets]
            for (x, y, b, dx), yy, target in zip(pending, ys, targets):
                made.append(ax.annotate(SHORT[b], (x, y), xytext=(dx, target - yy), textcoords="offset points",
                                        va="center", fontsize=7, color=INK2, annotation_clip=False))
        if rnd == 0:
            fig.tight_layout()
    for ax, _ in queued:
        ax._pending_labels = []


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    _place_direct_labels(fig)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("  wrote", name)


def _backends(df):
    return [b for b in ORDER if b in set(df["backend"])]


def _series(ax, x, y, b, lo=None, hi=None, label=True):
    ax.plot(x, y, linestyle=LINE[b], color=COLOR[b], marker=MARKER[b], markeredgecolor="white", markeredgewidth=0.8,
            label=LABEL[b])
    if lo is not None:
        ax.fill_between(x, lo, hi, color=COLOR[b], alpha=0.12, linewidth=0)


def _direct_label(ax, x, y, b, dx=4):
    if not len(x):
        return
    # placed in _save, after every series is drawn and the axis limits are final
    ax._pending_labels = getattr(ax, "_pending_labels", []) + [(x[-1], y[-1], b, dx)]


def _lat(run_ids):
    fr = [pd.read_parquet(RAW / r / "items.parquet", columns=["status", "commit_latency_s"]) for r in run_ids]
    if not fr:
        return np.array([])
    d = pd.concat(fr)
    return d.loc[d["status"] == "committed", "commit_latency_s"].dropna().to_numpy()


def _agg(g, col):
    from .stats import ci95
    x = g[col].astype(float).dropna()
    m = x.mean()
    lo, hi = ci95(x) if len(x) > 1 else (m, m)
    return m, lo, hi


# ---------------------------------------------------------------- F1 latency CDF
def fig_latency_cdf(df):
    sub = df[df["workload"] == "W2"]
    if not len(sub):
        sub = df[df["workload"].str.match(r"W\d")]
    if not len(sub):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for b in _backends(sub):
        lat = np.sort(_lat(sub.loc[sub["backend"] == b, "run_id"]))
        if not len(lat):
            continue
        y = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, y, linestyle=LINE[b], color=COLOR[b], label=LABEL[b])
        i = int(0.5 * len(lat))
        ax.plot(lat[i], y[i], MARKER[b], color=COLOR[b], markeredgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Commit latency (s, log scale)")
    ax.set_ylabel("Fraction of records")
    ax.axhline(0.95, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(ax.get_xlim()[0] * 1.1, 0.955, "P95", fontsize=6.5, color=INK2, va="bottom")
    ax.set_title(f"Commit-latency distribution ({sub['workload'].iloc[0]})", loc="left")
    ax.legend(loc="lower right")
    _save(fig, "F1_latency_cdf")


# ---------------------------------------------------------------- F2 offered-load sweep
def fig_load_sweep(df):
    sub = df[df["workload"] == "E2"].copy()
    if not len(sub):
        return
    maxr = sub.loc[sub["rate"] > 0, "rate"].max()
    sub["x"] = np.where(sub["rate"] > 0, sub["rate"], maxr * 2)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
    for b in _backends(sub):
        g = sub[sub["backend"] == b].groupby("x")
        xs = np.array(sorted(g.groups))
        th = np.array([_agg(g.get_group(x), "throughput_tps") for x in xs])
        p50 = np.array([_agg(g.get_group(x), "commit_latency_s_median") for x in xs])
        p95 = np.array([_agg(g.get_group(x), "commit_latency_s_p95") for x in xs])
        _series(axs[0], xs, th[:, 0], b, th[:, 1], th[:, 2])
        _direct_label(axs[0], xs, th[:, 0], b)
        _series(axs[1], xs, p50[:, 0], b)
        axs[1].plot(xs, p95[:, 0], linestyle=LINE[b], color=COLOR[b], alpha=0.45, lw=1.0)
        _direct_label(axs[1], xs, p50[:, 0], b)
    lim = [1, maxr * 2.6]
    axs[0].plot(lim, lim, color=INK2, lw=0.6, ls=(0, (2, 2)), label="ideal (throughput = offered)")
    for ax in axs:
        ax.set_xscale("log")
        ticks = sorted(sub["x"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" if t <= maxr else "max" for t in ticks], rotation=45, ha="right")
        ax.set_xlabel("Offered load (tx/s)")
    axs[0].set_yscale("log"); axs[0].set_ylabel("Achieved throughput (tx/s)")
    axs[0].set_title("Throughput vs offered load", loc="left")
    axs[1].set_yscale("log"); axs[1].set_ylabel("Commit latency (s)")
    axs[1].set_title("Latency vs offered load (solid P50, faint P95)", loc="left")
    axs[0].legend(loc="upper left", fontsize=6.3)
    fig.tight_layout()
    _save(fig, "F2_offered_load_sweep")


# ---------------------------------------------------------------- F3 workloads W1-W5
def fig_workloads(df):
    sub = df[df["workload"].str.match(r"W\d")]
    if not len(sub):
        return
    ws = sorted(sub["workload"].unique())
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.4), sharey=False)
    for ax, col, title in zip(axs, ["commit_latency_s_median", "commit_latency_s_p95", "throughput_tps"],
                              ["Median commit latency (s)", "P95 commit latency (s)", "Throughput (tx/s)"]):
        for b in _backends(sub):
            g = sub[sub["backend"] == b].groupby("workload")
            xs = [w for w in ws if w in g.groups]
            v = np.array([_agg(g.get_group(w), col) for w in xs])
            xi = np.array([ws.index(w) for w in xs])
            ax.errorbar(xi, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], marker=MARKER[b], linestyle=LINE[b],
                        color=COLOR[b], markeredgecolor="white", capsize=2, elinewidth=0.8, label=LABEL[b])
        ax.set_xticks(range(len(ws))); ax.set_xticklabels(ws)
        ax.set_title(title, loc="left")
        if col != "throughput_tps":
            ax.set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=len(l), fontsize=6.5, frameon=False)
    fig.text(0.5, -0.02, "Workload (W1 1k/10 KB · W2 10k/10 KB · W3 50k/100 KB · W4 100k/100 KB · W5 10k/1 MB); "
             "error bars: 95% CI over trials", ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout()
    _save(fig, "F3_workloads")


# ---------------------------------------------------------------- F4 evidence size
def fig_evidence_size(df):
    sub = df[df["workload"] == "E3"].copy()
    if not len(sub):
        return
    sub["kb"] = sub["doc_bytes"].replace(0, 1024 * 0.5) / 1024
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for b in _backends(sub):
        g = sub[sub["backend"] == b].groupby("kb")
        xs = np.array(sorted(g.groups))
        prep = np.array([_agg(g.get_group(x).assign(p=lambda d: 1e3 * (d["prep_evidence_s_mean"] + d["prep_store_s_mean"])), "p") for x in xs])
        lat = np.array([_agg(g.get_group(x), "commit_latency_s_median") for x in xs])
        _series(axs[0], xs, prep[:, 0], b, prep[:, 1], prep[:, 2]); _direct_label(axs[0], xs, prep[:, 0], b)
        _series(axs[1], xs, lat[:, 0], b, lat[:, 1], lat[:, 2]); _direct_label(axs[1], xs, lat[:, 0], b)
    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Document size per record (KB; leftmost = none)")
    axs[0].set_ylabel("Client evidence processing (ms/record)")
    axs[0].set_title("Evidence generation + hashing + off-chain store", loc="left")
    axs[1].set_ylabel("Median commit latency (s)")
    axs[1].set_title("Commit latency is independent of evidence size", loc="left")
    fig.tight_layout()
    _save(fig, "F4_evidence_size")


# ---------------------------------------------------------------- F5 complexity
def fig_complexity(df):
    sub = df[df["workload"] == "E4"]
    if not len(sub):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.4))
    for b in _backends(sub):
        g = sub[sub["backend"] == b].groupby("level")
        xs = np.array(sorted(g.groups))
        pkg = np.array([_agg(g.get_group(x).assign(p=lambda d: 1e3 * d["prep_evidence_s_mean"]), "p") for x in xs])
        lat = np.array([_agg(g.get_group(x), "commit_latency_s_median") for x in xs])
        _series(axs[0], xs, pkg[:, 0], b, pkg[:, 1], pkg[:, 2]); _direct_label(axs[0], xs, pkg[:, 0], b)
        _series(axs[1], xs, lat[:, 0], b, lat[:, 1], lat[:, 2]); _direct_label(axs[1], xs, lat[:, 0], b)
    for ax in axs:
        ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["L1\nmetadata", "L2\n+invoice", "L3\n+authoriz.", "L4\ncomplete"])
    axs[0].set_ylabel("Evidence generation + hash (ms)"); axs[0].set_title("Evidence complexity cost (client)", loc="left")
    axs[1].set_ylabel("Median commit latency (s)"); axs[1].set_yscale("log")
    axs[1].set_title("Commit latency by complexity level", loc="left")
    fig.tight_layout()
    _save(fig, "F5_complexity")


# ---------------------------------------------------------------- F6 validators
def fig_validators(df):
    sub = df[(df["workload"] == "E5") & ~df["tag"].fillna("").str.contains("confirm")].copy()
    if not len(sub):
        return
    sub["v"] = sub["tag"].str.extract(r"v(\d+)")[0].astype(int)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for b in _backends(sub):
        for rate in sorted(sub["rate"].unique()):
            g = sub[(sub["backend"] == b) & (sub["rate"] == rate)].groupby("v")
            xs = np.array(sorted(g.groups))
            if not len(xs):
                continue
            th = np.array([_agg(g.get_group(x), "throughput_tps") for x in xs])
            ok = np.array([g.get_group(x)["success_rate"].mean() for x in xs])
            lat = np.array([g.get_group(x)["e2e_latency_s_median"].mean() for x in xs])
            ls = "-" if rate == sub["rate"].min() else "--"
            for ax, v in ((axs[0], th[:, 0]), (axs[1], lat)):
                ax.plot(xs, v, ls, color=COLOR[b], marker=MARKER[b], markeredgecolor="white",
                        label=f"{SHORT[b]} @ {rate:g} rec/s")
                _lbl = f"{SHORT[b]} @ {rate:g}"
                ax.annotate(_lbl, (xs[-1], v[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.3,
                            color=INK2, va="center")
            for x, o, t in zip(xs, ok, th[:, 0]):
                if o < 0.99:
                    axs[0].annotate(f"{100 * o:.0f}% committed", (x, t), xytext=(-6, 10), textcoords="offset points",
                                    fontsize=6.3, color=INK2, ha="right",
                                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    for ax in axs:
        ax.set_xticks(sorted(sub["v"].unique())); ax.set_xlabel("Number of validators (1 CPU each)")
        ax.set_xlim(3.5, 12.5)
    axs[0].set_ylabel("Throughput (records/s)"); axs[0].set_title("Throughput vs validator count", loc="left")
    axs[1].set_yscale("log"); axs[1].set_ylabel("Median inclusion latency (s)")
    axs[1].set_title("Latency vs validator count", loc="left")
    fig.tight_layout()
    _save(fig, "F6_validators")


# ---------------------------------------------------------------- F7 energy & cost
def fig_energy_cost(df):
    sub = df[df["workload"] == "W2"]
    if not len(sub):
        sub = df[df["workload"].str.match(r"W\d")]
    if not len(sub):
        return
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    bs = _backends(sub)
    y = np.arange(len(bs))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.3))
    panels = ((axs[0], "energy_j_per_tx", "Attributed energy (J/record)", 1.0),
              (axs[1], "energy_dynamic_j_per_tx", "Measured dynamic energy (J/record)", 1.0),
              (axs[2], "cost_total_usd_per_tx", "Total cost per record (USD, incl. fees)", 1.0))
    for ax, col, title, scale in panels:
        vals = [_agg(sub[sub["backend"] == b], col) for b in bs]
        for i, (b, v) in enumerate(zip(bs, vals)):
            ax.barh(i, v[0] * scale, color=COLOR[b], height=0.6, edgecolor="white", linewidth=2)
            txt = f"{v[0]:.3g}" if col != "cost_total_usd_per_tx" else (f"${v[0]:.2f}" if v[0] >= 0.01 else f"${v[0]:.1e}")
            if col == "energy_dynamic_j_per_tx" and b == "purechain":
                txt += " (client only)"
            ax.text(v[0] * scale * 1.15, i, txt, va="center", fontsize=6.8, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[b] for b in bs] if ax is axs[0] else [""] * len(bs))
        ax.invert_yaxis(); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext()); ax.xaxis.set_minor_formatter(NullFormatter())
        lo = min(v[0] for v in vals if v[0] > 0)
        hi = max(v[0] for v in vals)
        ax.set_xlim(lo / 3, hi * 30)
        ax.set_title(title, loc="left", fontsize=8)
    fig.tight_layout()
    _save(fig, "F7_energy_cost")


# ---------------------------------------------------------------- F8 verification breakdown
def fig_verification(df):
    sub = df[df["workload"].str.match(r"W\d")]
    if not len(sub):
        return
    bs = _backends(sub)
    rows = []
    for b in bs:
        fr = [pd.read_parquet(RAW / r / "verify.parquet") for r in sub.loc[sub["backend"] == b, "run_id"]
              if (RAW / r / "verify.parquet").exists()]
        fr = [f for f in fr if len(f)]
        if fr:
            v = pd.concat(fr)
            rows.append((b, 1e3 * v["chain_fetch_s"].median(), 1e3 * v["store_fetch_s"].median(),
                         1e3 * v["rehash_s"].median(), v["match"].mean(), len(v)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    parts = [("record lookup", "#1c5cab"), ("evidence retrieval", "#5598e7"), ("re-hash + compare", "#b7d3f6")]
    for i, (b, *vals, ok, n) in enumerate(rows):
        left = 0
        for (name, c), v in zip(parts, vals):
            ax.barh(i, v, left=left, color=c, height=0.6, edgecolor="white", linewidth=2, label=name if i == 0 else None)
            left += v
        ax.text(left, i, f"  {left:.2f} ms · {100 * ok:.1f}% verified (n={n})", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([SHORT[r[0]] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Median verification time per record (ms)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(r[1:4]) for r in rows) * 2.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.5, frameon=False)
    ax.set_title("Continuous evidence verification", loc="left")
    _save(fig, "F8_verification")


# ---------------------------------------------------------------- F9 finality ladder
def fig_finality(df):
    sub = df[df["workload"] == "W2"]
    if not len(sub):
        sub = df[df["workload"].str.match(r"W\d")]
    if not len(sub):
        return
    stages = [("ack_latency_s", "Submission ack"), ("commit_latency_s", "Canonical inclusion"),
              ("conf1_latency_s", "1 confirmation"), ("conf3_latency_s", "3 confirmations"),
              ("final_latency_s", "Finality (backend rule)")]
    bs = [b for b in _backends(sub) if b != "db_postgres"]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    off = {b: (i - (len(bs) - 1) / 2) * 0.22 for i, b in enumerate(bs)}
    for b in bs:
        g = sub[sub["backend"] == b]
        med = [g[f"{c}_median"].mean() for c, _ in stages]
        p95 = [g[f"{c}_p95"].mean() for c, _ in stages]
        y = np.arange(len(stages)) + off[b]
        ax.hlines(y, med, p95, color=COLOR[b], lw=1.4)
        ax.plot(med, y, MARKER[b], color=COLOR[b], markeredgecolor="white", label=f"{SHORT[b]} (marker median, line to P95)")
    if "db_postgres" in set(sub["backend"]):
        v = sub.loc[sub["backend"] == "db_postgres", "commit_latency_s_median"].mean()
        ax.axvline(v, color=COLOR["db_postgres"], ls=":", lw=1.2)
        ax.text(v * 1.15, len(stages) - 0.55, f"PostgreSQL commit {1e3 * v:.0f} ms", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels([n for _, n in stages]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("Latency from submission (s, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Inclusion vs confirmation-depth finality", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=6.3, ncol=1, frameon=False)
    _save(fig, "F9_finality_ladder")


# ---------------------------------------------------------------- F10 batching
def fig_batching(df):
    sub = df[df["workload"] == "E9"]
    if not len(sub):
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    rate = sub["rate"].max()
    g0 = sub[sub["rate"] == rate]
    for b in _backends(g0):
        g = g0[g0["backend"] == b].groupby("batch_size")
        xs = np.array(sorted(g.groups))
        for ax, col in ((axs[0], "throughput_tps"), (axs[1], "final_e2e_latency_s_median"), (axs[2], "gas_used_mean")):
            if col == "gas_used_mean" and b == "db_postgres":
                continue
            v = np.array([_agg(g.get_group(x), col) for x in xs])
            _series(ax, xs, v[:, 0], b, v[:, 1], v[:, 2])
            if ax is not axs[0]:
                _direct_label(ax, xs, v[:, 0], b)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("Records per commitment k")
    axs[0].axhline(rate, color=INK2, lw=0.6, ls=(0, (2, 2)))
    axs[0].set_ylabel("Records committed per second"); axs[0].set_title(f"Throughput @ {rate:g} rec/s offered", loc="left")
    axs[1].set_yscale("log"); axs[1].set_ylabel("Arrival -> finality (s)"); axs[1].set_title("End-to-end finality latency", loc="left")
    axs[2].set_yscale("log"); axs[2].set_ylabel("Gas per record"); axs[2].set_title("On-chain cost per record", loc="left")
    fig.tight_layout()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(l), fontsize=6.3, frameon=False)
    _save(fig, "F10_batching")


# ---------------------------------------------------------------- F11 insider tampering
def fig_insider(df):
    frames = [pd.read_parquet(p).assign(run_id=p.parent.name) for p in RAW.glob("*/insider.parquet")]
    if not frames:
        return
    d = pd.concat(frames)
    d["backend"] = d["run_id"].map(df.set_index("run_id")["backend"])
    d = d.dropna(subset=["backend"])
    scen = [("S1_evidence_only", "Alter evidence"), ("S2_commitment_rewrite", "Alter evidence +\nrewrite commitment"),
            ("S3_backdated_record", "Fabricate +\nbackdate record")]
    bs = _backends(d)
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    w = 0.8 / len(bs)
    for i, b in enumerate(bs):
        vals = []
        for sc, _ in scen:
            g = d[(d["backend"] == b) & (d["scenario"] == sc)]
            vals.append(100 * g["outcome"].isin(["prevented", "detected"]).mean() if len(g) else np.nan)
        x = np.arange(len(scen)) + (i - (len(bs) - 1) / 2) * w
        ax.bar(x, vals, width=w, color=COLOR[b], edgecolor="white", linewidth=2, label=SHORT[b])
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", rotation=90, fontsize=6, color=INK)
    ax.set_xticks(range(len(scen))); ax.set_xticklabels([n for _, n in scen])
    ax.set_ylim(0, 125); ax.set_ylabel("Attacks prevented or detected (%)")
    ax.set_title("Privileged-insider tampering", loc="left"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, ncol=len(bs), frameon=False)
    _save(fig, "F11_insider")


# ---------------------------------------------------------------- F12 live forks
def fig_forks(df):
    """Reorganisation depths on live PureChain per client release, and the derived confirmation depth."""
    from .report import client_generation
    runs = []
    for sp in [*RAW.glob("*/summary.json"), *(RESULTS / "_archive_v0" / "raw").glob("*/summary.json"),
               *(RESULTS / "_archive_live_v1").glob("*/summary.json")]:
        sm = json.loads(sp.read_text())
        bf = sp.parent / "blocks.csv"
        if sm.get("backend") == "purechain" and bf.exists() and bf.stat().st_size > 200:
            runs.append((sp.parent, client_generation(sm.get("client"))))
    if not runs:
        return
    depths = {"old": [], "new": []}
    heights = {"old": 0, "new": 0}
    for d, gen in runs:
        b = pd.read_csv(d / "blocks.csv")
        if "parent" not in b:
            continue
        if "canonical" not in b:           # early trials: rebuild the canonical chain from the tip's ancestry
            parent = dict(zip(b["hash"], b["parent"]))
            h, canon_h = b.sort_values(["number", "t_observed"]).iloc[-1]["hash"], set()
            while h in parent:
                canon_h.add(h)
                h = parent[h]
            b["canonical"] = b["hash"].isin(canon_h)
        canon = set(b.loc[b["canonical"], "number"])
        replaced = sorted(set(b.loc[~b["canonical"], "number"]) & canon)
        heights[gen] += len(canon)
        run, prev = 0, None
        for h in replaced:
            if prev is not None and h == prev + 1:
                run += 1
            else:
                if run:
                    depths[gen].append(run)
                run = 1
            prev = h
        if run:
            depths[gen].append(run)
    dmax = max(max(v, default=0) for v in depths.values())
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs = np.arange(1, dmax + 2)
    for gen, lab, col, mk, off in (("old", "Sept. 2026 client", INK2, "s", -0.18), ("new", "Release 1.0.1", COLOR["purechain"], "o", 0.18)):
        if not heights[gen]:
            continue
        # share of observed heights at which a reorganisation of at least depth x started
        y = [100 * sum(1 for v in depths[gen] if v >= x) / heights[gen] for x in xs]
        ax.bar(xs + off, y, width=0.34, color=col, label=f"{lab} ({heights[gen]:,} heights)")
    ax.set_yscale("log")                    # deep reorganisations are orders of magnitude rarer than shallow ones
    ax.axvline(dmax + 1, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(dmax + 1.08, 10 ** np.mean(np.log10(ax.get_ylim())), f"k = {dmax + 1}", fontsize=7, color=INK)
    ax.set_xticks(xs)
    ax.set_xlabel("Reorganisation depth d (blocks)")
    ax.set_ylabel("Reorgs of depth ≥ d\nper 100 heights (log)")
    ax.set_title("Live PureChain forks and finality depth", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=1, fontsize=6.8, frameon=False)
    _save(fig, "F12_forks")


# ---------------------------------------------------------------- F13 demand-driven sealing
def fig_idle(df):
    w = RESULTS / "idle" / "wake.csv"
    if not w.exists():
        return
    d = pd.read_csv(w)
    d = d[d["status"] == "committed"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for t, g in d.groupby("target_idle_s"):
        y = g["commit_latency_s"].to_numpy()
        ax.scatter(np.full(len(y), t) + np.random.default_rng(0).uniform(-2, 2, len(y)), y, s=8,
                   color=COLOR["purechain"], alpha=0.5, linewidths=0)
        ax.plot([t - 5, t + 5], [np.median(y)] * 2, color=INK, lw=1.4)
    ax.set_xlabel("Time the chain had been idle before submission (s)")
    ax.set_ylabel("Inclusion latency (s)")
    ax.set_title("Cost of waking an idle PureChain network", loc="left")
    _save(fig, "F13_wake")


# ---------------------------------------------------------------- F14 / F15 burst-aware anchoring
POLICY_LABEL = {"record": "Per record", "fixed": "Fixed b=50", "adaptive": "Controller"}
POLICY_HATCH = {"record": "", "fixed": "///", "adaptive": "xx"}


def _policy(s) -> str:
    if s.get("batch_policy") == "adaptive":
        return "adaptive"
    return "fixed" if int(s.get("batch_size", 1) or 1) > 1 else "record"


def fig_bursts(df):
    e = df[df["workload"] == "E11"].copy()
    if not len(e):
        return
    e["pol"] = e.apply(_policy, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (src, title) in zip(axs, (("saml_d", "SAML-D: business day"), ("ibm_aml", "IBM AML: batch posting"))):
        bs = [b for b in ORDER if b in set(e.loc[e["source"] == src, "backend"])]   # PostgreSQL: per record only
        for i, b in enumerate(bs):
            for j, pol in enumerate(POLICY_LABEL):
                g = e[(e["source"] == src) & (e["backend"] == b) & (e["pol"] == pol)]
                if not len(g):
                    continue
                m, lo, hi = _agg(g, "e2e_latency_s_p95")
                x = i + (j - 1) * 0.26
                ax.bar(x, m, width=0.24, color=COLOR[b], hatch=POLICY_HATCH[pol], edgecolor="white", linewidth=0.5,
                       label=POLICY_LABEL[pol] if i == 0 else None)
                ax.plot([x, x], [max(lo, 1e-3), hi], color=INK, lw=0.8)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([SHORT[b].replace(" ", "\n", 1) for b in bs])   # two-line labels never collide
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
    for ax in axs:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].set_ylabel("Arrival → inclusion, P95 (s, log)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1), frameon=False)
    _save(fig, "F14_bursts")


def fig_controller(df):
    e = df[(df["workload"] == "E11") & (df["batch_policy"] == "adaptive") & (df["source"] == "ibm_aml")]
    live = e[e["backend"] == "purechain"]
    e = live if len(live) else e
    if not len(e):
        return
    rid = e.sort_values("trial")["run_id"].iloc[0]
    it = pd.read_parquet(RAW / rid / "items.parquet", columns=["t_sched", "t_send", "e2e_latency_s", "status"])
    rid_rec = rid.replace("_ad16w1", "")
    fig, axs = plt.subplots(2, 1, figsize=(3.6, 3.4), sharex=True)
    bins = np.arange(0, it["t_sched"].max() + 5, 5)
    axs[0].plot(bins[:-1], np.histogram(it["t_sched"], bins)[0] / 5, color=INK2, lw=1.0, label="Arrivals")
    sub = it.groupby("t_send").size()                      # records per submission (one t_send per transaction)
    axs[0].scatter(sub.index, sub.values, s=4, color=COLOR["purechain"], label="Records per transaction")
    axs[0].set_yscale("log"); axs[0].set_ylabel("Records/s ·\nrecords/tx"); axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.3, frameon=False)
    axs[1].scatter(it["t_sched"], it["e2e_latency_s"], s=2, color=COLOR["purechain"], label="Adaptive")
    if (RAW / rid_rec / "items.parquet").exists():
        rec = pd.read_parquet(RAW / rid_rec / "items.parquet", columns=["t_sched", "e2e_latency_s"])
        axs[1].scatter(rec["t_sched"], rec["e2e_latency_s"], s=2, color=COLOR["besu"], alpha=0.5, label="Per record")
    axs[1].set_yscale("log"); axs[1].set_ylabel("Inclusion latency (s)"); axs[1].set_xlabel("Time in trial (s)")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.3, markerscale=3, frameon=False)
    axs[0].set_title(f"Controller on {SHORT[e['backend'].iloc[0]]}, IBM AML trace", loc="left")
    _save(fig, "F15_controller")


def main(k: int | None = None):
    df = load_summaries()
    if not len(df):
        print("no results")
        return
    if k:
        df = apply_finality_depth(df, k)
    for f in (fig_latency_cdf, fig_load_sweep, fig_workloads, fig_evidence_size, fig_complexity,
              fig_validators, fig_energy_cost, fig_verification, fig_finality, fig_batching, fig_insider,
              fig_forks, fig_idle, fig_bursts, fig_controller):
        try:
            f(df)
        except Exception as e:
            print(f"  {f.__name__} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
