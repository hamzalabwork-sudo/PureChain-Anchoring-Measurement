"""Analyses added in revision (called from analysis.report.build).
* E13 submission-policy baselines and utilisation sweep: per-record inclusion and finality percentiles with
  block-bootstrap intervals, gas per record, Pareto view (Figure F16).
* E14 finality with the client-side tail and on the locally run PureChain client, and a validator outage.
* Inclusion-delay distribution D per ledger, and the share of records beyond the bound w + 2D.
* Reorganisation-depth risk: per-event depths, geometric fit, rule of three, k for a target risk.
* Censored finality: Kaplan-Meier against complete-case finality for trials whose block log ended early.
* Signer coalitions on the Clique emulation; write-once gas by batch size.
* E16 forks, out-of-turn sealing and finality of the local PureChain client under network delay between validators.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from finverity.config import RESULTS
from .ots import build as ots_build
from .passive import build as passive_build
from .stats import RAW

W = 1.0                      # batch timeout w used by every controller trial (s)
RATE_WORD = {5: "Five", 15: "Fifteen", 30: "Thirty", 60: "Sixty", 120: "OneTwenty"}
BKEY = {"purechain": "Live", "poa2_local": "Emu", "besu": "Besu", "db_postgres": "DB", "purechain_local": "Pcl"}
POLICIES = {"record": ("Per record", "Rec"), "fixed_w0.1": ("Fixed $b{=}50$, $w{=}0.1$\\,s", "FixTenth"),
            "fixed_w0.25": ("Fixed $b{=}50$, $w{=}0.25$\\,s", "FixQuarter"),
            "fixed_w0.5": ("Fixed $b{=}50$, $w{=}0.5$\\,s", "FixHalf"), "fixed_w1": ("Fixed $b{=}50$, $w{=}1$\\,s", "FixOne"),
            "interval": ("Interval, 1\\,s", "Iv"), "drain": ("Drain per block", "Drain"),
            "adaptive_k1": ("Controller, $b_{\\max}{=}1$", "AdOne"), "adaptive": ("Controller", "Ad")}


def policy(s) -> str:
    bp = s.get("batch_policy") or "fixed"
    if bp == "adaptive":
        return "adaptive_k1" if int(s.get("kmax", 1000) or 1000) == 1 else "adaptive"
    if bp in ("interval", "drain"):
        return bp
    k = int(s.get("batch_size", 1) or 1)
    return "record" if k == 1 else f"fixed_w{float(s.get('batch_wait_s', 1.0)):g}"


def _depth(backend: str, k: int) -> int:
    return 0 if backend == "besu" else k          # QBFT is final on inclusion


def record_latencies(rid: str, backend: str, k: int) -> pd.DataFrame:
    """Per committed record: arrival time, inclusion and k-finality latencies (NaN: k-th block not in the log),
    and the censoring time (last logged canonical block - arrival)."""
    it = pd.read_parquet(RAW / rid / "items.parquet",
                         columns=["status", "block", "t_sched", "t_send", "e2e_latency_s", "commit_latency_s"])
    ok = it[it["status"] == "committed"].copy()
    if (RAW / rid / "blocks.csv").exists():
        bl = pd.read_csv(RAW / rid / "blocks.csv", usecols=["number", "t_observed", "canonical"])
        seen = bl[bl["canonical"]].groupby("number")["t_observed"].min()
        ok["final"] = (ok["block"] + _depth(backend, k)).map(seen) - ok["t_sched"]
        ok["censor"] = seen.max() - ok["t_sched"]
    else:
        ok["final"], ok["censor"] = ok["e2e_latency_s"], np.nan
    return ok


def block_bootstrap(t: np.ndarray, x: np.ndarray, q: float, block_s: float = 10.0, boot: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """95% interval of the q-th percentile, resampling 10-s windows of arrivals (records within a burst are
    dependent, so resampling single records would understate the uncertainty)."""
    ok = ~np.isnan(x)
    t, x = t[ok], x[ok]
    if len(x) < 20:
        return (float("nan"), float("nan"))
    lab = np.floor((t - t.min()) / block_s).astype(int)
    groups = [x[lab == g] for g in np.unique(lab)]
    rng = np.random.default_rng(seed)
    est = [np.percentile(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), q)
           for _ in range(boot)]
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


# ---------------------------------------------------------------- burstiness of the replayed windows
def trace_stats(df: pd.DataFrame) -> dict:
    """Intensity profile of the 24-h windows replayed in E1 (full, unthinned window around the sampled records):
    peak minute against the mean minute, share of transactions between 08:00 and 20:00, time compression."""
    from finverity import datasets
    from .report import f
    e = df[(df["workload"] == "E11") & (df["backend"] == "poa2_local") & (df["batch_size"] == 1)
           & (df["batch_policy"] != "adaptive")]
    m = {}
    for src, key in (("saml_d", "Saml"), ("ibm_aml", "Ibm")):
        g = e[e["source"] == src]
        if not len(g):
            continue
        full = pd.read_parquet(datasets.processed_path(src), columns=["tx_id", "timestamp"])
        full["ts"] = pd.to_datetime(full["timestamp"])
        ts = full.set_index("tx_id")["ts"]
        peak, day, comp = [], [], []
        for rid in g["run_id"]:
            it = pd.read_parquet(RAW / rid / "items.parquet", columns=["tx_id", "t_sched"])
            t0 = ts.reindex(it["tx_id"]).min()
            w = full[(full["ts"] >= t0) & (full["ts"] < t0 + pd.Timedelta(hours=24))]
            per_min = w.set_index("ts").resample("1min").size()
            peak.append(per_min.max() / (len(w) / 1440))
            h = w["ts"].dt.hour
            day.append(float(((h >= 8) & (h < 20)).mean()))
            comp.append(86400 / it["t_sched"].max())
        m.update({f"Tr{key}PeakMax": f(max(peak), 1), f"Tr{key}PeakMean": f(float(np.mean(peak)), 1),
                  f"Tr{key}DayShare": f(100 * float(np.mean(day)), 0)})
        m["TrCompress"] = f(float(np.mean(comp)), 0)
    return m


# ---------------------------------------------------------------- E13
def e13_frame(df: pd.DataFrame, k: int) -> pd.DataFrame:
    e = df[df["workload"] == "E13"].copy()
    rows = []
    for _, s in e.iterrows():
        r = record_latencies(s["run_id"], s["backend"], k)
        inc, fin, t = r["e2e_latency_s"].to_numpy(float), r["final"].to_numpy(float), r["t_sched"].to_numpy(float)
        lo, hi = block_bootstrap(t, inc, 95)
        flo, fhi = block_bootstrap(t, fin, 95)
        pct = lambda x, q: float(np.nanpercentile(x, q)) if np.isfinite(x).any() else float("nan")
        rows.append({"backend": s["backend"], "rate": int(s["rate"]), "pol": policy(s), "n": int(s["n"]),
                     "success": float(s["success_rate"]), "rpt": float(s["records_per_submission"]),
                     "gas": float(s["gas_used_mean"]) if s["backend"] != "db_postgres" else float("nan"),
                     "inc50": pct(inc, 50), "inc95": pct(inc, 95), "inc99": pct(inc, 99), "inc95lo": lo, "inc95hi": hi,
                     "fin50": pct(fin, 50), "fin95": pct(fin, 95), "fin99": pct(fin, 99), "fin95lo": flo, "fin95hi": fhi,
                     "final_missing": float(np.isnan(fin).mean()), "reorged": s.get("reorged_blocks"),
                     "max_reorg": s.get("max_reorg_depth")})
    return pd.DataFrame(rows)


def t_e13(df: pd.DataFrame, k: int) -> dict:
    from .report import GEN, NAME, f, table
    d = e13_frame(df, k)
    if not len(d):
        return {}
    d.to_csv(RESULTS / "tables" / "e13_sweep.csv", index=False)
    order = list(POLICIES)
    rows, m, last = [], {}, None
    for b in ("poa2_local", "besu", "purechain_local"):
        for rate in sorted(d.loc[d["backend"] == b, "rate"].unique()):
            for pol in order:
                g = d[(d["backend"] == b) & (d["rate"] == rate) & (d["pol"] == pol)]
                if not len(g):
                    continue
                r = g.iloc[0]
                rows.append([NAME[b] if last != b else "", f(rate, 0), POLICIES[pol][0], f(r["success"], 1, True),
                             f(r["rpt"], 1), f(r["inc50"]), f"{f(r['inc95'])} [{f(r['inc95lo'])}, {f(r['inc95hi'])}]",
                             f(r["inc99"]), f(r["fin50"]), f"{f(r['fin95'])} [{f(r['fin95lo'])}, {f(r['fin95hi'])}]",
                             f(r["fin99"]), f(r["gas"], 0)])
                last = b
                key = f"Sw{BKEY[b]}{RATE_WORD.get(rate, str(rate))}{POLICIES[pol][1]}"
                m.update({f"{key}Inc": f(r["inc50"]), f"{key}IncP": f(r["inc95"]), f"{key}IncPP": f(r["inc99"]),
                          f"{key}Fin": f(r["fin50"]), f"{key}FinP": f(r["fin95"]), f"{key}FinPP": f(r["fin99"]),
                          f"{key}Gas": f(r["gas"], 0), f"{key}Rpt": f(r["rpt"], 1), f"{key}Succ": f(r["success"], 1, True)})
    table(GEN / "t_e13.tex", "Submission-policy baselines and utilisation sweep (E2): the IBM AML batch-posting trace "
          "replayed for about 90\\,s at each mean rate, one trial per cell, every policy on the same records. "
          "Per-record arrival-to-inclusion and arrival-to-finality percentiles (finality: "
          f"{k} confirmations; QBFT: inclusion); brackets: 95\\% block-bootstrap interval of the P95 over 10-s "
          "arrival windows. Rec./tx: records per transaction. Gas/record includes the per-identifier uniqueness "
          "check of the registry.", "tab:e13",
          ["Ledger", "Rate (rec/s)", "Policy", "Success (\\%)", "Rec./tx", "Incl. P50", "Incl. P95", "Incl. P99",
           "Final P50", "Final P95", "Final P99", "Gas/record"], rows, "lllrrrcrrcrr", size="\\scriptsize")
    # compact view for the main text: every policy at 30 records/s on the three local ledgers
    rows = []
    for pol in order:
        cells = [POLICIES[pol][0]]
        for b in ("poa2_local", "besu", "purechain_local"):
            g = d[(d["backend"] == b) & (d["rate"] == 30) & (d["pol"] == pol)]
            cells += ([f(g.iloc[0]["inc95"]), f(g.iloc[0]["fin95"]), f(g.iloc[0]["gas"], 0)] if len(g) else ["--"] * 3)
        if any(c != "--" for c in cells[1:]):
            rows.append(cells)
    table(GEN / "t_e13_main.tex", "Submission policies at a mean of 30 records/s under the IBM AML batch-posting trace "
          f"(E2): 95th-percentile arrival-to-inclusion and arrival-to-finality latency (s; finality {k} "
          "confirmations, QBFT on inclusion) and gas per record. The utilisation sweep and P50/P99 are in "
          "Supplementary Table~S2.", "tab:e13main",
          ["Policy", "\\makecell{Clique\\\\Incl.}", "\\makecell{Clique\\\\Final}", "\\makecell{Clique\\\\Gas}",
           "\\makecell{Besu\\\\Incl.}", "\\makecell{Besu\\\\Final}", "\\makecell{Besu\\\\Gas}",
           "\\makecell{PureChain\\\\client Incl.}", "\\makecell{PureChain\\\\client Final}",
           "\\makecell{PureChain\\\\client Gas}"], rows, "lrrrrrrrrr", size="\\scriptsize", wide=True)
    # where does the controller stop being the fastest? the largest rate at which per-record anchoring kept up
    emu = d[d["backend"] == "poa2_local"]
    rec, ad = emu[emu["pol"] == "record"].set_index("rate"), emu[emu["pol"] == "adaptive"].set_index("rate")
    both = rec.join(ad, lsuffix="_r", rsuffix="_a", how="inner")
    if len(both):
        m["SwEmuRecWorseFrom"] = f(int(both.index[both["inc95_r"] > both["inc95_a"]].min()), 0) \
            if (both["inc95_r"] > both["inc95_a"]).any() else "--"
    return m


def fig_pareto(df: pd.DataFrame, k: int) -> None:
    from . import figures as F
    plt = F.plt
    path = RESULTS / "tables" / "e13_sweep.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    style = {"record": ("#0b0b0b", "o", "-"), "fixed_w1": ("#eb6834", "s", "--"), "fixed_w0.25": ("#eda100", "v", "--"),
             "fixed_w0.1": ("#b07d00", "<", ":"), "fixed_w0.5": ("#c9562a", ">", ":"),
             "interval": ("#1baf7a", "^", "-."), "drain": ("#7da7dc", "D", (0, (5, 2, 1, 2))),
             "adaptive_k1": ("#9a9a9a", "X", ":"), "adaptive": ("#2a78d6", "P", "-")}
    lab = {"record": "Per record", "fixed_w1": "Fixed b=50, w=1 s", "fixed_w0.25": "Fixed b=50, w=0.25 s",
           "fixed_w0.1": "Fixed b=50, w=0.1 s", "fixed_w0.5": "Fixed b=50, w=0.5 s", "interval": "Interval 1 s",
           "drain": "Drain per block", "adaptive_k1": "Controller, b_max=1", "adaptive": "Controller"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    emu = d[d["backend"] == "poa2_local"]
    for pol in POLICIES:
        g = emu[emu["pol"] == pol].sort_values("rate")
        if len(g) < 2 or pol == "adaptive_k1":      # b_max = 1 (minutes) would flatten every other curve
            continue
        c, mk, ls = style[pol]
        axs[0].plot(g["rate"], g["inc95"], color=c, marker=mk, linestyle=ls, markeredgecolor="white", label=lab[pol])
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xticks(sorted(emu["rate"].unique())); axs[0].set_xticklabels([str(x) for x in sorted(emu["rate"].unique())])
    axs[0].yaxis.set_major_formatter(F.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].yaxis.set_minor_formatter(F.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].set_xlabel("Mean arrival rate (records/s, log)")
    axs[0].set_ylabel("Arrival → inclusion, P95 (s, log)")
    axs[0].set_title("(a) Clique emulation: utilisation sweep", loc="left")
    thirty = d[d["rate"] == 30]
    for b, mfill in (("poa2_local", "full"), ("besu", "none"), ("purechain_local", "left")):
        for pol in POLICIES:
            g = thirty[(thirty["backend"] == b) & (thirty["pol"] == pol)]
            if not len(g) or not np.isfinite(g["gas"].iloc[0]):
                continue
            c, mk, _ = style[pol]
            axs[1].scatter(g["gas"], g["fin95"], color=c if mfill != "none" else "white", edgecolors=c, marker=mk,
                           s=30, linewidths=1.1, zorder=3)
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].yaxis.set_major_formatter(F.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axs[1].set_xticks([30_000, 50_000, 100_000, 150_000])
    axs[1].xaxis.set_major_formatter(F.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    axs[1].xaxis.set_minor_formatter(F.matplotlib.ticker.NullFormatter())
    axs[1].set_xlabel("Gas per record (log)")
    axs[1].set_ylabel("Arrival → finality, P95 (s, log)")
    axs[1].set_title("(b) 30 records/s: cost against finality", loc="left")
    from matplotlib.lines import Line2D
    ledg = [Line2D([], [], marker="o", color="#52514e", linestyle="", markerfacecolor=fc, label=t)
            for fc, t in (("#52514e", "Clique emulation"), ("white", "Besu QBFT"))]
    ledg.append(Line2D([], [], marker="o", color="#52514e", linestyle="", fillstyle="left", label="PureChain client (local)"))
    axs[1].legend(handles=ledg, loc="upper left", fontsize=6.3, frameon=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16), fontsize=6.5, frameon=False)
    F._save(fig, "F16_pareto")


# ---------------------------------------------------------------- E14
def finality_rows(e: pd.DataFrame, labels: dict, k: int) -> tuple[list, dict]:
    from .report import f, span
    rows, m = [], {}
    for (b, tag, tail), (name, sealing, load, key) in labels.items():
        g = e[(e["backend"] == b) & (e["tag"].fillna("") == tag) & (e["tail_k"].fillna(0).astype(int) == tail)]
        if not len(g):
            continue
        share, last_final, tails, txs, ts = [], [], [], [], []
        for _, s in g.iterrows():
            r = record_latencies(s["run_id"], b, k)
            share.append(float(r["final"].notna().mean()))
            last = r.loc[r["t_sched"].idxmax()]
            last_final.append(last["final"])
            bl = pd.read_csv(RAW / s["run_id"] / "blocks.csv", usecols=["number", "canonical"])
            tails.append(int((bl.loc[bl["canonical"], "number"].drop_duplicates() > r["block"].max()).sum()))
            txs.append(float(s.get("tail_txs") or 0)); ts.append(float(s.get("tail_s") or 0))
        lf = pd.Series(last_final, dtype=float)
        rows.append([name, sealing, load, str(len(g)), f(100 * float(np.mean(share)), 1),
                     f(lf.median()) if lf.notna().any() else "never", f"{lf.notna().sum()}/{len(lf)}",
                     span(min(tails), max(tails)), f(float(np.mean(txs)), 1) if tail else "--"])
        m.update({f"Tl{key}Final": f(100 * float(np.mean(share)), 1),
                  f"Tl{key}Last": f(lf.median()) if lf.notna().any() else "never",
                  f"Tl{key}Trials": f"{lf.notna().sum()}/{len(lf)}", f"Tl{key}TailMin": str(min(tails)),
                  f"Tl{key}TailMax": str(max(tails)), f"Tl{key}TailRange": span(min(tails), max(tails))})
        if tail:
            m.update({f"Tl{key}Txs": f(float(np.mean(txs)), 1), f"Tl{key}Secs": f(float(np.mean(ts)), 1),
                      f"Tl{key}Gas": f(21_000 * float(np.mean(txs)), 0)})
    return rows, m


def t_e14(df: pd.DataFrame, k: int) -> dict:
    from .report import GEN, table
    e = df[df["workload"] == "E14"]
    if not len(e):
        return {}
    labels = {("poa2_local", "p0", 0): ("Clique, period 0", "on demand, no tail", "burst", "PzBurst"),
              ("poa2_local", "p0_tail", 8): ("Clique, period 0", "on demand + client tail", "burst", "PzBurstTail"),
              ("poa2_local", "p0_lone", 0): ("Clique, period 0", "on demand, no tail", "lone", "PzLone"),
              ("poa2_local", "p0_lone_tail", 8): ("Clique, period 0", "on demand + client tail", "lone", "PzLoneTail"),
              ("purechain_local", "", 0): ("PureChain client (local)", "Smart Auto-Mining v2", "burst", "PclBurst"),
              ("purechain_local", "lone", 0): ("PureChain client (local)", "Smart Auto-Mining v2", "lone", "PclLone"),
              ("purechain_local", "outage", 0): ("PureChain client (local)", "SAM v2, 1 of 4 down", "burst",
                                                 "PclOutage"),
              ("purechain", "live_lone_tail", 8): ("PureChain (live)", "SAM v2 + client tail", "lone", "LiveLoneTail")}
    rows, m = finality_rows(e, labels, k)
    table(GEN / "t_e14.tex", "Finality when traffic stops, with the client-side tail and on the released PureChain "
          "client run locally (E4). Burst: 300 records at 50 records/s; lone: one record after 60\\,s of idleness. "
          f"Records final: share with {k} confirmations within the window; last record final: median time from the "
          "last arrival; tail: blocks after the last record's block (observed up to 20); tail txs: no-op "
          "transactions the client sent (21{,}000 gas each).", "tab:e14",
          ["Ledger", "Sealing", "Load", "$n$", "Records final (\\%)", "Last record final (s)", "Trials final",
           "Tail (blocks)", "Tail txs"], rows, "lllrrrrrr", size="\\scriptsize")
    return m


# ---------------------------------------------------------------- inclusion delay D and the bound w + 2D
def t_delay(df: pd.DataFrame) -> dict:
    """D: submission -> first observation of the including block, for controller trials (E11, E13)."""
    from .report import GEN, NAME, client_generation, f, table
    sel = df[df["workload"].isin(["E11", "E13"]) & (df["batch_policy"] == "adaptive")
             & (df["kmax"].fillna(1000).astype(int) != 1)]
    sel = sel[(sel["backend"] != "purechain") | (sel["client"].map(client_generation) == "new")]
    rows, m = [], {}
    for b in ("purechain", "poa2_local", "besu", "purechain_local"):
        g = sel[sel["backend"] == b]
        if not len(g):
            continue
        fr = [pd.read_parquet(RAW / r / "items.parquet", columns=["status", "commit_latency_s", "e2e_latency_s"])
              for r in g["run_id"]]
        it = pd.concat(fr)
        it = it[it["status"] == "committed"]
        D, L = it["commit_latency_s"].dropna().to_numpy(), it["e2e_latency_s"].dropna().to_numpy()
        d99, dmax = np.percentile(D, 99), D.max()
        over99, overmax = float((L > W + 2 * d99).mean()), float((L > W + 2 * dmax).mean())
        rows.append([NAME[b], str(len(g)), f(len(D), 0), f(np.median(D)), f(np.percentile(D, 95)), f(d99), f(dmax),
                     f(W + 2 * d99), f(100 * over99, 2), f(W + 2 * dmax), f(100 * overmax, 2)])
        k = BKEY[b]
        m.update({f"Dl{k}Med": f(np.median(D)), f"Dl{k}Pnn": f(d99), f"Dl{k}Max": f(dmax),
                  f"Dl{k}Over": f(100 * over99, 2), f"Dl{k}OverMax": f(100 * overmax, 2), f"Dl{k}Recs": f(len(D), 0)})
    if rows:
        table(GEN / "t_delay.tex", "Inclusion delay $D$ (submission to first observation of the including block) "
              "under the controller (E1, E2), and the share of records whose arrival-to-inclusion latency exceeded "
              f"the bound $w+2D$ of \\Cref{{prop:bound}} ($w={W:g}$\\,s), with $D$ set to the observed P99 or "
              "maximum of the same ledger.", "tab:delay",
              ["Ledger", "Trials", "Records", "$D$ P50", "$D$ P95", "$D$ P99", "$D$ max", "$w{+}2D_{99}$",
               "Beyond (\\%)", "$w{+}2D_{\\max}$", "Beyond (\\%)"], rows, "lrrrrrrrrrr", size="\\scriptsize")
    return m


# ---------------------------------------------------------------- reorganisation-depth risk
def reorg_events() -> dict[str, dict]:
    """Per-event reorganisation depths (runs of consecutive replaced heights) on live PureChain per client."""
    from .report import client_generation
    out = {"old": {"depths": [], "heights": 0}, "new": {"depths": [], "heights": 0}}
    seen = set()
    for sp in [*RAW.glob("*/summary.json"), *(RESULTS / "_archive_v0" / "raw").glob("*/summary.json"),
               *(RESULTS / "_archive_live_v1").glob("*/summary.json")]:
        sm = json.loads(sp.read_text())
        bf = sp.parent / "blocks.csv"
        if sm.get("backend") != "purechain" or not bf.exists() or bf.stat().st_size <= 200 or sp.parent.name in seen:
            continue
        seen.add(sp.parent.name)
        b = pd.read_csv(bf)
        if "parent" not in b:
            continue
        if "canonical" not in b:
            parent = dict(zip(b["hash"], b["parent"]))
            h, canon_h = b.sort_values(["number", "t_observed"]).iloc[-1]["hash"], set()
            while h in parent:
                canon_h.add(h)
                h = parent[h]
            b["canonical"] = b["hash"].isin(canon_h)
        gen = client_generation(sm.get("client"))
        canon = set(b.loc[b["canonical"], "number"])
        replaced = sorted(set(b.loc[~b["canonical"], "number"]) & canon)
        out[gen]["heights"] += len(canon)
        run, prev = 0, None
        for h in replaced:
            if prev is not None and h == prev + 1:
                run += 1
            else:
                if run:
                    out[gen]["depths"].append(run)
                run = 1
            prev = h
        if run:
            out[gen]["depths"].append(run)
    return out


def sci(x: float) -> str:
    """LaTeX scientific notation, e.g. $7.5\\times10^{-6}$."""
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def t_reorg(k: int = 8) -> dict:
    """Threshold model: depth-1 events at their observed share, depths >= 2 geometric,
    P(D >= d | D >= 2) = q^(d-2); a generalized Pareto fit of the same excesses checks the tail shape."""
    from scipy import stats as st
    from .report import GEN, f, table
    ev = reorg_events()
    m, rows = {}, []
    for gen, key, name in (("new", "New", "Release 1.0.1"), ("old", "Old", "Sept. 2026 client")):
        dep, H = np.array(ev[gen]["depths"]), ev[gen]["heights"]
        if not H or len(dep) < 10:
            continue
        N, r, dmax = len(dep), len(dep) / H, int(dep.max())
        tail = dep[dep >= 2]
        pt = len(tail) / N
        q = 1 - 1 / np.mean(tail - 1)
        surv = lambda d: 1.0 if d <= 1 else pt * q ** (d - 2)          # P(D >= d) per event

        def k_for(eps, r=r, surv=surv):
            kk = 1
            while r * surv(kk + 1) > eps and kk < 200:    # a record with kk confirmations is lost by depth >= kk+1
                kk += 1
            return kk

        # parametric-free bootstrap of the model: Poisson event count, events resampled with replacement
        rng = np.random.default_rng(1)
        k_bs, risk_bs = [], []
        for _ in range(2000):
            nb = rng.poisson(N)
            db = rng.choice(dep, nb)
            tb = db[db >= 2]
            if nb == 0 or len(tb) < 2:
                continue
            rb, ptb, qb = nb / H, len(tb) / nb, 1 - 1 / np.mean(tb - 1)
            sb = lambda d, ptb=ptb, qb=qb: 1.0 if d <= 1 else ptb * qb ** (d - 2)
            k_bs.append(k_for(1e-6, rb, sb))
            risk_bs.append(rb * sb(k + 1))
        m.update({f"Rg{key}KSixLo": str(int(np.percentile(k_bs, 2.5))),
                  f"Rg{key}KSixHi": str(int(np.ceil(np.percentile(k_bs, 97.5)))),
                  f"Rg{key}RiskKLo": sci(float(np.percentile(risk_bs, 2.5))),
                  f"Rg{key}RiskKHi": sci(float(np.percentile(risk_bs, 97.5)))})
        m.update({f"Rg{key}Events": f(N, 0), f"Rg{key}Heights": f(H, 0), f"Rg{key}Rate": f(100 * r, 2),
                  f"Rg{key}Max": str(dmax), f"Rg{key}TailShare": f(100 * pt, 1), f"Rg{key}Q": f(q, 2),
                  f"Rg{key}RuleThree": sci(3 / H),          # no depth > dmax at H heights: 95% bound per height
                  f"Rg{key}RiskK": sci(r * surv(k + 1)),    # fitted per-height reversal risk at depth k
                  f"Rg{key}KSix": str(k_for(1e-6)), f"Rg{key}KNine": str(k_for(1e-9))})
        if len(tail) >= 30:
            x = tail - 1.5                                # continuity-corrected excesses over the threshold
            xi = st.genpareto.fit(x, floc=0)[0]
            rng = np.random.default_rng(0)
            bs = [st.genpareto.fit(rng.choice(x, len(x)), floc=0)[0] for _ in range(300)]
            m.update({f"Rg{key}Xi": f(xi, 2), f"Rg{key}XiLo": f(float(np.percentile(bs, 2.5)), 2),
                      f"Rg{key}XiHi": f(float(np.percentile(bs, 97.5)), 2)})
        for d in range(1, dmax + 2):
            rows.append([name if d == 1 else "", str(d) if d <= dmax else f"$\\geq${d}", f(int((dep >= d).sum()), 0),
                         f(N * surv(d), 1 if d <= dmax else 2), f"{100 * r * surv(d):.2g}"])
    if rows:
        table(GEN / "t_reorg.tex", "Reorganisation events on live PureChain (runs of consecutive replaced heights): "
              "events of depth at least $d$, against a threshold model (depth-1 events at their observed share, "
              "geometric tail $P(D\\geq d\\mid D\\geq2)=q^{d-2}$), and the fitted share of heights at which a "
              "reorganisation of depth $\\geq d$ starts. The September 2026 client kept parent hashes in few "
              "block logs.", "tab:reorg",
              ["Client", "Depth $d$", "Events $\\geq d$", "Fitted", "Heights (\\%)"], rows, "lrrrr", size="\\scriptsize")
    return m


# ---------------------------------------------------------------- censored finality (Kaplan-Meier)
def km_quantile(t: np.ndarray, event: np.ndarray, q: float) -> float:
    order = np.argsort(t, kind="stable")
    t, event = t[order], event[order]
    n, s = len(t), 1.0
    for i, (ti, ei) in enumerate(zip(t, event)):
        if ei:
            s *= 1 - 1 / (n - i)
            if 1 - s >= q:
                return float(ti)
    return float("nan")


def km_finality(df: pd.DataFrame, k: int) -> dict:
    from .report import f
    fk = pd.read_csv(RESULTS / "tables" / f"finality_k{k}.csv")
    aff = fk[fk["final_k_missing"] > 0]["run_id"]
    sub = df[df["run_id"].isin(aff)]
    diffs, diffs95, n_cens, n_all = [], [], 0, 0
    for _, s in sub.iterrows():
        r = record_latencies(s["run_id"], s["backend"], k)
        ev = r["final"].notna().to_numpy()
        t = np.where(ev, r["final"].to_numpy(float), r["censor"].to_numpy(float))
        n_cens += int((~ev).sum()); n_all += len(r)
        cc50, cc95 = np.nanpercentile(r["final"], 50), np.nanpercentile(r["final"], 95)
        km50, km95 = km_quantile(t, ev, 0.5), km_quantile(t, ev, 0.95)
        diffs.append(km50 - cc50); diffs95.append(km95 - cc95)
    if not diffs:
        return {}
    d, d95 = np.abs(np.array(diffs)), np.abs(np.array(diffs95, dtype=float))
    return {"KmTrials": f(len(diffs), 0), "KmCensored": f(n_cens, 0), "KmMedDiffMax": f(float(np.nanmax(d)), 2),
            "KmMedDiffMean": f(float(np.nanmean(d)), 3), "KmPDiffMax": f(float(np.nanmax(d95)), 2)}


# ---------------------------------------------------------------- coalitions, outage, gas
def t_coalition() -> dict:
    from .report import GEN, f, table
    m, rows = {}, []
    p = RESULTS / "coalition.json"
    if p.exists():
        for r in json.loads(p.read_text()):
            c = r["coalition"]
            word = {1: "One", 2: "Two", 3: "Three"}[c]
            rows.append([f"{c} of {r['signers']}", "yes" if c >= r["clique_limit"] else "no", f(r["blocks_sealed_alone"], 0),
                         f(r["stalled_after_s"], 0) if r["stalled_after_s"] is not None else "never",
                         "yes" if r["chain_resumed_after_heal"] else "no"])
            m[f"Coal{word}Blocks"] = f(r["blocks_sealed_alone"], 0)
            m[f"Coal{word}Stall"] = f(r["stalled_after_s"], 0) if r["stalled_after_s"] is not None else "never"
            m["CoalWindow"] = f(r["window_s"], 0)
        table(GEN / "t_coalition.tex", "Signer coalitions on the 4-signer Clique emulation: the coalition's signers "
              "keep sealing while the honest signers are cut off for "
              f"{m.get('CoalWindow', '60')}\\,s (Clique lets a signer seal one of any 3 consecutive blocks).",
              "tab:coalition", ["Coalition", "Reaches limit", "Blocks sealed alone", "Stalled after (s)",
                                "Resumed after heal"], rows, "lcrrc", size="\\scriptsize")
    return m


def t_gas(df: pd.DataFrame) -> dict:
    """Gas per record by batch size: first registry version, revised with stored IDs (rejects re-commitment) and
    revised with logged IDs (the auditor detects re-commitment from the complete event log)."""
    from .report import GEN, f, table
    e9 = df[(df["workload"] == "E9") & (df["backend"] == "poa2_local")]
    v2 = e9["tag"].fillna("") == "v2"
    ids = e9["batch_ids"].fillna("store") if "batch_ids" in e9 else pd.Series("store", index=e9.index)
    # the revised registry was measured at 200 records/s, where batches fill; compare the first version at the same rate
    old, new, logd = e9[~v2 & (e9["rate"] == 200)], e9[v2 & (ids == "store")], e9[v2 & (ids == "log")]
    rows, m = [], {}
    word = {1: "One", 10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred"}
    gas = lambda g, k: g.loc[g["batch_size"] == k, "gas_used_mean"].astype(float).mean()
    for k in (1, 10, 25, 50, 100):
        a, b, c = gas(old, k), gas(new, k), gas(logd, k)
        if b != b:
            continue
        rows.append([str(k), f(a, 0), f(b, 0), f(c, 0) if c == c else "--"])
        m[f"GasNew{word[k]}"], m[f"GasOld{word[k]}"] = f(b, 0), f(a, 0)
        if c == c:
            m[f"GasLog{word[k]}"] = f(c, 0)
            m[f"GasLogSave{word[k]}"] = f(b / c, 1)
    if rows:
        table(GEN / "t_gas.tex", "Gas per record by batch size $b$ (Clique emulation, B7, 200 records/s): a registry that does not check "
              "identifiers inside a batch; the registry storing every batched identifier once, which rejects "
              "re-commitment; and the registry logging the identifiers in an event, with which the auditor detects "
              "re-commitment from the complete log instead.", "tab:gas",
              ["$b$", "Unchecked IDs", "Stored IDs (prevent)", "Logged IDs (detect)"], rows, "rrrr",
              size="\\scriptsize")
    return m


def orphan_turn() -> dict:
    """Share of replaced (orphaned) live blocks that an out-of-turn signer sealed (difficulty 1), per client."""
    from .report import client_generation, f
    tot = {"old": [0, 0], "new": [0, 0]}
    for sp in [*RAW.glob("*/summary.json"), *(RESULTS / "_archive_v0" / "raw").glob("*/summary.json"),
               *(RESULTS / "_archive_live_v1").glob("*/summary.json")]:
        s = json.loads(sp.read_text())
        if s.get("backend") == "purechain" and s.get("orphan_blocks"):
            g = client_generation(s.get("client"))
            tot[g][0] += int(s["orphan_blocks"])
            tot[g][1] += int(s.get("orphans_out_of_turn") or 0)
    return {f"Orphan{k.capitalize()}OotPct": f(100 * v[1] / v[0], 0) for k, v in tot.items() if v[0]} | \
        {f"Orphan{k.capitalize()}Blocks": f(v[0], 0) for k, v in tot.items() if v[0]}


# ---------------------------------------------------------------- E16 network delay between validators
ND_WORD = {0: "Zero", 50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred", 800: "EightHundred"}


def _events(b: pd.DataFrame) -> tuple[int, list[int], pd.DataFrame]:
    """Canonical heights, reorganisation-event depths (runs of consecutive replaced heights) and orphaned versions."""
    canon = set(b.loc[b["canonical"], "number"])
    orph = b[~b["canonical"] & b["number"].isin(canon)]
    depths, run, prev = [], 0, None
    for h in sorted(set(orph["number"])):
        if prev is not None and h == prev + 1:
            run += 1
        else:
            if run:
                depths.append(run)
            run = 1
        prev = h
    if run:
        depths.append(run)
    return len(canon), depths, orph


def _k_model(dep: np.ndarray, H: int, eps: float = 1e-6) -> int | None:
    """Confirmations for per-height risk eps under the threshold model of t_reorg (needs >= 2 events of depth >= 2)."""
    tail = dep[dep >= 2]
    if len(dep) < 10 or len(tail) < 2:
        return None
    r, pt, q = len(dep) / H, len(tail) / len(dep), 1 - 1 / np.mean(tail - 1)
    surv = lambda d: 1.0 if d <= 1 else pt * q ** (d - 2)
    kk = 1
    while r * surv(kk + 1) > eps and kk < 200:
        kk += 1
    return kk


def t_netdelay(k: int = 8) -> dict:
    """E16: forks, out-of-turn sealing and finality of the local PureChain client against one-way delay d between
    validators (tc netem, normal jitter 10 % of d), per delay level over all trials."""
    from .report import GEN, f, table
    runs = sorted(RAW.glob("E16_purechain_local_*_nd*/summary.json"))
    if not runs:
        return {}
    per = {}
    for sp in runs:
        rid = sp.parent.name
        net = json.loads((sp.parent / "netem.json").read_text()) if (sp.parent / "netem.json").exists() else {}
        d = int(float(rid.rsplit("_nd", 1)[1]))
        b = pd.read_csv(sp.parent / "blocks.csv")
        H, dep, orph = _events(b)
        lat = record_latencies(rid, "purechain_local", k)
        g = per.setdefault(d, {"trials": 0, "H": 0, "dep": [], "orph": 0, "orph_oot": 0, "canon_oot": 0,
                               "canon": 0, "incl": [], "final": [], "missing": 0, "rec": 0, "rtt": [], "stalled": 0, "tps": [], "gap": []})
        canon = b[b["canonical"]].drop_duplicates("number")
        g["trials"] += 1
        sm = json.loads(sp.read_text())
        g["stalled"] += int(sm.get("committed", 0) < sm.get("n", 0))   # chain stopped before every record committed
        g["tps"].append(float(sm.get("throughput_tps") or np.nan))
        g["gap"].append(float(canon.sort_values("number")["timestamp"].diff().mean()))
        g["H"] += H
        g["dep"] += dep
        g["orph"] += len(orph)
        g["orph_oot"] += int((orph["difficulty"] == 1).sum())
        g["canon"] += len(canon)
        g["canon_oot"] += int((canon["difficulty"] == 1).sum())
        g["incl"].append(lat["e2e_latency_s"].to_numpy())
        g["final"].append(lat["final"].to_numpy())
        g["missing"] += int(lat["final"].isna().sum())
        g["rec"] += len(lat)
        pairs = [v for kk, v in net.get("rtt_ms", {}).items() if kk != "1-gateway" and v is not None]
        if pairs:
            g["rtt"].append(float(np.mean(pairs)))
    rows, m = [], {"NdTrials": f(sum(g["trials"] for g in per.values()), 0), "NdLevels": f(len(per), 0)}
    for d in sorted(per):
        g = per[d]
        dep = np.array(g["dep"], dtype=float)
        incl, fin = np.concatenate(g["incl"]), np.concatenate(g["final"])
        fin = fin[np.isfinite(fin)]
        kobs = int(dep.max()) + 1 if len(dep) else 1          # confirmations above every observed reorganisation
        kmod = _k_model(dep, g["H"])
        rep = sum(dep) / g["H"] if g["H"] else np.nan
        w = ND_WORD.get(d, f"D{d}")
        vals = {"Rtt": f(np.mean(g["rtt"]), 0) if g["rtt"] else "--", "Trials": f(g["trials"], 0),
                "Heights": f(g["H"], 0), "Stalled": f(g["stalled"], 0), "TpsMin": f(float(np.nanmin(g["tps"])), 1), "BlockGap": f(float(np.mean(g["gap"]))), "ReplacedPct": f(100 * rep, 1), "Events": f(len(dep), 0),
                "MaxDepth": str(int(dep.max())) if len(dep) else "0",
                "MeanDepth": f(float(dep.mean()), 2) if len(dep) else "--",
                "OrphanOotPct": f(100 * g["orph_oot"] / g["orph"], 0) if g["orph"] else "--",
                "OotPct": f(100 * g["canon_oot"] / g["canon"], 1), "KObs": str(kobs),
                "KModel": str(kmod) if kmod else "--",
                "InclMed": f(float(np.median(incl))), "InclTail": f(float(np.percentile(incl, 95))),
                "FinalMed": f(float(np.median(fin))) if len(fin) else "--",
                "FinalTail": f(float(np.percentile(fin, 95))) if len(fin) else "--",
                "FinalMissingPct": f(100 * g["missing"] / g["rec"], 1)}
        m.update({f"Nd{w}{key}": v for key, v in vals.items()})
        rows.append([f"{d:g}", vals["Rtt"], vals["Trials"], vals["Stalled"], vals["TpsMin"], vals["Heights"], vals["ReplacedPct"], vals["Events"],
                     vals["MaxDepth"], vals["MeanDepth"], vals["OotPct"], vals["OrphanOotPct"], vals["KObs"],
                     vals["InclMed"], vals["InclTail"], vals["FinalMed"], vals["FinalTail"]])
    table(GEN / "t_netdelay.tex", "Network-delay emulation (E9): local PureChain network (released client 1.0.1 with Smart Auto-Mining v2, four "
          "signers) under one-way network delay $d$ between validators (tc netem, normal jitter of 10\\% of $d$; "
          "traffic between client and RPC node not delayed), sustained load of 30 records/s, one transaction per "
          "record. RTT: measured validator-to-validator round trip. Stalled: trials in which the chain stopped before "
          "every record was included (validators held different heads). Rec/s min: lowest achieved throughput of a trial; below the "
          "offered load a backlog builds and inclusion latency grows. Replaced: share of canonical heights whose "
          "first-seen block was replaced; events are runs of consecutive replaced heights. Out-of-turn: share of "
          "canonical blocks, and of replaced blocks, sealed out of turn (difficulty 1). $k_{\\mathrm{obs}}$: "
          "confirmations above every observed reorganisation. Latencies from arrival, seconds: inclusion (first "
          f"observation of the including block) and finality at $k={k}$ confirmations, median and P95.",
          "tab:netdelay",
          ["$d$ (ms)", "RTT (ms)", "Trials", "Stalled", "Rec/s min", "Heights", "Replaced (\\%)", "Events", "Max depth", "Mean depth",
           "Out-of-turn (\\%)", "Of replaced (\\%)", "$k_{\\mathrm{obs}}$", "Incl. P50", "Incl. P95",
           f"Final P50", f"Final P95"], rows, "r" * 17, size="\\scriptsize")
    return m


def t_forkcond() -> dict:
    """Compact main-text table: fork rates on the live network with and without our load (E1-E7 trials, E8) and on the
    local client under network delay (E9). Cells are macros, so every number stays generated."""
    from .report import GEN, table
    rows = [["Live, during our trials", "\M{LiveNewBlocksObserved}", "\M{LiveNewReorgPct}", "\M{LiveNewMaxReorg}",
             "\M{OrphanNewOotPct}"],
            ["Live, no traffic of ours (E8)", "\M{PsvHeights}", "\M{PsvReplacedPct}", "\M{PsvMaxDepth}",
             "\M{PsvOrphanOotPct}"]]
    for d in sorted({int(float(x.parent.name.rsplit("_nd", 1)[1])) for x in RAW.glob("E16_purechain_local_*_nd*/summary.json")}):
        w = ND_WORD.get(d, f"D{d}")
        rows.append([f"Local, $d={d}$\,ms (E9)" if d else "Local, no delay (E9)", f"\M{{Nd{w}Heights}}",
                     f"\M{{Nd{w}ReplacedPct}}", f"\M{{Nd{w}MaxDepth}}", f"\M{{Nd{w}OrphanOotPct}}"])
    table(GEN / "t_forkcond.tex", "Forks on PureChain client release 1.0.1 by condition: the live network while our "
          "trials loaded it and while we only watched it, and the released client run locally as four signers at "
          "30 records/s with one-way delay $d$ between validators (Supplementary Table~S18). Replaced: share of "
          "canonical heights whose first-seen block was replaced; out of turn: share of the replaced blocks that a "
          "signer had sealed out of turn.", "tab:forkcond",
          ["Condition", "Heights", "Replaced (\%)", "Max depth", "Out of turn (\%)"], rows, "lrrrr",
          size="\scriptsize", wide=False)
    return {}


def build(df: pd.DataFrame, k: int) -> dict:
    m = orphan_turn()
    for fn in (lambda: t_e13(df, k), lambda: t_e14(df, k), lambda: t_delay(df), lambda: t_reorg(k), lambda: km_finality(df, k),
               t_coalition, lambda: t_gas(df), lambda: trace_stats(df), lambda: t_netdelay(k),
               lambda: passive_build(k), ots_build, t_forkcond):
        try:
            m.update(fn())
        except Exception as e:                      # a missing experiment must not stop the report
            print(f"  revision analysis failed: {type(e).__name__}: {e}")
    try:
        fig_pareto(df, k)
    except Exception as e:
        print(f"  F16 failed: {type(e).__name__}: {e}")
    return m
