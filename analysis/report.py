"""Build the Overleaf-ready experimental report (paper/) from the measured data and compile it to PDF.

    python -m analysis.report            # regenerate tables/figures/macros, compile paper/main.pdf, zip for Overleaf

Every number in the generated tables and in the macros file comes from results/raw; the report can be
rebuilt at any time and fills in as the campaign progresses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.plan import plan
from finverity.config import RESULTS, ROOT
from . import figures, revision, stats
from .stats import RAW, TABLES, clopper_pearson, load_summaries

PAPER = ROOT / "paper"
GEN = PAPER / "generated"
FIGDIR = PAPER / "figures"

NAME = {"purechain": "PureChain (live)", "poa2_local": "Clique emulation", "besu": "Besu QBFT",
        "db_postgres": "PostgreSQL", "ethpos": "Ethereum PoS", "purechain_local": "PureChain client (local)"}
ORDER = ["purechain", "purechain_local", "poa2_local", "besu", "db_postgres", "ethpos"]
EXP_TITLE = {"E1": "Controlled workloads W1--W5", "E2": "Offered-load sweep", "E3": "Evidence-size sweep",
             "E4": "Evidence-complexity sweep", "E5": "Validator-count sweep", "E6": "Tamper detection",
             "E7": "Cross-dataset replication", "E8": "Insider tampering", "E9": "Batched Merkle commitments",
             "E10": "Demand-driven block production", "E11": "Burst-aware anchoring",
             "E12": "Finality after a burst", "E13": "Policy baselines and utilisation sweep",
             "E14": "Client-side tail, local PureChain client", "E8v2": "Insider attacks, revised registry",
             "E6v2": "Tamper detection, widened", "E9v2": "Write-once gas by batch size"}
# experiment numbers used in the manuscript (in order of appearance) for the internal IDs of experiments/plan.py
PAPER_ID = {"E11": "E1", "E13": "E2", "E12": "E3", "E14": "E4", "E10": "E5", "E6": "E6", "E8": "E7",
            "E1": "B1", "E2": "B2", "E3": "B3", "E4": "B4", "E5": "B5", "E7": "B6", "E9": "B7",
            "E8v2": "E7", "E6v2": "E6", "E9v2": "B7"}


def tex(s) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def f(v, nd=2, pct=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if pct:
        return f"{100 * v:.{nd}f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], align: str,
          note: str = "", size: str = "\\small", wide: bool | None = None) -> None:
    """wide: span both columns of a two-column layout (table*); by default tables with six or more columns do."""
    wide = len(header) >= 6 if wide is None else wide
    env, width = ("table*", "\\textwidth") if wide else ("table", "\\linewidth")
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows) if rows else \
        f"\\multicolumn{{{len(header)}}}{{c}}{{\\emph{{pending -- experiment still running}}}} \\\\"
    s = (f"\\begin{{{env}}}[t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n" +
         f"\\begin{{adjustbox}}{{max width={width}}}\n" +
         f"\\begin{{tabular}}{{{align}}}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n" + body +
         "\n\\bottomrule\n\\end{tabular}" + "\\end{adjustbox}" + "\n" + (f"\\par\\smallskip\\footnotesize {note}\n" if note else "") +
         f"\\end{{{env}}}\n")
    path.write_text(s, encoding="utf-8")


def span(lo, hi) -> str:
    """A range a--b, or a single number when both ends are equal."""
    return f"{lo}--{hi}" if lo != hi else str(lo)


def mean_ci(x: pd.Series) -> str:
    x = x.dropna().astype(float)
    if not len(x):
        return "--"
    if len(x) < 2:
        return f(x.mean())
    lo, hi = stats.ci95(x)
    return f"{f(x.mean())} [{f(max(lo, 0.0))}, {f(hi)}]"   # all reported metrics are non-negative


# ------------------------------------------------------------------ individual tables
def t_status(df: pd.DataFrame) -> None:
    P = plan("full")
    rows = []
    for e, cfgs in P.items():
        by = {}
        for c in cfgs:
            by.setdefault(c["backend"], 0)
            by[c["backend"]] += 1
        base = e[:-2] if e.endswith("v2") else e
        done = df[df["workload"].isin([base] if not e == "E1" else ["W1", "W2", "W3", "W4", "W5"])]
        if base in ("E6", "E8", "E9"):
            v2 = done["tag"].fillna("").str.startswith("v2")
            done = done[v2] if e.endswith("v2") else done[~v2]
        cells = []
        for b in ORDER:
            if b in by:
                d = int((done["backend"] == b).sum())
                cells.append(f"{min(d, by[b])}/{by[b]}")
            else:
                cells.append("")
        rows.append([PAPER_ID.get(e, e), tex(e), EXP_TITLE[e], *cells])
    rows.sort(key=lambda r: (r[0][0] != "E", int(r[0][1:]), r[1]))
    table(GEN / "t_status.tex", "Campaign completion status (completed / planned trials) at the time this report "
          "was generated. ID: experiment number in the manuscript; code: identifier in the experiment plan.",
          "tab:status", ["ID", "Code", "Experiment", *[NAME[b] for b in ORDER]], rows, "lll" + "c" * len(ORDER),
          size="\\scriptsize")


def t_tamper() -> dict:
    tt = stats.tamper_table()
    if not len(tt):
        return {}
    runs = {}
    for p in RAW.glob("*/tamper.parquet"):
        b = json.loads((p.parent / "summary.json").read_text())["backend"]
        runs[b] = runs.get(b, 0) + 1
    pooled = "; ".join(f"{NAME[k]}: {v} runs" for k, v in sorted(runs.items(), key=lambda kv: ORDER.index(kv[0])))
    rows = []
    for _, r in tt.iterrows():
        rows.append([tex(r["kind"]), tex(r["modification"]), f(int(r["cases"]), 0), f(int(r["detected"]), 0),
                     f(r["rate_pct"]), f"[{f(r['ci95_lo_pct'], 3)}, {f(r['ci95_hi_pct'], 3)}]"])
    table(GEN / "t_tamper.tex", "Tamper detection against on-chain commitments (E6). Malicious rows report the "
          "detection rate (TDR); benign rows report the false-positive rate (FPR); rejected rows report the share of "
          "non-canonical inputs refused before hashing. Exact Clopper--Pearson 95\\% intervals. "
          f"Pooled over {pooled}.",
          "tab:tamper", ["Kind", "Modification", "Cases", "Flagged", "Rate (\\%)", "95\\% CI"],
          rows, "llrrrc", size="\\scriptsize")
    a = tt[tt["modification"] == "ALL (TDR)"].iloc[0]
    b = tt[tt["modification"] == "ALL (FPR)"].iloc[0]
    out = {"TamperCases": f(int(a["cases"]), 0), "TDR": f(a["rate_pct"]), "TDRlo": f(a["ci95_lo_pct"], 3),
           "FPR": f(b["rate_pct"]), "FPRhi": f(b["ci95_hi_pct"], 3), "BenignCases": f(int(b["cases"]), 0),
           "BenignKinds": str(int((tt["kind"] == "benign").sum()) - 1)}
    rj = tt[tt["modification"] == "ALL (rejected)"]
    if len(rj):
        out.update({"RejectCases": f(int(rj.iloc[0]["cases"]), 0), "RejectRate": f(rj.iloc[0]["rate_pct"]),
                    "RejectKinds": str(int((tt["kind"] == "rejected").sum()) - 1)})
    return out


def t_insider(df: pd.DataFrame) -> dict:
    """Attack x infrastructure matrix: share protected (prevented or detected), its outcome, and the case count."""
    fr = [pd.read_parquet(p).assign(run_id=p.parent.name) for p in RAW.glob("*/insider.parquet")]
    if not fr:
        return {}
    d = pd.concat(fr)
    d["backend"] = d["run_id"].map(df.set_index("run_id")["backend"])
    d = d.dropna(subset=["backend"])
    scen = {"S1_evidence_only": "S1 alter evidence", "S2_commitment_rewrite": "S2 + rewrite commitment",
            "S3_backdated_record": "S3 fabricate + backdate",
            "S4a_duplicate_after_record": "S4a re-commit recorded ID",
            "S4b_duplicate_after_batch": "S4b re-commit batched ID"}
    if "outcome_digest" not in d:
        d["outcome_digest"] = None
    cols = [b for b in ORDER if b in set(d["backend"])]
    mark = {"prevented": "P", "detected": "D"}
    rows, macros, min_lo = [], {}, 1.0
    for sc, label in scen.items():
        cells, dig_cell = [], "--"
        for b in cols:
            g = d[(d["scenario"] == sc) & (d["backend"] == b)]
            if not len(g):
                cells.append("--")
                continue
            n = len(g)
            prot = g["outcome"].isin(["prevented", "detected"])
            how = "/".join(mark[o] for o in ("prevented", "detected") if (g["outcome"] == o).any()) or "U"
            cells.append(f"{100 * prot.mean():.0f} {how} ({n})")
            if prot.all():
                min_lo = min(min_lo, clopper_pearson(n, n)[0])
            key = {"S1": "SOne", "S2": "STwo", "S3": "SThree", "S4a": "SFourA", "S4b": "SFourB"}[sc.split("_")[0]] + \
                {"poa2_local": "Emu", "besu": "Besu", "db_postgres": "DB", "ethpos": "Pos", "purechain": "Live",
                 "purechain_local": "Pcl"}[b]
            macros[f"Ins{key}"] = f"{100 * prot.mean():.1f}"
            macros[f"Ins{key}Cases"] = f(n, 0)
            if b == "db_postgres":
                gd = g[g["outcome_digest"].notna()].copy()
                if len(gd):            # a write the database itself rejected stays prevented under the digest audit
                    gd["outcome_digest"] = gd["outcome_digest"].where(gd["outcome"] != "prevented", "prevented")
                    dp = gd["outcome_digest"].isin(["prevented", "detected"])
                    macros[f"Ins{key}Digest"] = f"{100 * dp.mean():.1f}"
                    how = "/".join(mark[o] for o in ("prevented", "detected") if (gd["outcome_digest"] == o).any()) or "U"
                    dig_cell = f"{100 * dp.mean():.0f} {how} ({len(gd)})"
        rows.append([label, *cells, dig_cell])
    short = {"purechain": "\\makecell{PureChain\\\\(live)}", "purechain_local": "\\makecell{PureChain\\\\client (local)}",
             "poa2_local": "\\makecell{Clique\\\\emulation}", "besu": "\\makecell{Besu\\\\QBFT}",
             "db_postgres": "PostgreSQL", "ethpos": "\\makecell{Ethereum\\\\PoS}"}
    table(GEN / "t_insider.tex", "Privileged-insider tampering (E7): share of attacks protected (\\%), how (P: write "
          "prevented; D: detected by the audit; U: undetected) and cases. S4 re-commits an already recorded identifier "
          "inside a new batch (run only against the registry that stores batched identifiers). PostgreSQL + digest: the same database attacks audited against "
          "a hash-chain digest of the table published before the attack. Every protected share of 100\\% has a "
          f"Clopper--Pearson 95\\% lower bound of at least {100 * min_lo:.0f}\\%.", "tab:insider",
          ["Attack", *[short[b] for b in cols], "\\makecell{PostgreSQL\\\\+ digest}"], rows,
          "l" + "c" * (len(cols) + 1), size="\\scriptsize", wide=True)
    return macros


def t_batching(df: pd.DataFrame) -> None:
    sub = df[(df["workload"] == "E9") & ~df["tag"].fillna("").str.startswith("v2")]
    rows = []
    for (b, rate, k), g in sub.groupby(["backend", "rate", "batch_size"]):
        rows.append([NAME[b], f(rate, 0), f(int(k), 0), str(len(g)), mean_ci(g["throughput_tps"]),
                     mean_ci(g["e2e_latency_s_median"]), mean_ci(g["final_e2e_latency_s_median"]),
                     f(g["gas_used_mean"].mean(), 0) if b != "db_postgres" else "--"])
    rows.sort(key=lambda r: (ORDER.index(next(k for k, v in NAME.items() if v == r[0])), float(r[1].replace(",", "")),
                             int(r[2].replace(",", ""))))
    table(GEN / "t_batching.tex", "Batched Merkle commitments (B7): records per commitment $b$ versus throughput, "
          "end-to-end inclusion latency and finality latency (mean over trials, 95\\% bootstrap CI). Gas: registry "
          "without stored batch identifiers; with them, see the gas table of the main text.", "tab:batching",
          ["Infrastructure", "Offered", "$b$", "$n$", "Throughput (rec/s)", "Arrival$\\rightarrow$incl. P50 (s)",
           "Arrival$\\rightarrow$final P50 (s)", "Gas/record"], rows, "lrrrcccr", size="\\scriptsize")


def _fmt_factor(exp: str, fv) -> str:
    if exp == "E2":
        return "max" if float(fv) == 0 else f"{float(fv):g}"
    if exp == "E3":
        kb = int(fv) / 1024
        return "none" if kb == 0 else (f"{kb:g}\\,KB" if kb < 1024 else f"{kb / 1024:g}\\,MB")
    if exp == "E5":
        tag, rate = fv
        return f"{tex(str(tag).replace('_confirm', ' (confirm)'))} @ {float(rate):g}"
    return tex(fv)


def t_generic(df: pd.DataFrame, exp: str, factor, factor_name: str, caption: str, label: str) -> None:
    sub = df[df["workload"].isin(["W1", "W2", "W3", "W4", "W5"] if exp == "E1" else [exp])]
    rows = []
    keys = factor if isinstance(factor, list) else [factor]
    if len(sub):
        sub = sub.copy()
        if exp == "E2":
            sub["_ord"] = sub["rate"].replace(0, 1e9)
            keys = ["_ord"]
        for key, g in sub.groupby(keys + ["backend"]):
            fv, b = (tuple(key[:-1]) if len(keys) > 1 else key[0]), key[-1]
            if exp == "E2":
                fv = 0 if fv == 1e9 else fv
            rows.append([_fmt_factor(exp, fv), NAME[b], str(len(g)), f(g["success_rate"].mean(), 1, True), mean_ci(g["throughput_tps"]),
                         mean_ci(g["e2e_latency_s_median"]), mean_ci(g["e2e_latency_s_p95"]),
                         mean_ci(g["final_e2e_latency_s_median"]), f(g["energy_j_per_tx"].mean(), 3),
                         f(g["cost_total_usd_per_tx"].mean() * 1e6, 2)])
    table(GEN / f"t_{exp.lower()}.tex", caption, label,
          [factor_name, "Infrastructure", "$n$", "Success (\\%)", "Throughput (rec/s)", "Incl. P50 (s)",
           "Incl. P95 (s)", "Final P50 (s)", "J/record", "\\textmu USD/record"], rows, "llrrcccccr",
          size="\\scriptsize")


def client_generation(client: str) -> str:
    """'old': the September 2026 client (stock Geth build, halted); 'new': PureChain release 1.0.1 and later."""
    return "new" if "purechain-1." in str(client) else "old"


def live_forks() -> dict:
    """Fork statistics over every live-PureChain trial ever run (campaign and ad-hoc, incl. archived),
    overall and per client generation. The finality depth is the smallest depth exceeding every observed
    reorganisation."""
    rows = []
    for sp in [*RAW.glob("*/summary.json"), *(RESULTS / "_archive_v0" / "raw").glob("*/summary.json"),
               *(RESULTS / "_archive_live_v1").glob("*/summary.json")]:
        s = json.loads(sp.read_text())
        if s.get("backend") == "purechain" and s.get("blocks_checked") is not None and s.get("max_reorg_depth") is not None:
            rows.append((s["blocks_checked"], s["reorged_blocks"], s["max_reorg_depth"], client_generation(s.get("client"))))
    if not rows:
        return {}
    m = {}
    for tag, sel in (("", rows), ("Old", [r for r in rows if r[3] == "old"]), ("New", [r for r in rows if r[3] == "new"])):
        if not sel:
            continue
        blocks, reorg = sum(r[0] for r in sel), sum(r[1] for r in sel)
        m.update({f"Live{tag}Runs": f(len(sel), 0), f"Live{tag}BlocksObserved": f(blocks, 0),
                  f"Live{tag}ReorgBlocks": f(reorg, 0), f"Live{tag}ReorgPct": f(100 * reorg / blocks, 1),
                  f"Live{tag}MaxReorg": str(max(r[2] for r in sel))})
    m["FinalityDepth"] = str(int(m["LiveMaxReorg"]) + 1)
    return m


def t_live_campaign(df: pd.DataFrame) -> None:
    """Live PureChain campaign on the current client, one row per experiment condition."""
    live = df[(df["backend"] == "purechain") & (df["client"].map(client_generation) == "new")]
    if not len(live):
        return
    tag_name = {"lone": "lone record", "live_lone_tail": "lone record, client tail", "v2live": "S4 check"}

    def cond(s):
        w, tag = s["workload"], s["tag"] or ""
        if w.startswith("W"):
            return w
        if w == "E2":
            return f"{PAPER_ID[w]} @ {'max' if s['rate'] == 0 else int(s['rate'])}"
        return f"{PAPER_ID.get(w, w)} {tag_name.get(tag, tag)}".strip()
    live = live.assign(cond=live.apply(cond, axis=1))
    rows = []
    for c, g in live.groupby("cond", sort=False):
        blocks = g["blocks_checked"].astype(float).sum()
        rows.append([tex(c), str(len(g)), f(g["success_rate"].mean(), 1, True), mean_ci(g["throughput_tps"]),
                     mean_ci(g["e2e_latency_s_median"]), mean_ci(g["final_e2e_latency_s_median"]),
                     f(100 * g["reorged_blocks"].astype(float).sum() / blocks, 1) if blocks else "--",
                     f(int(g["max_reorg_depth"].max()), 0), f(g["verify_success_rate"].mean(), 1, True)])
    table(GEN / "t_live.tex", "Live PureChain campaign (PureChain release 1.0.1, 4 validators, public RPC): mean over "
          "trials with 95\\% CI. Finality is the arrival-to-$k$-confirmation latency with "
          "$k$ set above every observed reorganisation.", "tab:live",
          ["Condition", "$n$", "Success (\\%)", "Throughput (rec/s)", "Incl. P50 (s)", "Final P50 (s)",
           "Reorged (\\%)", "Max depth", "Verified (\\%)"], rows, "lrrcccrrr", size="\\scriptsize")


def t_twins(df: pd.DataFrame) -> dict:
    """Emulation fidelity: every live trial against its identically configured emulation twin."""
    from scipy import stats as st
    live = df[(df["backend"] == "purechain") & (df["client"].map(client_generation) == "new")]
    if not len(live):
        return {}
    emu = df[df["backend"] == "poa2_local"].set_index(df.loc[df["backend"] == "poa2_local", "run_id"]
                                                       .str.replace("_poa2_local_", "_purechain_", regex=False))
    pairs = live.join(emu, on="run_id", rsuffix="_emu", how="inner")
    if not len(pairs):
        return {}
    pairs = pairs.assign(grp=pairs["workload"].map(lambda w: "W1--W5" if w.startswith("W") else w))
    rows, m = [], {}
    for col, label, key in (("throughput_tps", "Throughput (rec/s)", "Tps"),
                            ("e2e_latency_s_median", "Inclusion P50 (s)", "Incl"),
                            ("final_e2e_latency_s_median", "Finality P50 (s)", "Final")):
        a, b = pairs[col].astype(float), pairs[f"{col}_emu"].astype(float)
        ok = a.notna() & b.notna()
        p = st.wilcoxon(a[ok], b[ok]).pvalue if ok.sum() >= 6 else float("nan")
        ratio = float(np.median(a[ok] / b[ok]))
        rows.append([label, str(int(ok.sum())), f(float(a[ok].median())), f(float(b[ok].median())),
                     f(ratio), f"{p:.1e}" if p == p else "--"])
        m[f"Twin{key}Ratio"] = f(ratio, 2)
    m["TwinPairs"] = f(len(pairs), 0)
    table(GEN / "t_calibration.tex", "Emulation fidelity: live PureChain trials against identically configured "
          "Clique-emulation twins (same workload window, seed, evidence and offered load). "
          "Medians over pairs; ratio = median of per-pair live/emulation ratios; $p$: Wilcoxon signed-rank test.",
          "tab:calibration", ["Metric", "Pairs", "Live", "Emulation", "Live/emu.", "$p$"], rows, "lrrrrr",
          note="Live latency additionally contains the wide-area round trip, the public load balancer and forks.")
    return m


POLICY = {"record": "Per record", "fixed": "Fixed $b{=}50$", "adaptive": "Controller"}
TRACE = {"saml_d": "SAML-D (business day)", "ibm_aml": "IBM AML (batch posting)"}


def _policy(s: pd.Series) -> str:
    if s.get("batch_policy") == "adaptive":
        return "adaptive"
    return "fixed" if int(s.get("batch_size", 1) or 1) > 1 else "record"


def t_e11(df: pd.DataFrame) -> dict:
    """E11: burst-aware anchoring under trace-driven bursty arrivals."""
    e = df[df["workload"] == "E11"].copy()
    if not len(e):
        return {}
    e["pol"] = e.apply(_policy, axis=1)
    rows, m, shown, db_note = [], {}, (None, None), []
    key_b = {"purechain": "Live", "poa2_local": "Emu", "besu": "Besu", "db_postgres": "DB"}
    key_p = {"record": "Rec", "fixed": "Fix", "adaptive": "Ad"}
    key_s = {"saml_d": "Saml", "ibm_aml": "Ibm"}
    for src in TRACE:
        for b in ORDER:
            for pol in POLICY:
                g = e[(e["source"] == src) & (e["backend"] == b) & (e["pol"] == pol)]
                if not len(g):
                    continue
                gas = g["gas_used_mean"].astype(float).mean() if b != "db_postgres" else float("nan")
                if b == "db_postgres":            # a durable commit in milliseconds: summarised in the note
                    db_note.append(f"{TRACE[src].split(' (')[0]} P95 {f(g['e2e_latency_s_p95'].astype(float).mean())}\\,s")
                else:
                    rows.append([TRACE[src] if shown[0] != src else "", NAME[b] if shown != (src, b) else "",
                                 POLICY[pol], str(len(g)), f(g["success_rate"].mean(), 1, True),
                                 f(g["records_per_submission"].mean(), 1), mean_ci(g["e2e_latency_s_median"]),
                                 mean_ci(g["e2e_latency_s_p95"]), f(g["e2e_latency_s_max"].astype(float).mean()),
                                 mean_ci(g["final_e2e_latency_s_median"]), mean_ci(g["final_e2e_latency_s_p95"]),
                                 f(gas, 0) if gas == gas else "--"])
                    shown = (src, b)
                k = f"Burst{key_p[pol]}{key_s[src]}{key_b[b]}"
                m[f"{k}Med"] = f(g["e2e_latency_s_median"].astype(float).mean())
                m[f"{k}Tail"] = f(g["e2e_latency_s_p95"].astype(float).mean())
                m[f"{k}Max"] = f(g["e2e_latency_s_max"].astype(float).mean())
                m[f"{k}Fin"] = f(g["final_e2e_latency_s_median"].astype(float).mean())
                m[f"{k}FinP"] = f(g["final_e2e_latency_s_p95"].astype(float).mean())
                m[f"{k}Rpt"] = f(g["records_per_submission"].astype(float).mean(), 1)
                m[f"{k}Succ"] = f(g["success_rate"].astype(float).mean(), 1, True)
                if gas == gas:
                    m[f"{k}Gas"] = f(gas, 0)
    table(GEN / "t_e11.tex", "Anchoring under bursty arrivals (E1): per trial, a 24-hour window of the synthetic "
          "transaction set thinned to 10{,}000 records and replayed, time-compressed, at a mean of 30 records/s "
          "(intraday intensity profile preserved); the same windows under every policy and infrastructure. "
          "Latencies are arrival-to-inclusion P50 and P95, Incl. max (the largest arrival-to-inclusion latency of "
          "any record in the trial), and arrival-to-finality P50 and P95; mean over trials with 95\\% bootstrap CI. "
          "Gas: registry without stored batch identifiers (storing them adds the per-identifier check, "
          "Table~\\ref{tab:gas}).",
          "tab:e11", ["Trace", "Infrastructure", "Policy", "$n$", "Success (\\%)", "Rec./tx", "Incl. P50 (s)",
                      "Incl. P95 (s)", "Incl. max (s)", "Final P50 (s)", "Final P95 (s)", "Gas/record"], rows,
          "lllrrrcccccr", size="\\scriptsize",
          note=("PostgreSQL, per record (durable commit): " + "; ".join(db_note) + ".") if db_note else "")
    return m


def t_e12(df: pd.DataFrame, k: int) -> dict:
    """E12: does every record of a burst become final once traffic stops?"""
    e = df[df["workload"] == "E12"]
    if not len(e):
        return {}
    rows, m = [], {}
    label = {("purechain", ""): ("PureChain (live)", "on demand + tail", "burst", "Live"),
             ("poa2_local", ""): ("Clique emulation", "fixed 1\\,s", "burst", "Emu"),
             ("poa2_local", "p0"): ("Clique, period 0", "on demand, no tail", "burst", "Pzero"),
             ("besu", ""): ("Besu QBFT", "fixed 1\\,s", "burst", "Besu"),
             ("purechain", "lone"): ("PureChain (live)", "on demand + tail", "lone", "LiveLone"),
             ("poa2_local", "lone"): ("Clique emulation", "fixed 1\\,s", "lone", "EmuLone"),
             ("poa2_local", "p0_lone"): ("Clique, period 0", "on demand, no tail", "lone", "PzeroLone")}
    for (b, tag), (name, sealing, load, key) in label.items():
        g = e[(e["backend"] == b) & (e["tag"].fillna("") == tag)]
        if not len(g):
            continue
        final_share, last_final, tails = [], [], []
        for rid in g["run_id"]:
            it = pd.read_parquet(RAW / rid / "items.parquet", columns=["status", "block", "t_sched"])
            bl = pd.read_csv(RAW / rid / "blocks.csv")
            canon = bl[bl["canonical"]] if "canonical" in bl else bl
            seen = canon.groupby("number")["t_observed"].min()
            ok = it[it["status"] == "committed"]
            depth = 0 if b == "besu" else k            # QBFT is final on inclusion
            t_k = (ok["block"] + depth).map(seen)
            final_share.append(float(t_k.notna().mean()))
            last = ok.loc[ok["t_sched"].idxmax()]
            tk = seen.get(last["block"] + depth)
            last_final.append(tk - last["t_sched"] if tk is not None else float("nan"))
            # tail: canonical blocks sealed after the last record's block within the observation window
            tails.append(int((canon["number"].drop_duplicates() > ok["block"].max()).sum()))
        lf = pd.Series(last_final)
        rows.append([name, sealing, load, str(len(g)), f(100 * float(np.mean(final_share)), 1),
                     f(lf.median()) if lf.notna().any() else "never", f"{lf.notna().sum()}/{len(lf)}",
                     span(min(tails), max(tails))])
        m[f"Starve{key}Final"] = f(100 * float(np.mean(final_share)), 1)
        m[f"Starve{key}Last"] = f(lf.median()) if lf.notna().any() else "never"
        m[f"Starve{key}Trials"] = f"{lf.notna().sum()}/{len(lf)}"
        m[f"Starve{key}TailMin"], m[f"Starve{key}TailMax"] = str(min(tails)), str(max(tails))
        m[f"Starve{key}TailRange"] = span(min(tails), max(tails))
    gen = RESULTS / "live_submitter_generation.json"
    if gen.exists():
        m["StrandedTxs"] = f(int(json.loads(gen.read_text())["stranded_txs"]), 0)
    table(GEN / "t_e12.tex", f"Finality when traffic stops (E3). Burst: 300 records at 50 records/s, then no "
          f"traffic; lone: one record after the chain had been idle for 60\\,s. Share of records that reached "
          f"finality ({k} confirmations; QBFT: inclusion), time from the last arrival until the last record was final "
          f"(median over trials; ``never'': not final within 90\\,s for bursts, 180\\,s for lone records), trials in "
          f"which every record became final, and the tail: blocks sealed after the last record's block within "
          f"the window, observed up to 20.", "tab:e12",
          ["Infrastructure", "Sealing", "Load", "$n$", "Records final (\\%)", "Last record final (s)", "Trials final",
           "Tail (blocks)"], rows, "lllrrrrr", size="\\scriptsize")
    return m


def t_idle() -> dict:
    """E10: demand-driven block production on live PureChain against fixed-period chains."""
    d = RESULTS / "idle"
    m, rows = {}, []
    if (d / "purechain_ledger.json").exists():
        L = json.loads((d / "purechain_ledger.json").read_text())
        rows.append(["PureChain (live)", "on demand", f(L["span_h"], 0), f(L["blocks_per_h"], 0),
                     f(L["empty_share"], 1, True), f(L["bytes_per_h"] / 1e6 * 24, 2), "n/m"])
        m.update({"IdleSpanDays": f(L["span_h"] / 24, 1), "IdleBlocksPerH": f(L["blocks_per_h"], 0),
                  "IdleAvoidedPct": f(L["blocks_avoided_share"], 1, True), "IdleTimePct": f(L["idle_share_of_time"], 1, True),
                  "IdleTail": f(L["wake_tail_blocks_median"], 0),
                  "IdleWakePeriods": f(L["wake_periods"], 0),
                  "IdleTailMin": str(min(int(k) for k in L["wake_tail_counts"])),
                  "IdleAvoidedMB": f(L["empty_bytes_avoided_per_year_mb"], 0)})
    if (d / "purechain_blocks.csv").exists():   # longest gap after a block with transactions in the E10 scan
        B = pd.read_csv(d / "purechain_blocks.csv", usecols=["timestamp", "tx_count"])
        m["IdleLoadGapMax"] = f(B["timestamp"].diff().shift(-1)[B["tx_count"] > 0].max(), 0)
    if (d / "purechain_tails_recent.json").exists():   # later blocks: is the tail sealed after every idle period?
        R = json.loads((d / "purechain_tails_recent.json").read_text())
        m.update({"TailRecentHours": f(R["span_h"], 0), "TailRecentPeriods": f(R["wake_periods"], 0),
                  "TailRecentNone": f(R["wake_tail_counts"].get("0", 0), 0)})
    if (d / "local_idle.json").exists():
        for b, v in json.loads((d / "local_idle.json").read_text()).items():
            rows.append([NAME[b], "fixed 1\\,s", f(v["window_s"] / 3600, 2), f(v["blocks_per_h"], 0),
                         f(v["empty_share"], 1, True), f(v["bytes_per_h"] / 1e6 * 24, 2), f(v["idle_energy_wh_per_day"], 1)])
            k = {"besu": "Besu", "poa2_local": "Emu"}[b]
            m[f"IdleBlocksPerH{k}"] = f(v["blocks_per_h"], 0)
            m[f"IdleWhDay{k}"] = f(v["idle_energy_wh_per_day"], 1)
    if rows:
        table(GEN / "t_idle.tex", "Block production without evidence traffic (E5). PureChain: passive observation "
              "of the public chain; local chains: 4 validators, no transactions sent. Energy: measured dynamic CPU "
              "energy of the validator containers; n/m: remote validators cannot be metered.", "tab:idle",
              ["Infrastructure", "Sealing", "Observed (h)", "Blocks/h", "Empty (\\%)", "Ledger growth (MB/day)",
               "Idle energy (Wh/day)"], rows, "llrrrrr", size="\\scriptsize")
    if (d / "wake.csv").exists():
        w = pd.read_csv(d / "wake.csv")
        w = w[w["status"] == "committed"]
        for t, g in w.groupby("target_idle_s"):
            k = {0: "Warm", 10: "Ten", 30: "Thirty", 60: "Sixty", 120: "OneTwenty"}.get(int(t))
            if k:
                m[f"Wake{k}"] = f(g["commit_latency_s"].median(), 2)
    return m



DSNAME = {"momtsim": "MoMTSim", "saml_d": "SAML-D", "ibm_aml": "IBM AML"}


def _workload_view(df: pd.DataFrame) -> pd.DataFrame:
    """E1 (MoMTSim) and E7 (SAML-D, IBM AML) trials under one (workload, dataset) key."""
    e1 = df[df["workload"].isin(["W1", "W2", "W5"])].assign(wl=lambda d: d["workload"])
    e7 = df[df["workload"] == "E7"].assign(wl=lambda d: d["tag"].fillna(""))
    return pd.concat([e1, e7])


def t_datasets(df: pd.DataFrame) -> None:
    from scipy import stats as st
    v = _workload_view(df)
    rows = []
    for (wl, b), g in v.groupby(["wl", "backend"]):
        per = {s: gg for s, gg in g.groupby("source")}
        groups = [gg["e2e_latency_s_median"].dropna().astype(float) for gg in per.values()]
        p = st.kruskal(*groups).pvalue if len(groups) >= 2 and all(len(x) >= 2 for x in groups) and \
            len({round(x, 12) for grp in groups for x in grp}) > 1 else float("nan")
        for i, (s, gg) in enumerate(sorted(per.items(), key=lambda kv: list(DSNAME).index(kv[0]))):
            rows.append([tex(wl) if i == 0 else "", NAME[b] if i == 0 else "", DSNAME[s], str(len(gg)),
                         mean_ci(gg["throughput_tps"]), mean_ci(gg["e2e_latency_s_median"]),
                         mean_ci(gg["final_e2e_latency_s_median"]), f(gg["verify_success_rate"].mean(), 1, True),
                         (f"{p:.3f}" if p == p else "--") if i == 0 else ""])
    table(GEN / "t_datasets.tex", "Cross-dataset replication (E1 on MoMTSim, E7 on SAML-D and IBM AML): mean over "
          "trials with 95\\% CI. $p$: Kruskal--Wallis test of a dataset effect on median inclusion latency.",
          "tab:datasets", ["Workload", "Infrastructure", "Dataset", "$n$", "Throughput (rec/s)", "Incl. P50 (s)",
                           "Final P50 (s)", "Verified (\\%)", "$p$"], rows, "lllrcccrr", size="\\scriptsize")


def t_tamper_by_dataset() -> None:
    fr = []
    for p in RAW.glob("*/tamper.parquet"):
        sm = json.loads((p.parent / "summary.json").read_text())
        fr.append(pd.read_parquet(p).assign(source=sm["source"], backend=sm["backend"]))
    if not fr:
        return
    d = pd.concat(fr)
    rows = []
    for (s, b), g in d.groupby(["source", "backend"]):
        mal, ben = g[g["kind"] == "malicious"], g[g["kind"] == "benign"]
        lo, _ = clopper_pearson(int(mal["detected"].sum()), len(mal))
        _, hi = clopper_pearson(int(ben["detected"].sum()), len(ben))
        rows.append([DSNAME[s], NAME[b], f(len(mal), 0), f(100 * mal["detected"].mean()), f(100 * lo, 3),
                     f(len(ben), 0), f(100 * ben["detected"].mean()), f(100 * hi, 3)])
    table(GEN / "t_tamper_ds.tex", "Tamper detection by dataset and infrastructure (E6).", "tab:tamperds",
          ["Dataset", "Infrastructure", "Malicious", "TDR (\\%)", "TDR 95\\% lower", "Benign", "FPR (\\%)",
           "FPR 95\\% upper"], rows, "llrrrrrr", size="\\scriptsize")


def t_efficiency(df: pd.DataFrame) -> None:
    v = _workload_view(df)
    v = v[v["source"] == "momtsim"] if len(v[v["source"] == "momtsim"]) else v
    rows = []
    for (wl, b), g in v.groupby(["wl", "backend"]):
        measured = bool(g["infra_energy_measured"].fillna(False).astype(bool).any())
        win = g["window_s"].astype(float)
        cpu_client = (g["window_client_cpu_s"].astype(float) / win).mean()
        cpu_infra = (g["window_infra_cpu_s"].astype(float) / win).mean() if measured else float("nan")
        per_rec = lambda col: (g[col].astype(float) / g["committed"].astype(float)).mean()
        rows.append([tex(wl), NAME[b], f(cpu_client, 2), f(cpu_infra, 2) if measured else "n/m",
                     f(g["infra_mem_peak_mb"].astype(float).mean(), 0) if measured else "n/m",
                     f(per_rec("network_bytes") / 1024, 2), f(g["onchain_payload_bytes_mean"].astype(float).mean(), 0),
                     f(g["evidence_bytes_mean"].astype(float).mean() / 1024, 1),
                     f(g["energy_j_per_tx"].astype(float).mean(), 3), f(g["energy_dynamic_j_per_tx"].astype(float).mean(), 4),
                     f(g["cost_total_usd_per_tx"].astype(float).mean() * 1e6, 2)])
    table(GEN / "t_efficiency.tex", "Efficiency metrics (MoMTSim workloads): CPU in cores used on average during the "
          "processing window, peak infrastructure memory, network I/O, storage overhead per record, energy and "
          "operational cost. n/m: not measurable (remote validators).", "tab:efficiency",
          ["Workload", "Infrastructure", "CPU client", "CPU infra", "Mem (MB)", "Net (KB/rec)", "On-chain (B/rec)",
           "Off-chain (KB/rec)", "E attr. (J)", "E dyn. (J)", "\\textmu USD/rec"], rows, "llrrrrrrrrr",
          size="\\scriptsize")


def key_macros(df: pd.DataFrame) -> dict:
    """Headline comparison numbers used in the results text (all derived from trial means)."""
    m = {}
    def mean(sel, col):
        x = df.loc[sel, col].dropna().astype(float)
        return float(x.mean()) if len(x) else float("nan")
    wl = df["workload"]
    B = lambda b: df["backend"] == b
    # E2: sustained per-record capacity (unthrottled) and latency under 200 rec/s
    for b, k in (("poa2_local", "Emu"), ("besu", "Besu"), ("db_postgres", "DB")):
        m[f"MaxTps{k}"] = f(mean((wl == "E2") & B(b) & (df["rate"] == 0), "throughput_tps"), 0)
        m[f"TpsTwoH{k}"] = f(mean((wl == "E2") & B(b) & (df["rate"] == 200), "throughput_tps"), 0)
        m[f"LatTwoH{k}"] = f(mean((wl == "E2") & B(b) & (df["rate"] == 200), "e2e_latency_s_median"), 2)
    r = mean((wl == "E2") & B("poa2_local") & (df["rate"] == 0), "throughput_tps") / \
        mean((wl == "E2") & B("besu") & (df["rate"] == 0), "throughput_tps")
    m["MaxTpsRatio"] = f(r, 1)
    # E1 W2 latency / finality
    for b, k in (("poa2_local", "Emu"), ("besu", "Besu"), ("db_postgres", "DB"), ("ethpos", "Pos")):
        m[f"InclWtwo{k}"] = f(mean((wl == "W2") & B(b), "e2e_latency_s_median"), 2)
        m[f"FinalWtwo{k}"] = f(mean((wl == "W2") & B(b), "final_e2e_latency_s_median"), 2 if b != "ethpos" else 0)
        m[f"TpsWtwo{k}"] = f(mean((wl == "W2") & B(b), "throughput_tps"), 1)
    fe, fp = mean((wl == "W2") & B("poa2_local"), "final_e2e_latency_s_median"), \
        mean((wl == "W2") & B("ethpos"), "final_e2e_latency_s_median")
    m["FinalRatioPos"] = f(fp / fe, 0)
    m["InclRatioPos"] = f(mean((wl == "W2") & B("ethpos"), "e2e_latency_s_median") /
                          mean((wl == "W2") & B("poa2_local"), "e2e_latency_s_median"), 0)
    # fees and cost
    m["FeePos"] = f(mean(wl.isin(["W1", "W2"]) & B("ethpos"), "cost_fee_usd_per_tx"), 2)
    for b, k in (("poa2_local", "Emu"), ("besu", "Besu"), ("db_postgres", "DB"), ("ethpos", "Pos")):
        m[f"Cost{k}"] = f(mean((wl == "W2") & B(b), "cost_total_usd_per_tx") * 1e6, 1)
        m[f"Edyn{k}"] = f(mean((wl == "W2") & B(b), "energy_dynamic_j_per_tx"), 3)
        m[f"Eatt{k}"] = f(mean((wl == "W2") & B(b), "energy_j_per_tx"), 2)
    m["EdynRatioBesu"] = f(mean((wl == "W2") & B("besu"), "energy_dynamic_j_per_tx") /
                           mean((wl == "W2") & B("poa2_local"), "energy_dynamic_j_per_tx"), 1)
    m["EattRatioPos"] = f(mean((wl == "W2") & B("ethpos"), "energy_j_per_tx") /
                          mean((wl == "W2") & B("poa2_local"), "energy_j_per_tx"), 1)
    # verification time
    for b, k in (("poa2_local", "Emu"), ("besu", "Besu"), ("db_postgres", "DB")):
        m[f"Verify{k}"] = f(1e3 * mean(wl.str.match(r"W\d") & B(b), "verify_total_s_median"), 1)
    # E5 validator scaling
    e5 = df[wl == "E5"]
    for v in (4, 7, 10):
        for b, k in (("poa2_local", "Emu"), ("besu", "Besu")):
            g = e5[(e5["backend"] == b) & (e5["tag"] == f"v{v}") & (e5["rate"] == 200)]
            m[f"VTps{k}{['','','','','Four','','','Seven','','','Ten'][v]}"] = f(g["throughput_tps"].mean(), 0) if len(g) else "--"
    g = e5[(e5["backend"] == "besu") & (e5["tag"] == "v10")]
    m["BesuTenSuccess"] = f(g["success_rate"].mean(), 0, True) if len(g) else "--"
    # E9 batching
    e9 = df[(wl == "E9") & (df["backend"] == "poa2_local")]
    if e9["tag"].fillna("").str.startswith("v2").any():     # gas of the registry as described: the revised one
        e9 = e9[e9["tag"].fillna("").str.startswith("v2")]
        if "batch_ids" in e9:
            e9 = e9[e9["batch_ids"].fillna("store") == "store"]
    m["GasKone"] = f(e9.loc[e9["batch_size"] == 1, "gas_used_mean"].mean(), 0)
    m["GasKhundred"] = f(e9.loc[e9["batch_size"] == 100, "gas_used_mean"].mean(), 0)
    ep = df[(wl == "E9") & (df["backend"] == "ethpos")]
    m["PosTpsKone"] = f(ep.loc[ep["batch_size"] == 1, "throughput_tps"].mean(), 1)
    m["PosTpsKfifty"] = f(ep.loc[ep["batch_size"] == 50, "throughput_tps"].mean(), 1)
    m["PosFeeKfifty"] = f(ep.loc[ep["batch_size"] == 50, "cost_fee_usd_per_tx"].mean(), 3)
    # E7 cross-dataset: smallest Kruskal-Wallis p for the PoA2 emulation
    from scipy import stats as st
    v = _workload_view(df)
    ps = []
    for wlk, g in v[v["backend"] == "poa2_local"].groupby("wl"):
        groups = [gg["e2e_latency_s_median"].dropna() for _, gg in g.groupby("source")]
        if len(groups) >= 2 and all(len(x) >= 2 for x in groups):
            ps.append(st.kruskal(*groups).pvalue)
    m["KWminEmu"] = f(min(ps), 2) if ps else "--"
    # live PureChain (current client) at the same operating points as the emulation
    live = df["client"].map(client_generation).eq("new") & B("purechain")
    m["InclWoneLive"] = f(mean((wl == "W1") & live, "e2e_latency_s_median"), 2)
    m["FinalWoneLive"] = f(mean((wl == "W1") & live, "final_e2e_latency_s_median"), 2)
    m["InclWtwoLive"] = f(mean((wl == "W2") & live, "e2e_latency_s_median"), 2)
    m["FinalWtwoLive"] = f(mean((wl == "W2") & live, "final_e2e_latency_s_median"), 2)
    m["TpsWtwoLive"] = f(mean((wl == "W2") & live, "throughput_tps"), 1)
    m["MaxTpsLive"] = f(mean((wl == "E2") & live & (df["rate"] == 0), "throughput_tps"), 0)
    m["TpsTwoHLive"] = f(mean((wl == "E2") & live & (df["rate"] == 200), "throughput_tps"), 0)
    m["LatTwoHLive"] = f(mean((wl == "E2") & live & (df["rate"] == 200), "e2e_latency_s_median"), 2)
    m["VerifyLive"] = f(1e3 * mean(wl.str.match(r"W\d") & live, "verify_total_s_median"), 1)
    m["FinalRatioPosLive"] = f(mean((wl == "W2") & B("ethpos"), "final_e2e_latency_s_median") /
                               mean((wl == "W2") & live, "final_e2e_latency_s_median"), 0)
    lv = v[(v["backend"] == "purechain") & v["client"].map(client_generation).eq("new")]
    ps = [st.kruskal(*[gg["e2e_latency_s_median"].dropna() for _, gg in g.groupby("source")]).pvalue
          for _, g in lv.groupby("wl") if g["source"].nunique() >= 2 and g.groupby("source").size().min() >= 2]
    m["KWminLive"] = f(min(ps), 2) if ps else "--"
    lb = df[(wl == "E9") & live]
    for k, name in ((1, "Kone"), (100, "Khundred")):
        g = lb[(lb["batch_size"] == k) & (lb["rate"] == 200)]
        m[f"LiveTps{name}"] = f(g["throughput_tps"].mean(), 0) if len(g) else "--"
    return m


def macros_file(m: dict, df: pd.DataFrame) -> None:
    probe = json.loads((RESULTS / "network_probe.json").read_text()) if (RESULTS / "network_probe.json").exists() else {}
    m.update({"Trials": f(len(df), 0), "LiveValidators": str(probe.get("n_validators", 4)),
              "LiveRTT": f(probe.get("rpc_rtt_ms_median", float("nan")), 1),
              "LiveGasLimit": f(probe.get("block_gas_limit", 30_000_000), 0),
              "LiveInTurn": f(probe.get("in_turn_share", float("nan")), 0, True)})
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    for k in ("TDR", "TDRlo", "FPR", "FPRhi", "TamperCases", "BenignCases"):
        if k not in m:
            lines.append(f"\\newcommand{{\\{k}}}{{--}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ build
def t_stats() -> None:
    """Pairwise tests of the PoA2 emulation against each baseline on W2 (stats.compare_backends)."""
    d = pd.read_csv(TABLES / "e1_backend_comparisons.csv")
    d = d[(d.workload == "W2") & ((d.a == "poa2_local") | (d.b == "poa2_local"))]
    metrics = [("commit_latency_s_median", "Inclusion latency P50 (s)", 1), ("commit_latency_s_p95",
               "Inclusion latency P95 (s)", 1), ("energy_j_per_tx", "Attributed energy (J/record)", 1),
               ("cost_total_usd_per_tx", "Cost ($\\mu$USD/record)", 1e6)]
    names = {"besu": "Besu QBFT", "db_postgres": "PostgreSQL", "ethpos": "Ethereum PoS"}

    def p(v):
        return "$<$0.001" if v < 0.001 else f"{v:.3f}"
    rows, dagger = [], False
    for metric, label, scale in metrics:
        for i, (other, name) in enumerate(names.items()):
            r = d[(d.metric == metric) & ((d.a == other) | (d.b == other))]
            if r.empty:
                continue
            r = r.iloc[0]
            mine_first = r.a == "poa2_local"
            m_other, m_mine = (r.mean_b, r.mean_a) if mine_first else (r.mean_a, r.mean_b)
            n_other = r.n_b if mine_first else r.n_a
            delta = r.cliffs_delta if mine_first else -r.cliffs_delta
            mark = "$^{\\dagger}$" if n_other < 3 else ""
            dagger |= bool(mark)
            rows.append([label if i == 0 else "", name, f(scale * m_other, 3), f(scale * m_mine, 3),
                         f"${delta:+.2f}$", p(r.p_welch_holm), p(r.p_mwu_holm) + mark])
    n_poa = int(d.n_a[d.a == "poa2_local"].tolist()[0] if (d.a == "poa2_local").any() else d.n_b.iloc[0])
    table(GEN / "t_stats.tex",
          "Pairwise comparison of the Clique emulation with each baseline (E1, W2, 10{,}000 "
          f"records at 50 records/s, {n_poa} emulation trials). Trial means; Cliff's $\\delta$ is "
          "the effect size of the emulation relative to the comparator ($+1$: every "
          "emulation trial is larger). $p$-values are Holm--Bonferroni adjusted across the "
          "comparisons of each workload." + (" $^{\\dagger}$With two PoS trials the Mann--Whitney test cannot "
          "reach significance; the Welch test is reported for completeness." if dagger else ""),
          "tab:stats", ["Metric", "Comparator", "Comparator mean", "Emulation mean",
                        "Cliff's $\\delta$", "$p_{\\mathrm{Welch}}$", "$p_{\\mathrm{MWU}}$"], rows, "llrrrrr",
          size="\\scriptsize")


def build() -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    m = live_forks()
    k = int(m.get("FinalityDepth", 6))
    stats.main()
    figures.main(k)
    for p in (RESULTS / "figures").glob("*.pdf"):
        shutil.copy(p, FIGDIR / p.name)
    # PoA2-style finality: k confirmations, k above the deepest reorganisation ever observed on the live chain
    df = stats.apply_finality_depth(load_summaries(), k)
    # trials logged blocks only to the deepest depth recorded when they ran (7), so records in a trial's last
    # block(s) have no k-th confirmation in the log and are left out of finality latencies
    fk = pd.read_csv(TABLES / f"finality_k{k}.csv")
    m["FinalMissingPct"] = f(100 * fk["final_k_missing"].sum() / df.loc[df["run_id"].isin(fk["run_id"]), "committed"].sum(), 1)
    t_status(df)
    t_live_campaign(df)
    m.update(t_twins(df))
    m.update(t_idle())
    m.update(t_e11(df))
    m.update(t_e12(df, k))
    m.update(t_tamper())
    m.update(t_insider(df))
    m.update(revision.build(df, k))
    t_batching(df)
    t_datasets(df)
    t_tamper_by_dataset()
    t_efficiency(df)
    t_stats()
    t_generic(df, "E1", "workload", "Workload", "Controlled workloads (B1), offered load 50 records/s.", "tab:e1")
    t_generic(df, "E2", "rate", "Offered (rec/s)", "Offered-load sweep (B2).", "tab:e2")
    t_generic(df, "E3", "doc_bytes", "Evidence (bytes)", "Evidence-size sweep (B3).", "tab:e3")
    t_generic(df, "E4", "level", "Level", "Evidence-complexity sweep (B4).", "tab:e4")
    t_generic(df, "E5", ["tag", "rate"], "Validators @ offered", "Validator-count sweep (B5); every validator is "
              "limited to 1 CPU. Besu QBFT with 10 validators lost liveness in 3 of 4 trials (BFT event-queue overflow).",
              "tab:e5")
    m.update(key_macros(df))
    for key in ("BurstRecIbmLiveTail", "BurstRecSamlLiveTail", "BurstFixIbmLiveTail", "BurstAdIbmLiveTail",
                "BurstFixSamlLiveTail", "BurstAdSamlLiveTail", "BurstFixIbmLiveFin", "BurstAdIbmLiveFin"):
        if key in m and m[key] != "--":
            v = float(m[key].replace(",", ""))
            m[f"{key}R"] = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    macros_file(m, df)
    figs = sorted(p.stem for p in FIGDIR.glob("*.pdf"))
    (GEN / "figures.tex").write_text("".join(
        f"\\IfFileExists{{figures/{s}.pdf}}{{\\def\\has{s.split('_')[0]}{{1}}}}{{}}\n" for s in figs), encoding="utf-8")
    return compile_pdf()


def compile_pdf() -> Path:
    if not (PAPER / "main.tex").exists():   # public code: tables, figures and macros only
        print("generated", GEN, "and", FIGDIR, "(no manuscript source to compile)")
        return GEN
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    if r.returncode != 0:   # stale auxiliary files after structural edits: clean once and retry
        subprocess.run(["latexmk", "-c"], cwd=PAPER, capture_output=True)
        r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError("LaTeX build failed")
    zp = PAPER / "finverity_overleaf.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [PAPER / "main.tex", PAPER / "references.bib", *GEN.glob("*.tex"), *FIGDIR.glob("*.pdf")]:
            if p.exists():
                z.write(p, p.relative_to(PAPER))
    print("built", PAPER / "main.pdf", "and", zp)
    return PAPER / "main.pdf"


if __name__ == "__main__":
    build()
