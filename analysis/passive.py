"""E15: passive fork monitoring of live PureChain (experiments/passive_forks.py), no transactions of ours.

The canonical chain is rebuilt from parent hashes, walking back from the highest block seen; heights within SETTLE
blocks of it are left out because their fate is still open. A height is replaced when a block version other than the
canonical one was seen at it, and a reorganisation event is a run of consecutive replaced heights, as for the trials
(analysis.revision.reorg_events). Signers are recovered from the stored headers; difficulty 1 marks out-of-turn seals.

    python -m analysis.passive [DIR]      # default results/passive/out; prints the summary
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finverity.config import RESULTS

DIR = RESULTS / "passive" / "out"           # hourly block files copied from the monitor host
SETTLE = 64


def load(d: Path = DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(d.glob("blocks_*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("hash")
    for c in ("number", "timestamp", "difficulty", "gasUsed"):
        df[c] = df[c].map(lambda x: int(x, 16))
    return df.sort_values(["number", "t_observed"]).reset_index(drop=True)


def signers(df: pd.DataFrame) -> pd.Series:
    from finverity.backends.evm import recover_clique_signer
    return df.apply(lambda r: recover_clique_signer({k: v for k, v in r.items() if isinstance(v, str)}
                                                    | {k: hex(int(r[k])) for k in ("number", "timestamp",
                                                                                   "difficulty", "gasUsed")}), axis=1)


def analyse(d: Path = DIR) -> dict:
    df = load(d)
    if df.empty:
        return {}
    parent = dict(zip(df["hash"], df["parentHash"]))
    h, canon = df.loc[df["number"].idxmax(), "hash"], set()
    while h in parent:
        canon.add(h)
        h = parent[h]
    df["canonical"] = df["hash"].isin(canon)
    top = df["number"].max() - SETTLE
    seen = (pd.to_datetime(df["t_observed"].min(), unit="s"), pd.to_datetime(df["t_observed"].max(), unit="s"))
    df = df[df["number"] <= top].copy()
    df["signer"] = signers(df)
    cn = df[df["canonical"]]
    heights = cn["number"].to_numpy()
    lo, hi = int(heights.min()), int(heights.max())
    missing = (hi - lo + 1) - len(np.unique(heights))   # heights never observed (monitor down or gap in the walk)
    replaced = sorted(set(df.loc[~df["canonical"], "number"]) & set(heights))
    depths, run, prev = [], 0, None
    for x in replaced:
        if prev is not None and x == prev + 1:
            run += 1
        else:
            if run:
                depths.append(run)
            run = 1
        prev = x
    if run:
        depths.append(run)
    orph = df[~df["canonical"]]
    span_s = float(cn["timestamp"].max() - cn["timestamp"].min())
    gaps = np.diff(np.sort(cn["timestamp"].to_numpy()))
    status = d / "status.jsonl"
    clients = sorted({json.loads(x)["client"] for x in status.read_text().splitlines() if x.strip()}) \
        if status.exists() else []
    return {
        "watch_from_utc": seen[0].isoformat(), "watch_to_utc": seen[1].isoformat(),
        "first_utc": pd.to_datetime(cn["timestamp"].min(), unit="s").isoformat(),
        "last_utc": pd.to_datetime(cn["timestamp"].max(), unit="s").isoformat(),
        "observed_days": span_s / 86400, "heights": int(len(cn)), "missing_heights": int(missing),
        "replaced": len(replaced), "replaced_pct": 100 * len(replaced) / len(cn), "events": len(depths),
        "depths": depths, "max_depth": int(max(depths)) if depths else 0,
        "orphans": int(len(orph)), "orphan_oot_pct": 100 * float((orph["difficulty"] == 1).mean()) if len(orph) else None,
        "canon_oot_pct": 100 * float((cn["difficulty"] == 1).mean()),
        "signers": int(cn["signer"].nunique()), "tx_blocks_pct": 100 * float((cn["tx_count"] > 0).mean()),
        "txs": int(cn["tx_count"].sum()), "active_share": float(len(cn)) / max(span_s, 1.0),
        "idle_gaps_over_60s": int((gaps > 60).sum()), "clients": clients,
    }


def build(k: int = 8) -> dict:
    """Macros for the report (E8 in the paper); the fork table by condition is analysis.revision.t_forkcond."""
    from .report import f
    r = analyse()
    if not r:
        return {}
    m = {"PsvDays": f(r["observed_days"], 1), "PsvHeights": f(r["heights"], 0), "PsvMissing": f(r["missing_heights"], 0),
         "PsvReplaced": f(r["replaced"], 0), "PsvReplacedPct": f(r["replaced_pct"], 2), "PsvEvents": f(r["events"], 0),
         "PsvMaxDepth": str(r["max_depth"]), "PsvSigners": str(r["signers"]), "PsvTxBlocksPct": f(r["tx_blocks_pct"], 1),
         "PsvTxs": f(r["txs"], 0), "PsvFrom": str(pd.Timestamp(r["watch_from_utc"]).day),
         "PsvTo": f"{pd.Timestamp(r['watch_to_utc']).day} {pd.Timestamp(r['watch_to_utc']):%B %Y}", "PsvCanonOotPct": f(r["canon_oot_pct"], 1),
         "PsvBeyondK": f(sum(d > k for d in r["depths"]), 0)}
    if r["orphan_oot_pct"] is not None:
        m["PsvOrphanOotPct"] = f(r["orphan_oot_pct"], 0)
    return m


if __name__ == "__main__":
    out = analyse(Path(sys.argv[1]) if len(sys.argv) > 1 else DIR)
    print(json.dumps({k: v for k, v in out.items() if k != "depths"}, indent=1, default=str))
    print("depths:", sorted(out.get("depths", []), reverse=True)[:20])
