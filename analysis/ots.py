"""B8: OpenTimestamps baseline (experiments/ots_baseline.py). The same record commitments stamped through the public
OpenTimestamps calendars and verified against Bitcoin block headers; latency to the Bitcoin attestation and to six
confirmations, proof size and cost, per record and per Merkle batch of 1,000 records.

Stamping rounds four hours apart give independent draws of the calendars' aggregation round and of Bitcoin's block
interval; the stamps of one round share a Bitcoin block, so the round, not the stamp, is the unit of independence.

    python -m analysis.ots            # prints the summary
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from finverity.config import RESULTS

DIR = RESULTS / "ots"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = sorted(r for r in DIR.glob("run_*") if (r / "verified.csv").exists())
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    v = pd.concat([pd.read_csv(r / "verified.csv").assign(run=r.name) for r in runs], ignore_index=True)
    s = pd.concat([pd.read_csv(r / "stamps.csv").assign(run=r.name) for r in runs], ignore_index=True)
    return v, s


def analyse() -> dict:
    v, s = load()
    if v.empty:
        return {}
    inc, c6 = v["inclusion_s"] / 60, v["conf6_s"] / 60
    return {"runs": int(v["run"].nunique()), "stamps": int(len(s)), "verified": int(v["verified"].sum()),
            "verified_pct": 100 * float(v["verified"].mean()), "blocks": int(v["height"].nunique()),
            "failed": int(s["failed"].notna().sum()), "calendars_min": int(s["calendars_ok"].min()),
            "batch_size": int(v.loc[v["mode"] == "batch", "records"].max()) if (v["mode"] == "batch").any() else 0,
            "stamp_s": float(s["stamp_s"].median()),
            "inc_min": float(inc.min()), "inc_med": float(inc.median()), "inc_p95": float(np.percentile(inc, 95)),
            "inc_max": float(inc.max()),
            "c6_min": float(c6.min()), "c6_med": float(c6.median()), "c6_max": float(c6.max()),
            "proof_min": float(v["proof_bytes"].min()) / 1000, "proof_med": float(v["proof_bytes"].median()) / 1000,
            "proof_max": float(v["proof_bytes"].max()) / 1000,
            "by_mode": {m: {"n": int(len(g)), "inc_med": float(g["inclusion_s"].median() / 60),
                            "inc_p95": float(np.percentile(g["inclusion_s"] / 60, 95)),
                            "c6_med": float(g["conf6_s"].median() / 60), "c6_max": float(g["conf6_s"].max() / 60),
                            "proof_med": float(g["proof_bytes"].median()) / 1000}
                        for m, g in v.groupby("mode")}}


def build() -> dict:
    """Macros (numbers only through macros) and the supplementary B8 table."""
    from .report import GEN, f, table
    r = analyse()
    if not r:
        return {}
    m = {"OtsRuns": str(r["runs"]), "OtsStamps": f(r["stamps"], 0), "OtsVerifiedPct": f(r["verified_pct"], 0),
         "OtsBlocks": str(r["blocks"]), "OtsFailed": str(r["failed"]), "OtsCalendarsMin": str(r["calendars_min"]),
         "OtsBatchSize": f(r["batch_size"], 0), "OtsStampSec": f(r["stamp_s"], 1),
         "OtsInclMin": f(r["inc_min"], 0), "OtsInclMed": f(r["inc_med"], 0), "OtsInclPnf": f(r["inc_p95"], 0),
         "OtsInclMax": f(r["inc_max"], 0),
         "OtsConfMin": f(r["c6_min"], 0), "OtsConfMed": f(r["c6_med"], 0), "OtsConfMaxH": f(r["c6_max"] / 60, 1),
         "OtsProofMin": f(r["proof_min"], 1), "OtsProofMed": f(r["proof_med"], 1), "OtsProofMax": f(r["proof_max"], 1)}
    names = {"record": "One stamp per record", "batch": f"One stamp per Merkle batch of {r['batch_size']:,} records"}
    rows = [[names.get(k, k), f(g["n"], 0), f(g["inc_med"], 0), f(g["inc_p95"], 0), f(g["c6_med"], 0),
             f(g["c6_max"], 0), f(g["proof_med"], 1)] for k, g in sorted(r["by_mode"].items(), reverse=True)]
    table(GEN / "t_ots.tex", "OpenTimestamps baseline (B8): the same record commitments stamped through the four public "
          f"OpenTimestamps aggregators in {r['runs']} rounds four hours apart and verified against Bitcoin block headers; "
          "latency from submission to the Bitcoin block that attests the stamp and to its sixth confirmation, in "
          "minutes, and the size of the upgraded proof (for a batch, without the registry's Merkle path to the record). "
          "Stamps of one round share a Bitcoin block.", "tab:ots",
          ["Mode", "Stamps", "Attested P50", "Attested P95", "6 conf. P50", "6 conf. max", "Proof (KB)"], rows,
          "lrrrrrr", size="\\scriptsize", wide=False)
    return m


if __name__ == "__main__":
    print(json.dumps(analyse(), indent=1))
