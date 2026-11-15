"""Signer-coalition experiment on the 4-signer Clique emulation (S2/S3 by rewriting history).

An insider who controls c of the N = 4 signers isolates them from the honest signers (the honest containers are
paused) and keeps sealing. Clique lets a signer seal at most one block in any floor(N/2) + 1 = 3 consecutive blocks,
so a coalition of c < 3 signers can extend a chain by at most c blocks and then stalls; it can never build a branch
longer than a k-confirmation suffix (k = 8) and so cannot rewrite final history. A majority (c = 3) keeps sealing.

    python -m experiments.coalition            # c = 1, 2, 3; 60 s each -> results/coalition.json
    python -m experiments.coalition outage     # local PureChain: one validator down during a burst -> results/outage.json

The outage run checks the liveness of the released client's per-validator sealing policy when one of the four
validators is unreachable: 3 of 4 signers still meet Clique's limit, so bursts must still commit and finalise.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import requests

from finverity.config import POA2_LOCAL, RESULTS
from infra import isolate, poa2


def _head() -> int:
    r = requests.post(POA2_LOCAL.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []},
                      timeout=5)
    return int(r.json()["result"], 16)


def run(window_s: float = 60.0) -> list[dict]:
    isolate("poa2_local")
    out = []
    for c in (1, 2, 3):
        poa2.up(4, 1)
        time.sleep(10)
        honest = poa2.names(4)[c:]                     # signer1 (the RPC node) is always an insider
        h0 = _head()
        subprocess.run(["docker", "pause", *honest], capture_output=True, check=True)
        heads, t0 = [], time.time()
        try:
            while time.time() - t0 < window_s:
                heads.append((round(time.time() - t0, 1), _head()))
                time.sleep(1.0)
        finally:
            subprocess.run(["docker", "unpause", *honest], capture_output=True)
        sealed = heads[-1][1] - h0
        stalled_at = next((t for t, h in heads if h == heads[-1][1]), None) if sealed < window_s / 2 else None
        time.sleep(15)
        h1 = _head()
        time.sleep(10)
        resumed = _head() > h1
        row = {"coalition": c, "signers": 4, "window_s": window_s, "blocks_sealed_alone": sealed,
               "stalled_after_s": stalled_at, "chain_resumed_after_heal": resumed,
               "clique_limit": 4 // 2 + 1}
        print(json.dumps(row), flush=True)
        out.append(row)
    poa2.down()
    (RESULTS / "coalition.json").write_text(json.dumps(out, indent=1))
    return out


def outage(trials: int = 3) -> list[dict]:
    """A 300-record burst on the local PureChain client with validator 4 paused throughout."""
    from experiments.run import _trial
    from infra import purechain_local
    isolate("purechain_local")
    out = []
    for t in range(trials):
        purechain_local.ensure(4)
        victim = purechain_local.names(4)[-1]
        subprocess.run(["docker", "pause", victim], capture_output=True, check=True)
        try:
            s = _trial(dict(workload="E14", backend="purechain_local", n=300, doc_bytes=10 * 1024, level=4, rate=50,
                            trial=t, tag="outage", wait_timeout_s=180, observe_blocks=20, observe_s=90))
        finally:
            subprocess.run(["docker", "unpause", victim], capture_output=True)
        row = {k: s.get(k) for k in ("run_id", "committed", "success_rate", "e2e_latency_s_median", "e2e_latency_s_p95",
                                     "final_e2e_latency_s_median", "max_reorg_depth", "reorged_blocks")} if s else {}
        print(json.dumps(row), flush=True)
        out.append(row)
    (RESULTS / "outage.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "outage":
        outage()
    else:
        run(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
