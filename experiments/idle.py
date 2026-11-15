"""E10  Demand-driven block production (PureChain-specific).

The production PureChain client (PureChain-Idle, v1.13.15-purechain-1.0.1) seals blocks only while there is
traffic: a transaction wakes the validators, a short tail of 1 s blocks follows, then sealing pauses. Fixed-period
chains (Besu QBFT, Clique, Ethereum PoS slots) keep producing empty blocks while no evidence arrives. Audit evidence
is bursty (close of business, month end), so the idle behaviour matters for ledger growth and validator energy.

    python -m experiments.idle scan  [--lo N] [--hi N]   # live PureChain block history -> blocks.csv + ledger.json
    python -m experiments.idle local [--minutes 10]      # idle block production and energy, Besu QBFT and PoA2 emulation
    python -m experiments.idle wake  [--reps 15]         # inclusion latency vs. time the live chain has been idle

Run it when no campaign is using the network: our own traffic keeps PureChain awake.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd

from finverity import accounts, datasets, evidence
from finverity.backends import make_backend
from finverity.backends.base import Item
from finverity.backends.evm import Rpc
from finverity.canonical import sha256
from finverity.config import BESU, POA2_LOCAL, PURECHAIN, RESULTS
from finverity.resources import ResourceMonitor

OUT = RESULTS / "idle"
RESUMED = 1_807_037           # first block after the September 2026 halt
SECONDS_PER_YEAR = 365.25 * 86400


def _blocks(rpc: Rpc, lo: int, hi: int) -> pd.DataFrame:
    rows = []
    for s in range(lo, hi + 1, 500):
        for b in rpc.batch([("eth_getBlockByNumber", [hex(h), False]) for h in range(s, min(s + 500, hi + 1))]):
            if b:
                rows.append({"number": int(b["number"], 16), "timestamp": int(b["timestamp"], 16),
                             "tx_count": len(b["transactions"]), "size": int(b["size"], 16),
                             "gas_used": int(b["gasUsed"], 16), "difficulty": int(b["difficulty"], 16)})
    return pd.DataFrame(rows)


# idle: no block for over 30 s. Under full load PureChain stretches its block interval up to 22 s (E10 scan),
# so shorter gaps are slow blocks, not idleness.
def ledger_stats(df: pd.DataFrame, idle_gap_s: int = 30) -> dict:
    """Block production of an observed chain segment against a fixed 1 s period over the same wall time."""
    span = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    gaps = df["timestamp"].diff().dropna()
    empty = df["tx_count"] == 0
    # wake tail: blocks sealed after a transaction-carrying block until sealing pauses
    tails, run = [], None
    for tx, g in zip(df["tx_count"].iloc[1:], gaps):
        if g > idle_gap_s:
            if run is not None:
                tails.append(run)
            run = None
        if tx > 0:
            run = 0
        elif run is not None:
            run += 1
    mean_empty = float(df.loc[empty, "size"].mean()) if empty.any() else float("nan")
    fixed = span / 1.0
    return {"blocks": len(df), "span_h": span / 3600, "tx": int(df["tx_count"].sum()),
            "blocks_per_h": len(df) / span * 3600, "empty_share": float(empty.mean()),
            "idle_share_of_time": float(gaps[gaps > idle_gap_s].sum() / span),
            "idle_periods": int((gaps > idle_gap_s).sum()), "idle_gap_median_s": float(gaps[gaps > idle_gap_s].median()),
            "wake_tail_blocks_median": float(np.median(tails)) if tails else float("nan"),
            "wake_periods": len(tails),
            "wake_tail_counts": {int(k): int(v) for k, v in zip(*np.unique(tails, return_counts=True))},
            "empty_block_bytes_mean": mean_empty, "bytes_per_h": float(df["size"].sum() / span * 3600),
            "fixed_1s_blocks_same_span": fixed, "blocks_avoided_share": 1 - len(df) / fixed,
            "empty_bytes_avoided_per_year_mb": (1 - len(df) / fixed) * SECONDS_PER_YEAR * mean_empty / 1e6}


def cmd_scan(args):
    rpc = Rpc(PURECHAIN.rpc_url)
    hi = args.hi or int(rpc.call("eth_blockNumber"), 16)
    df = _blocks(rpc, args.lo, hi)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "purechain_blocks.csv", index=False)
    out = {"client": rpc.call("web3_clientVersion"), "lo": args.lo, "hi": hi, **ledger_stats(df)}
    (OUT / "purechain_ledger.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_tails(args):
    """The tail after every idle period in a later block range (default: from the end of the E10 scan to the head)."""
    rpc = Rpc(PURECHAIN.rpc_url)
    lo = args.lo or json.loads((OUT / "purechain_ledger.json").read_text())["hi"] + 1
    hi = args.hi or int(rpc.call("eth_blockNumber"), 16)
    df = _blocks(rpc, lo, hi)
    st = ledger_stats(df)
    out = {"client": rpc.call("web3_clientVersion"), "lo": lo, "hi": hi, "span_h": st["span_h"],
           **{k: st[k] for k in ("wake_periods", "wake_tail_counts", "wake_tail_blocks_median")}}
    (OUT / "purechain_tails_recent.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_local(args):
    """Idle block production and node energy of the fixed-period local chains (no transactions sent)."""
    from infra import besu, isolate, poa2
    out = {}
    for name, chain, ensure, containers in (
            ("besu", BESU, lambda: besu.ensure_network(4), [f"besu-validator{i}" for i in range(1, 5)]),
            ("poa2_local", POA2_LOCAL, lambda: poa2.ensure(4), poa2.names(4))):
        isolate(name)
        ensure()
        rpc = Rpc(chain.rpc_url)
        time.sleep(30)                                   # settle after (re)start
        h0 = int(rpc.call("eth_blockNumber"), 16)
        mon = ResourceMonitor(containers)
        mon.start()
        time.sleep(args.minutes * 60)
        res = mon.stop()
        h1 = int(rpc.call("eth_blockNumber"), 16)
        df = _blocks(rpc, h0 + 1, h1)
        st = ledger_stats(df)
        out[name] = {"client": rpc.call("web3_clientVersion"), "window_s": res["elapsed_s"],
                     "blocks_per_h": st["blocks_per_h"], "empty_share": st["empty_share"],
                     "bytes_per_h": st["bytes_per_h"], "empty_block_bytes_mean": st["empty_block_bytes_mean"],
                     "infra_cpu_s": res["infra_cpu_s"], "infra_energy_j": res["infra_energy_j"],
                     "idle_energy_wh_per_day": res["infra_energy_j"] / res["elapsed_s"] * 86400 / 3600,
                     "infra_blk_write_bytes": res.get("infra_blk_write_bytes", 0)}
        print(name, json.dumps(out[name], indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_idle.json").write_text(json.dumps(out, indent=1))


class _Head:
    """Wall time at which the current head was first observed (block timestamps have 1 s resolution)."""

    def __init__(self, rpc: Rpc):
        self.rpc, self.n, self.t = rpc, -1, time.time()

    def poll(self) -> float:
        n = int(self.rpc.call("eth_blockNumber"), 16)
        if n != self.n:
            self.n, self.t = n, time.time()
        return time.time() - self.t


def cmd_wake(args):
    """One evidence record at a time, sent after the live chain has been idle for a target time."""
    bk = make_backend("purechain")
    rs = accounts.roles(16)
    bk.setup(rs.submitters, rs.approvers)
    targets = [float(x) for x in args.targets]
    order = [t for t in targets for _ in range(args.reps)]
    random.Random(args.seed).shuffle(order)           # interleave conditions so drift does not confound them
    txs = datasets.load_workload("momtsim", len(order), seed=args.seed)
    approvers = [a.address for a in rs.approvers]
    head, rows = _Head(bk.rpc), []
    for i, (target, tx) in enumerate(zip(order, txs)):
        b = evidence.generate(tx, 4, 10 * 1024, seed=args.seed, approver_addresses=approvers, run_id="E10_wake")
        it = Item(idx=i, tx_id=tx["tx_id"], record_id=sha256(f"E10|{time.time_ns()}|{tx['tx_id']}".encode()),
                  evidence_hash=b.evidence_hash, meta_hash=b.meta_hash, level=4, approver_index=b.approver_index,
                  submitter_index=0, evidence_bytes=b.size_bytes)
        if i == 0:
            bk.calibrate_gas(it, rs.submitters[0], rs.approvers[it.approver_index])
        bk.prepare(it, rs.submitters[0], rs.approvers[it.approver_index])
        bk.start(1)
        end = time.time() + args.max_wait_s
        idle = head.poll()
        if target == 0:                                   # warm: send while the chain is still sealing
            while idle > 1.5 and time.time() < end:
                time.sleep(0.2); idle = head.poll()
        else:
            while idle < target and time.time() < end:
                time.sleep(0.25); idle = head.poll()
        bk.send(it, 0)
        bk.wait([it], 180)
        bk.stop()
        rows.append({"i": i, "target_idle_s": target, "idle_at_send_s": idle, "status": it.status,
                     "block": it.block, "commit_latency_s": it.t_commit - it.t_send,
                     "conf7_latency_s": it.conf.get(7, float("nan")) - it.t_send})
        print(json.dumps(rows[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wake.csv", index=False)
    summ = df[df["status"] == "committed"].groupby("target_idle_s")[["idle_at_send_s", "commit_latency_s",
                                                                     "conf7_latency_s"]].median()
    summ["n"] = df[df["status"] == "committed"].groupby("target_idle_s").size()
    (OUT / "wake_summary.json").write_text(summ.reset_index().to_json(orient="records", indent=1))
    print(summ)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experiments.idle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan"); p.add_argument("--lo", type=int, default=RESUMED); p.add_argument("--hi", type=int)
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("tails"); p.add_argument("--lo", type=int); p.add_argument("--hi", type=int)
    p.set_defaults(fn=cmd_tails)
    p = sub.add_parser("local"); p.add_argument("--minutes", type=float, default=10); p.set_defaults(fn=cmd_local)
    p = sub.add_parser("wake"); p.add_argument("--reps", type=int, default=15)
    p.add_argument("--targets", nargs="*", default=["0", "10", "30", "60", "120"])
    p.add_argument("--max-wait-s", type=float, default=900); p.add_argument("--seed", type=int, default=10)
    p.set_defaults(fn=cmd_wake)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
