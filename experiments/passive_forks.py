"""Passive fork monitor (E15): observe a live chain for days without sending any transaction.

Polls the head every POLL seconds, re-reads a short window below it and walks back along parent hashes whenever a new
block does not extend the locally known chain, like the trials' block watcher (finverity.backends.evm.BlockWatcher).
Every distinct block version is appended to hourly gzip JSON-lines files with the wall-clock time it was first seen
and the header fields needed to recover the Clique signer offline; the client version is logged every hour. Standard
library only, so it runs on any machine with Python 3.

    python experiments/passive_forks.py RPC_URL OUT_DIR [--poll 0.5] [--hours 168]

Analysis: python -m analysis.passive OUT_DIR (canonical chain from parent hashes, replaced heights, reorg depths).
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

WINDOW = 2          # heights re-read below the previous head on every poll
WALK = 64           # deepest walk back along parent hashes
HEADER = ("number", "hash", "parentHash", "timestamp", "miner", "difficulty", "extraData", "gasUsed", "gasLimit",
          "sha3Uncles", "stateRoot", "transactionsRoot", "receiptsRoot", "logsBloom", "mixHash", "nonce",
          "baseFeePerGas", "withdrawalsRoot", "blobGasUsed", "excessBlobGas", "parentBeaconBlockRoot", "size")


class Monitor:
    def __init__(self, url: str, out: Path, poll: float):
        self.url, self.out, self.poll = url, out, poll
        self.out.mkdir(parents=True, exist_ok=True)
        self.tip: dict[int, str] = {}       # latest known hash per height (pruned below head - 4 * WALK)
        self.seen: set[str] = set()         # block hashes already written
        self.head = self.start = int(self.call("eth_blockNumber"), 16)
        self.errors = 0

    def rpc(self, payload):
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def call(self, method, params=()):
        return self.rpc({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)})["result"]

    def blocks(self, heights):
        out = self.rpc([{"jsonrpc": "2.0", "id": i, "method": "eth_getBlockByNumber", "params": [hex(h), False]}
                        for i, h in enumerate(heights)])
        return [o.get("result") for o in sorted(out, key=lambda o: o["id"])]

    def write(self, rec: dict) -> None:
        f = self.out / time.strftime("blocks_%Y%m%d_%H.jsonl.gz", time.gmtime(rec["t_observed"]))
        with gzip.open(f, "at", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def record(self, b: dict, t_obs: float) -> None:
        num, h = int(b["number"], 16), b["hash"]
        self.tip[num] = h
        if h in self.seen:
            return
        self.seen.add(h)
        rec = {k: b[k] for k in HEADER if k in b}
        rec.update(t_observed=t_obs, tx_count=len(b.get("transactions", [])))
        self.write(rec)

    def poll_once(self) -> None:
        head = int(self.call("eth_blockNumber"), 16)
        lo = max(self.start + 1, min(self.head, head) - WINDOW)
        if head < lo:
            return
        t_obs = time.time()
        got = [b for b in self.blocks(range(lo, head + 1)) if b]
        for b in got:
            self.record(b, t_obs)
        if got:   # walk back while the lowest fetched block does not extend the known chain
            b = got[0]
            for _ in range(WALK):
                num = int(b["number"], 16)
                if num - 1 <= self.start or self.tip.get(num - 1) in (None, b["parentHash"]):
                    break
                (b,) = self.blocks([num - 1])
                if not b:
                    break
                self.record(b, t_obs)
        self.head = max(self.head, head)
        if len(self.tip) > 8 * WALK:
            for n in [n for n in self.tip if n < self.head - 4 * WALK]:
                del self.tip[n]

    def status(self) -> None:
        rec = {"t": time.time(), "head": self.head, "client": self.call("web3_clientVersion"),
               "peers": self.call("net_peerCount"), "errors": self.errors, "versions": len(self.seen)}
        with open(self.out / "status.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def run(self, hours: float) -> None:
        end, last_status = time.time() + 3600 * hours, 0.0
        while time.time() < end:
            t0 = time.time()
            try:
                self.poll_once()
                if t0 - last_status >= 3600:
                    self.status()
                    last_status = t0
            except Exception as e:   # network errors: count, back off, keep going
                self.errors += 1
                with open(self.out / "errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%F %T')} {type(e).__name__}: {e}\n")
                time.sleep(5)
            time.sleep(max(0.0, self.poll - (time.time() - t0)))
        self.status()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--hours", type=float, default=168)
    a = ap.parse_args()
    Monitor(a.url, a.out, a.poll).run(a.hours)
