"""E16: forks and finality of the released PureChain client under network delay between validators.

The live network cannot be delayed, so this is its controlled counterpart: the local PureChain network (released
purechain-geth 1.0.1 with Smart Auto-Mining v2, 4 signers) is created afresh for each delay level, and tc netem in a
sidecar sharing each validator's network namespace delays every packet a validator sends to another validator by d
(normal jitter of 10 % of d). Traffic between the host and the RPC node is not delayed, so submission and observation
are unchanged. Under each level, sustained per-record load keeps every validator sealing while the block watcher
records every block version (blocks.csv), as in the other trials.

    python -m experiments.netdelay [--delays 0 50 100 200 400] [--trials 3] [--n 6000] [--rate 30]

Each run directory gets netem.json (delay, jitter, RTT measured between validators before the trial).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

os.environ.setdefault("PURECHAIN_FINALITY_DEPTH", "8")

from finverity.backends import make_backend          # noqa: E402
from finverity.config import RESULTS                 # noqa: E402
from finverity.loadgen import TrialConfig, run_trial  # noqa: E402
from infra import isolate, purechain_local           # noqa: E402

IMAGE = "finverity-netem:1"      # alpine with iproute2 and iputils (built by build_image)
N = 4


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(RESULTS / "netdelay.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def build_image() -> None:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "-t", IMAGE, "-"], input="FROM alpine:latest\n"
                       "RUN apk add --no-cache iproute2 iputils\n", text=True, check=True, capture_output=True)


def ip(i: int) -> str:
    return f"{purechain_local.SUBNET}.{10 + i}"


def sidecar(container: str, *cmd: str, check: bool = True) -> str:
    r = subprocess.run(["docker", "run", "--rm", "--net", f"container:{container}", "--cap-add", "NET_ADMIN", IMAGE,
                        *cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{container} {' '.join(cmd)}: {r.stderr}")
    return r.stdout


def apply_delay(d_ms: float) -> None:
    """Delay packets from each validator to the other validators only (prio band 3 + u32 filters per peer)."""
    for i, c in enumerate(purechain_local.names(N), start=1):
        sidecar(c, "tc", "qdisc", "del", "dev", "eth0", "root", check=False)
        if d_ms <= 0:
            continue
        sidecar(c, "tc", "qdisc", "add", "dev", "eth0", "root", "handle", "1:", "prio")
        sidecar(c, "tc", "qdisc", "add", "dev", "eth0", "parent", "1:3", "handle", "30:", "netem", "delay",
                f"{d_ms:g}ms", f"{0.1 * d_ms:g}ms", "distribution", "normal")
        for j in range(1, N + 1):
            if j != i:
                sidecar(c, "tc", "filter", "add", "dev", "eth0", "protocol", "ip", "parent", "1:0", "prio", "1", "u32",
                        "match", "ip", "dst", f"{ip(j)}/32", "flowid", "1:3")


def clear_delay() -> None:
    for c in purechain_local.names(N):
        sidecar(c, "tc", "qdisc", "del", "dev", "eth0", "root", check=False)


def rtt_ms() -> dict:
    """Mean ping RTT between validator pairs (1->2, 2->3, 3->4, 4->1) and from validator 1 to the host gateway."""
    out = {}
    for i in range(1, N + 1):
        j = i % N + 1
        txt = sidecar(purechain_local.names(N)[i - 1], "ping", "-c", "5", "-i", "0.2", "-q", ip(j), check=False)
        m = re.search(r"= [\d.]+/([\d.]+)/", txt)
        out[f"{i}-{j}"] = float(m.group(1)) if m else None
    txt = sidecar(purechain_local.names(N)[0], "ping", "-c", "5", "-i", "0.2", "-q", f"{purechain_local.SUBNET}.1",
                  check=False)
    m = re.search(r"= [\d.]+/([\d.]+)/", txt)
    out["1-gateway"] = float(m.group(1)) if m else None
    return out


def heads() -> list[int]:
    out = []
    for c in purechain_local.names(N):
        r = subprocess.run(["docker", "exec", c, f"/pc/{purechain_local.BINARY}", "attach", "--exec", "eth.blockNumber",
                            "/data/geth.ipc"], capture_output=True, text=True)
        out.append(int(r.stdout.strip()) if r.stdout.strip().isdigit() else -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", type=float, nargs="+", default=[0, 50, 100, 200, 400])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--rate", type=float, default=30)
    a = ap.parse_args()
    build_image()
    isolate("purechain_local")
    try:
        for d in a.delays:
            tag = f"nd{d:g}"
            todo = [t for t in range(a.trials) if not (RESULTS / "raw" / TrialConfig(
                workload="E16", backend="purechain_local", n=a.n, doc_bytes=10 * 1024, level=4, rate=a.rate,
                trial=t, tag=tag).run_id / "summary.json").exists()]
            if not todo:
                continue
            log(f"== delay {d:g} ms: fresh network, trials {todo}")
            purechain_local.up(N)
            apply_delay(d)
            for t in todo:
                net = {"delay_ms": d, "jitter_ms": 0.1 * d, "rtt_ms": rtt_ms(), "heads_before": heads()}
                cfg = TrialConfig(workload="E16", backend="purechain_local", n=a.n, doc_bytes=10 * 1024, level=4,
                                  rate=a.rate, trial=t, tag=tag, verify_n=200, wait_timeout_s=600)
                bk = make_backend("purechain_local", validators=N)
                s = run_trial(cfg, bk, log=log)
                time.sleep(20)                         # let the idle window close before comparing heads
                net["heads_after"] = heads()
                (RESULTS / "raw" / cfg.run_id / "netem.json").write_text(json.dumps(net, indent=1))
                log(f"[{cfg.run_id}] committed {s.get('committed')}/{a.n}, reorged {s.get('reorged_blocks')}, "
                    f"max depth {s.get('max_reorg_depth')}, rtt {net['rtt_ms']}, heads {net['heads_after']}")
                if len(set(net["heads_after"])) > 1 or s.get("committed", 0) == 0:
                    log(f"validators disagree or nothing committed: recreating the network for the next trial")
                    purechain_local.up(N)
                    apply_delay(d)
    finally:
        try:
            clear_delay()
        finally:
            purechain_local.down()


if __name__ == "__main__":
    main()
