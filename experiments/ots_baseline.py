"""OpenTimestamps baseline (B8): anchor the same record commitments in Bitcoin through the public OpenTimestamps
calendars, and measure latency to a Bitcoin attestation and to k confirmations, proof size and cost.

Records are the evidence packages of the PureChain trials (MoMTSim, level 4, 10 KB documents) and each commitment is
the registry's Merkle leaf (record ID, evidence hash, metadata hash). Two modes: one stamp per record, and one stamp
per batch over the Merkle root of the registry's tree. A stamp follows the OpenTimestamps client: a random nonce is
appended to the digest, the result hashed and submitted to the four default aggregators, and at least two must answer.
Upgrading asks each calendar for the Bitcoin attestation of its pending commitment; verification recomputes the
attested message and compares it with the block's Merkle root read from a public block explorer.

    python -m experiments.ots_baseline stamp   [--records 20] [--batches 5] [--batch-size 1000]
    python -m experiments.ots_baseline upgrade RUN_DIR [--every 600] [--hours 24]
    python -m experiments.ots_baseline verify  RUN_DIR [--k 6]

Writes results/ots/<run>/: stamps.csv (submission), proofs/*.ots, upgrades.csv, verified.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd
from opentimestamps.calendar import DEFAULT_AGGREGATORS, RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import BytesDeserializationContext, BytesSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

from finverity import accounts, datasets, evidence, merkle
from finverity.canonical import sha256
from finverity.config import RESULTS

OUT = RESULTS / "ots"
EXPLORER = "https://blockstream.info/api"
M_OF_N = 2
TIMEOUT = 15.0


def records(n: int, seed: int, tag: str) -> list[bytes]:
    """Registry Merkle leaves of n evidence packages, as in the PureChain trials."""
    approvers = [a.address for a in accounts.roles(16).approvers]
    out = []
    for tx in datasets.load_workload("momtsim", n, seed=seed):
        b = evidence.generate(tx, 4, 10 * 1024, seed=seed, approver_addresses=approvers, run_id=tag)
        rid = sha256(f"{tag}|{tx['tx_id']}".encode())
        out.append(merkle.leaf(rid, b.evidence_hash, b.meta_hash))
    return out


def stamp(digest: bytes) -> tuple[DetachedTimestampFile, dict]:
    """Stamp one 32-byte commitment as the client does: nonce, SHA-256, m-of-n calendar submission."""
    dtf = DetachedTimestampFile(OpSHA256(), Timestamp(hashlib.sha256(digest).digest()))
    tip = dtf.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    results, t0 = {}, time.time()

    def submit(url):
        t = time.time()
        try:
            results[url] = (RemoteCalendar(url).submit(tip.msg, timeout=TIMEOUT), time.time() - t)
        except Exception as e:   # one calendar down is tolerated
            results[url] = (e, time.time() - t)

    threads = [threading.Thread(target=submit, args=(u,)) for u in DEFAULT_AGGREGATORS]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ok = {u: r for u, r in results.items() if isinstance(r[0], Timestamp)}
    for ts, _ in ok.values():
        tip.merge(ts)
    lat = sorted(s for _, s in ok.values())
    info = {"t_submit": t0, "calendars_ok": len(ok), "stamp_s": lat[M_OF_N - 1] if len(lat) >= M_OF_N else None,
            "stamp_all_s": time.time() - t0, "failed": ";".join(f"{u}:{type(r[0]).__name__}"
                                                                    for u, r in results.items() if u not in ok)}
    return dtf, info


def save(dtf: DetachedTimestampFile, path: Path) -> int:
    ctx = BytesSerializationContext()
    dtf.serialize(ctx)
    data = ctx.getbytes()
    path.write_bytes(data)
    return len(data)


def load(path: Path) -> DetachedTimestampFile:
    return DetachedTimestampFile.deserialize(BytesDeserializationContext(path.read_bytes()))


def cmd_stamp(a) -> None:
    run = OUT / time.strftime("run_%Y%m%d_%H%M%S")
    (run / "proofs").mkdir(parents=True)
    rows = []
    for i, leaf in enumerate(records(a.records, a.seed, "B8_record")):
        dtf, info = stamp(leaf)
        rows.append({"id": f"r{i:03d}", "mode": "record", "records": 1, "digest": leaf.hex(), **info,
                     "proof_bytes": save(dtf, run / "proofs" / f"r{i:03d}.ots")})
        print(rows[-1]["id"], info["calendars_ok"], f"{info['stamp_s']}")
        time.sleep(a.gap)
    for j in range(a.batches):
        leaves = records(a.batch_size, a.seed + 1 + j, f"B8_batch{j}")
        root, _ = merkle.build(leaves)
        dtf, info = stamp(root)
        rows.append({"id": f"b{j:03d}", "mode": "batch", "records": len(leaves), "digest": root.hex(), **info,
                     "proof_bytes": save(dtf, run / "proofs" / f"b{j:03d}.ots")})
        print(rows[-1]["id"], info["calendars_ok"], f"{info['stamp_s']}")
        time.sleep(a.gap)
    pd.DataFrame(rows).to_csv(run / "stamps.csv", index=False)
    (run / "config.json").write_text(json.dumps(vars(a) | {"aggregators": list(DEFAULT_AGGREGATORS)}, default=str))
    print(run)


def upgrade(dtf: DetachedTimestampFile) -> bool:
    """Ask each calendar for its pending commitments; True once a Bitcoin attestation is present."""
    def walk(stamp):
        if stamp.attestations:
            yield stamp
        for sub in stamp.ops.values():
            yield from walk(sub)

    for sub in list(walk(dtf.timestamp)):
        for att in list(sub.attestations):
            if isinstance(att, PendingAttestation):
                try:
                    sub.merge(RemoteCalendar(att.uri).get_timestamp(sub.msg, timeout=TIMEOUT))
                except Exception:   # not yet committed, or calendar unreachable
                    pass
    return any(isinstance(att, BitcoinBlockHeaderAttestation) for _, att in dtf.timestamp.all_attestations())


def cmd_upgrade(a) -> None:
    run, end = Path(a.run), time.time() + 3600 * a.hours
    st = pd.read_csv(run / "stamps.csv")
    up = run / "upgrades.csv"
    done = set(pd.read_csv(up)["id"]) if up.exists() else set()
    while time.time() < end and len(done) < len(st):
        for pid in st["id"]:
            if pid in done:
                continue
            p = run / "proofs" / f"{pid}.ots"
            dtf = load(p)
            if upgrade(dtf):
                size = save(dtf, p)
                heights = sorted({att.height for _, att in dtf.timestamp.all_attestations()
                                  if isinstance(att, BitcoinBlockHeaderAttestation)})
                row = pd.DataFrame([{"id": pid, "t_upgraded": time.time(), "height": heights[0],
                                     "heights": ";".join(map(str, heights)), "proof_bytes_complete": size}])
                row.to_csv(up, mode="a", header=not up.exists(), index=False)
                done.add(pid)
                print(pid, "attested at", heights[0], flush=True)
        if len(done) < len(st):
            time.sleep(a.every)


def block(height: int) -> dict:
    get = lambda path: urllib.request.urlopen(f"{EXPLORER}/{path}", timeout=20).read().decode()
    return json.loads(get(f"block/{get(f'block-height/{height}').strip()}"))


def cmd_verify(a) -> None:
    run = Path(a.run)
    st = pd.read_csv(run / "stamps.csv").set_index("id")
    up = pd.read_csv(run / "upgrades.csv")
    rows, cache = [], {}
    for _, u in up.iterrows():
        dtf = load(run / "proofs" / f"{u['id']}.ots")
        digest = bytes.fromhex(st.loc[u["id"], "digest"])
        ok_digest = dtf.timestamp.msg == hashlib.sha256(digest).digest()
        for msg, att in dtf.timestamp.all_attestations():
            if not isinstance(att, BitcoinBlockHeaderAttestation):
                continue
            for h in (att.height, att.height + a.k - 1):
                cache.setdefault(h, block(h))
            b = cache[att.height]
            t0 = st.loc[u["id"], "t_submit"]
            rows.append({"id": u["id"], "mode": st.loc[u["id"], "mode"], "records": st.loc[u["id"], "records"],
                         "height": att.height, "verified": ok_digest and msg[::-1].hex() == b["merkle_root"],
                         "block_time": b["timestamp"], "inclusion_s": b["timestamp"] - t0,
                         f"conf{a.k}_s": cache[att.height + a.k - 1]["timestamp"] - t0,
                         "observed_s": u["t_upgraded"] - t0, "proof_bytes": u["proof_bytes_complete"]})
    df = pd.DataFrame(rows).sort_values(["id", "height"]).drop_duplicates("id")   # earliest attestation per stamp
    df.to_csv(run / "verified.csv", index=False)
    print(df.groupby("mode")[["inclusion_s", f"conf{a.k}_s", "proof_bytes"]].describe().T.to_string())
    print("all verified:", bool(df["verified"].all()), len(df), "of", len(st))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stamp")
    s.add_argument("--records", type=int, default=20)
    s.add_argument("--batches", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=1000)
    s.add_argument("--seed", type=int, default=8)
    s.add_argument("--gap", type=float, default=2.0)
    u = sp.add_parser("upgrade")
    u.add_argument("run")
    u.add_argument("--every", type=float, default=600)
    u.add_argument("--hours", type=float, default=24)
    v = sp.add_parser("verify")
    v.add_argument("run")
    v.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    {"stamp": cmd_stamp, "upgrade": cmd_upgrade, "verify": cmd_verify}[a.cmd](a)
