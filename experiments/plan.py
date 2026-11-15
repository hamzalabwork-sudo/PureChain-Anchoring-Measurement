"""Experimental design (methodology section "Experimental Design").

E1  Controlled workloads W1-W5 on all three configurations      -> RQ2, RQ3, RQ4
E2  Offered-load sweep (saturation throughput, latency knee)     -> RQ2, RQ3
E3  Evidence-size sweep (10 KB .. 5 MB)                          -> RQ3, RQ4
E4  Evidence-complexity sweep (levels 1-4)                       -> RQ1, RQ3
E5  Validator-count sweep (Besu QBFT 4/7/10; PureChain fixed n=4)-> RQ3
E6  Integrity / tamper-detection experiment                      -> RQ1, RQ5
E7  Cross-dataset replication (W1, W2, W5 on SAML-D and IBM AML)  -> RQ2-RQ4
E8  Insider tampering (privileged operator rewrites commitments)  -> RQ1, RQ5
E9  Batched Merkle commitments (k = 1..100)                       -> RQ2-RQ4
"""
from __future__ import annotations

from finverity.config import EVIDENCE_SIZES as S

BACKENDS = ["purechain", "besu", "db_postgres"]
PUBLIC = "ethpos"
EMU = "poa2_local"  # PoA2 emulation calibrated to live PureChain (infra/poa2.py)   # conventional public-chain configuration (Ethereum PoS, mainnet timing)
BASE_RATE = 50.0

WORKLOADS = {
    #       n        evidence        trials  purpose
    "W1": (1_000, S["small"], 30),   # baseline
    "W2": (10_000, S["small"], 10),  # sustained workload
    "W3": (50_000, S["medium"], 5),  # scalability
    "W4": (100_000, S["medium"], 3), # high-volume
    "W5": (10_000, S["large"], 10),  # evidence-processing overhead
}


def plan(scale: str = "full") -> dict[str, list[dict]]:
    """Return experiment -> list of TrialConfig kwargs. scale='quick' is a smoke test."""
    q = scale == "quick"
    n_ = (lambda n: max(50, n // 100)) if q else (lambda n: n)
    t_ = (lambda t: min(t, 2)) if q else (lambda t: t)
    E = {k: [] for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E11", "E12")}

    for w, (n, d, trials) in WORKLOADS.items():
        for b in BACKENDS:
            for t in range(t_(trials)):
                E["E1"].append(dict(workload=w, backend=b, n=n_(n), doc_bytes=d, level=4, rate=BASE_RATE, trial=t))

    for w, trials in (("W1", 5), ("W2", 2)):
        n, d, _ = WORKLOADS[w]
        for t in range(t_(trials)):
            E["E1"].append(dict(workload=w, backend=PUBLIC, n=n_(n), doc_bytes=d, level=4, rate=BASE_RATE, trial=t,
                                wait_timeout_s=3600))
    for rate in [10, 50]:
        for t in range(t_(2)):
            E["E2"].append(dict(workload="E2", backend=PUBLIC, n=n_(max(300, rate * 60)), doc_bytes=S["small"],
                                level=4, rate=rate, trial=t, wait_timeout_s=3600))

    for rate in [10, 25, 50, 100, 200, 400, 0]:
        for b in BACKENDS:
            for t in range(t_(5)):
                n = n_(max(2_000, int(rate * 30)) if rate else 5_000)
                E["E2"].append(dict(workload="E2", backend=b, n=n, doc_bytes=S["small"], level=4, rate=rate, trial=t))

    for size in ["none", "small", "medium", "large", "xlarge"]:
        for b in BACKENDS:
            for t in range(t_(5)):
                E["E3"].append(dict(workload="E3", backend=b, n=n_(1_000), doc_bytes=S[size], level=4,
                                    rate=BASE_RATE, trial=t, tag=size))

    for level in [1, 2, 3, 4]:
        for b in BACKENDS:
            for t in range(t_(5)):
                E["E4"].append(dict(workload="E4", backend=b, n=n_(1_000), doc_bytes=S["medium"], level=level,
                                    rate=BASE_RATE, trial=t))

    for v in [4, 7, 10]:
        for rate in [50, 200]:
            for t in range(t_(5)):
                E["E5"].append(dict(workload="E5", backend="besu", n=n_(3_000), doc_bytes=S["small"], level=4,
                                    rate=rate, trial=t, tag=f"v{v}", validators=v))

    for src in ["momtsim", "saml_d", "ibm_aml"]:
        for t in range(t_(3)):
            E["E6"].append(dict(workload="E6", backend="purechain", source=src, n=n_(1_000), doc_bytes=S["medium"],
                                level=4, rate=BASE_RATE, trial=t, keep_store=True, verify_n=n_(1_000)))

    # E7 cross-dataset replication: the core workloads repeated on the two secondary datasets
    # (MoMTSim results come from E1), plus the insider experiment on every dataset.
    for src in ["saml_d", "ibm_aml"]:
        for w, trials in (("W1", 10), ("W2", 5), ("W5", 3)):
            n, d, _ = WORKLOADS[w]
            for b in BACKENDS:
                for t in range(t_(trials)):
                    E["E7"].append(dict(workload="E7", backend=b, source=src, n=n_(n), doc_bytes=d, level=4,
                                        rate=BASE_RATE, trial=t, tag=w))
        for b in BACKENDS:
            for t in range(t_(2)):
                E["E8"].append(dict(workload="E8", backend=b, source=src, n=n_(300), doc_bytes=S["small"], level=4,
                                    rate=20, trial=t, verify_n=n_(300), insider=n_(100)))
    for b in BACKENDS + [PUBLIC]:
        for t in range(t_(3 if b != PUBLIC else 2)):
            E["E8"].append(dict(workload="E8", backend=b, n=n_(300), doc_bytes=S["small"], level=4, rate=20,
                                trial=t, verify_n=n_(300), insider=n_(100)))

    for k in [1, 50]:
        for t in range(t_(2)):
            E["E9"].append(dict(workload="E9", backend=PUBLIC, n=n_(3_000), doc_bytes=S["small"], level=4, rate=50,
                                trial=t, batch_size=k, batch_wait_s=1.0, wait_timeout_s=3600))
    for k in [1, 10, 25, 50, 100]:
        for rate in [50, 200]:
            for b in BACKENDS:
                for t in range(t_(5)):
                    E["E9"].append(dict(workload="E9", backend=b, n=n_(max(3_000, rate * 30)), doc_bytes=S["small"],
                                        level=4, rate=rate, trial=t, batch_size=k, batch_wait_s=1.0))
    # E5: the emulation can also vary its signer count (live PureChain is fixed at 4 validators)
    for v in [4, 7, 10]:
        for rate in [50, 200]:
            for t in range(t_(5)):
                E["E5"].append(dict(workload="E5", backend=EMU, n=n_(3_000), doc_bytes=S["small"], level=4,
                                    rate=rate, trial=t, tag=f"v{v}", validators=v))
    # E11 burst-aware anchoring: a day of real arrivals (business-hours SAML-D, batch-posting IBM AML) replayed at
    # a mean of 30 records/s, recorded per record, in fixed Merkle batches, and by the adaptive controller.
    # Trial t uses the same window under every policy and backend (paired design).
    policies = {"record": dict(batch_size=1),
                "fixed": dict(batch_size=50, batch_wait_s=1.0),
                "adaptive": dict(batch_policy="adaptive", window=16, batch_wait_s=1.0)}
    for src in ["saml_d", "ibm_aml"]:
        for pol, kw in policies.items():
            for b in ["purechain", "besu"] + (["db_postgres"] if pol == "record" else []):
                for t in range(t_(5)):
                    E["E11"].append(dict(workload="E11", backend=b, source=src, n=n_(10_000), doc_bytes=S["small"],
                                         level=4, rate=30, arrival="trace", trial=t, **kw))
    # E12 finality under on-demand sealing: a 300-record burst, then silence. PureChain keeps sealing a short
    # tail of blocks; Clique with period 0 seals only while transactions are pending (finality starvation).
    # The block log follows the chain 20 blocks past the last record (or 90 s), deeper than any confirmation depth.
    # A lone record sent after the chain has been idle for a minute tests the tail where no burst precedes it.
    obs = dict(observe_blocks=20, observe_s=90)
    lone = dict(n=1, rate=1, idle_before_s=60, observe_blocks=20, observe_s=180)
    for t in range(t_(5)):
        for b in ["purechain", "besu"]:
            E["E12"].append(dict(workload="E12", backend=b, n=300, doc_bytes=S["small"], level=4, rate=50, trial=t,
                                 **obs))
        E["E12"].append(dict(workload="E12", backend=EMU, n=300, doc_bytes=S["small"], level=4, rate=50, trial=t,
                             tag="p0", period=0, wait_timeout_s=120, **obs))
        E["E12"].append(dict(workload="E12", backend="purechain", doc_bytes=S["small"], level=4, trial=t,
                             tag="lone", **lone))
        E["E12"].append(dict(workload="E12", backend=EMU, doc_bytes=S["small"], level=4, trial=t,
                             tag="p0_lone", period=0, wait_timeout_s=120, **lone))
    # ---------------------------------------------------------------- revision experiments (local ledgers only)
    E.update({"E13": [], "E14": [], "E8v2": [], "E6v2": [], "E9v2": []})
    # E9v2 gas of the revised (write-once per record ID) registry by batch size
    for k in [1, 10, 25, 50, 100]:
        E["E9v2"].append(dict(workload="E9", backend=EMU, n=n_(1_000), doc_bytes=S["small"], level=4, rate=200,
                              trial=0, batch_size=k, batch_wait_s=1.0, tag="v2"))
        if k > 1:   # cheaper alternative: record IDs only logged, uniqueness detected by the auditor
            E["E9v2"].append(dict(workload="E9", backend=EMU, n=n_(1_000), doc_bytes=S["small"], level=4, rate=200,
                                  trial=0, batch_size=k, batch_wait_s=1.0, tag="v2", batch_ids="log"))
    # E13 policy baselines and utilisation sweep: the IBM AML batch-posting trace (bursty) replayed for ~90 s at
    # mean rates from 5 to 120 records/s under every submission policy (one trial each; latencies are bootstrapped
    # over records). The fixed-batch timeout is swept at 30 records/s.
    sweep_pol = {"record": dict(batch_size=1),
                 "fixed_w1": dict(batch_size=50, batch_wait_s=1.0),
                 "fixed_w0.25": dict(batch_size=50, batch_wait_s=0.25),
                 "interval": dict(batch_policy="interval", batch_wait_s=1.0),
                 "drain": dict(batch_policy="drain"),
                 "adaptive": dict(batch_policy="adaptive", window=16, batch_wait_s=1.0),
                 "adaptive_k1": dict(batch_policy="adaptive", window=16, batch_wait_s=1.0, kmax=1)}
    for b, rates in (("poa2_local", (5, 15, 30, 60, 120)), ("besu", (30, 60)), ("purechain_local", (30,))):
        for rate in rates:
            pols = dict(sweep_pol)
            if rate == 30 and b != "purechain_local":
                pols.update({"fixed_w0.1": dict(batch_size=50, batch_wait_s=0.1),
                             "fixed_w0.5": dict(batch_size=50, batch_wait_s=0.5)})
            for pol, kw in pols.items():
                E["E13"].append(dict(workload="E13", backend=b, source="ibm_aml", n=n_(max(450, rate * 90)),
                                     doc_bytes=S["small"], level=4, rate=rate, arrival="trace", trial=0,
                                     verify_n=200, observe_blocks=10, observe_s=60, wait_timeout_s=600, **kw))
    # E14 finality after a burst with the client-side tail (Clique period 0) and on the local PureChain client
    # (released v1.0.1 with its own Smart Auto-Mining sealing): bursts and lone records, as E12.
    for t in range(t_(3)):
        for tag, extra in (("", {}), ("tail", dict(tail_k=8))):
            E["E14"].append(dict(workload="E14", backend=EMU, n=300, doc_bytes=S["small"], level=4, rate=50, trial=t,
                                 tag=("p0_" + tag).rstrip("_"), period=0, wait_timeout_s=120, **obs, **extra))
            E["E14"].append(dict(workload="E14", backend=EMU, doc_bytes=S["small"], level=4, trial=t,
                                 tag=("p0_lone_" + tag).rstrip("_"), period=0, wait_timeout_s=120, **lone, **extra))
        E["E14"].append(dict(workload="E14", backend="purechain_local", n=300, doc_bytes=S["small"], level=4, rate=50,
                             trial=t, wait_timeout_s=120, **obs))
        E["E14"].append(dict(workload="E14", backend="purechain_local", doc_bytes=S["small"], level=4, trial=t,
                             tag="lone", wait_timeout_s=120, **lone))
    # E8v2 insider attacks against the revised registry (adds S4, duplicate record IDs across batches) and the
    # database with a published digest; E6v2 widened benign transformations and non-canonical inputs.
    for b in (EMU, "besu", "db_postgres", "purechain_local"):
        for t in range(t_(2)):
            E["E8v2"].append(dict(workload="E8", backend=b, n=n_(300), doc_bytes=S["small"], level=4, rate=20,
                                  trial=t, verify_n=n_(300), insider=n_(50), tag="v2"))
    for src in ("momtsim", "saml_d", "ibm_aml"):
        E["E6v2"].append(dict(workload="E6", backend=EMU, source=src, n=n_(500), doc_bytes=S["medium"], level=4,
                              rate=BASE_RATE, trial=0, keep_store=True, verify_n=n_(500), tag="v2"))
    # every live-PureChain trial gets an identical twin on the PoA2 emulation
    for k in E:
        # (a fixed-period emulation never goes idle, so its twins do not wait for idleness)
        E[k] += [{**{a: v for a, v in c.items() if a != "idle_before_s"}, "backend": EMU}
                 for c in E[k] if c["backend"] == "purechain"]
    # live sustained capacity is below the offered load of the long workloads: wait for every receipt so that
    # no backlog spills into the next trial (the emulation twins never build a backlog)
    for k in E:
        for c in E[k]:
            if c["backend"] == "purechain":
                c.setdefault("wait_timeout_s", 3600)
    # the live network is shared: once unpaced per-record load beyond its ~35 records/s capacity was seen to
    # strand transactions in validator pools (W2 trial 3), the configurations that offer far more than that
    # for minutes on end run on the emulation only
    for k in E:
        E[k] = [c for c in E[k] if c["backend"] != "purechain" or _live_ok(c)]
    return E


def _live_ok(c: dict) -> bool:
    w, rate, k = c["workload"], c["rate"], c.get("batch_size", 1)
    if w in ("W3", "W4", "W5") or (w == "W2" and c["trial"] >= 3):
        return False
    if w == "E2" and (rate == 0 or rate > 50):
        return False
    if w == "E7" and c.get("tag") != "W1":
        return False
    if w == "E9" and k == 1 and rate > 50:
        return False
    return True
