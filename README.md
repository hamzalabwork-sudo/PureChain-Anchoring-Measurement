# PureChain anchoring measurements

**Anchoring audit evidence on a production permissioned ledger that seals blocks on demand: forks, finality tails,
capacity and insider attacks, measured on live PureChain, its released client, Clique, Besu QBFT and PostgreSQL.**

Audit evidence kept in systems administered by the audited organisation can be altered by a privileged insider.
The client in this repository anchors canonicalised commitments to the evidence (RFC 8785 profile, SHA-256, approver
signature) in a write-once registry contract; the evidence itself stays with the organisation. The measurements ask
how a production proof-of-authority ledger that seals only when there is traffic behaves as such an anchor:

- **Forks:** how deep the live network reorganises, why, and what confirmation depth an auditor needs.
- **Finality tails:** whether every record becomes final once traffic stops, and what decides it.
- **Capacity:** what an evidence submitter gets, what happens beyond it, and what submission policies buy under
  bursty arrivals.
- **Insider attacks:** what a write-once registry stops, compared with a database that publishes a digest.

---

## Results at a glance

All numbers are regenerated from `results/` by `python -m analysis.report` (see `paper/generated/macros.tex`).

| Finding | Measurement |
|---|---|
| Forks under load (client release 1.0.1) | 5.8 % of 14,372 heights replaced, reorganisations up to 7 blocks deep; a threshold tail model puts the per-height risk beyond k = 8 confirmations at 7.5 × 10⁻⁶ (k = 11 for 10⁻⁶) |
| Forks without our traffic (E8, 6.3 days) | 0.35 % of 15,653 heights replaced, all one block deep, every replaced block sealed out of turn |
| Forks under network delay (E9, released client) | replaced heights rise from 0.6 % (no delay) to 28.3 % at 800 ms one-way delay, depth from 1 to 5 |
| Finality when traffic stops | the 15 s idle window of the sealing policy gave a tail of 17–20 blocks, but was missing after 2 of 86 idle periods; Clique with period 0 finalised 71.1 % of a burst; a client-side tail of at most k no-op transactions finalised every record |
| Capacity | 31.8 records/s on the live network with one commitment per record; beyond it, transactions were stranded |
| Bursts (replayed day traces, live) | 95th-percentile inclusion latency 91 s per record, 3.1 s with fixed batches of 50, 2.9 s with the inclusion-feedback controller |
| Insider attacks | every rewrite and backdated record rejected on chain; re-commitment of batched identifiers rejected (28,412 gas per record stored, 4,008 if only logged) |
| Public timestamping (B8, OpenTimestamps) | 121 stamps, all verified on Bitcoin: attestation after a median of 24 min, six confirmations after a median of 75 min (at most 2.9 h) |

---

## Repository map

```
contracts/            FinVerityRegistry.sol (solc 0.8.19, EVM paris): write-once records, Merkle batches with
                      per-identifier uniqueness, domain-separated approver signatures
finverity/            client: datasets, evidence, canonicalisation, commitment, backends, load generator
  loadgen.py          trial runner, arrival processes, submission policies (per record, fixed batches, fixed
                      interval, drain per block, inclusion-feedback controller), client-side tail
  backends/evm.py     pipelined submission, fork-aware block watcher, k-confirmation finality
experiments/          experiment plan (plan.py) and runner (run.py); idle.py, coalition.py, netdelay.py,
                      passive_forks.py, ots_baseline.py
infra/                Docker launchers: local PureChain network from the released client, Clique emulation,
                      Besu QBFT, Ethereum PoS devnet, PostgreSQL
analysis/             statistics, figures and the report builder (report.py, revision.py, passive.py, ots.py)
results/raw/          one folder per trial: summary.json, blocks.csv, verify/tamper/insider parquet
results/passive/out/  E8: hourly block logs of the live network (every block version seen)
results/ots/          B8: OpenTimestamps proofs (.ots), submission, upgrade and verification logs
results/idle/         chain scans; results/tables, results/figures: aggregated outputs
paper/generated/      generated LaTeX tables and macros; paper/figures: figures
scripts/get_data.py   dataset download with checksum verification
tests/                unit tests (canonicalisation, evidence, signatures, Merkle proofs, tampering)
```

---

## Experiment identifiers

The code uses the identifiers of the experiment plan; the write-up numbers experiments in order of appearance.

| Write-up | Code | Experiment |
|---|---|---|
| E1 | E11 | anchoring under replayed day traces, by submission policy and ledger |
| E2 | E13 | submission-policy baselines and utilisation sweep |
| E3 | E12 | finality when traffic stops: a 300-record burst, and a lone record after 60 s idle |
| E4 | E14 | client-side tail, and the released PureChain client run locally |
| E5 | E10 | idle behaviour of the live network (`experiments/idle.py`) |
| E6 | E6, E6v2 | tamper detection |
| E7 | E8, E8v2 | privileged-insider rewrites, backdating and re-commitment (S1–S4) |
| E8 | E15 | passive monitoring of the live network (`experiments/passive_forks.py`) |
| E9 | E16 | network-delay emulation on the local PureChain client (`experiments/netdelay.py`) |
| B1–B7 | E1–E5, E7, E9, E9v2 | infrastructure baselines: workloads, offered load, evidence size and complexity, validators, datasets, fixed batching |
| B8 | – | OpenTimestamps baseline (`experiments/ots_baseline.py`) |
| – | – | signer coalitions on the Clique emulation (`experiments/coalition.py`) |

---

## Reproducing the study

### 1. Environment

Python 3.13 and Docker (at least 12 CPUs and 16 GB assigned).

```bash
python -m venv .venv
pip install -r requirements.txt
pip install --no-deps purechainlib==2.1.7 web3==6.20.4
pytest -q
```

`requirements.lock.txt` pins the exact versions used for the published results. Live runs need `.env` (never commit
it):

```
PURECHAIN_RPC=https://purechainnode.com
PURECHAIN_CHAIN_ID=900520900520
PURECHAIN_PRIVATE_KEY=<hex key; PureChain has zero gas, the account needs no funds>
```

The local PureChain network (`infra/purechain_local.py`) runs the released `purechain-geth` 1.0.1 binary and its
unmodified Smart Auto-Mining v2 script (`autoMine-v2.js`), both from github.com/gieworld/purechain_testnet, placed in
`build/purechain/`; the launcher checks the binary's SHA-256 against the release. The binary is third-party and not
redistributed here; keys and chain state are generated locally and never committed.

### 2. Data

```bash
python scripts/get_data.py
python -c "from finverity import datasets as d; [d.preprocess(s) for s in d.SOURCES]"
```

MoMTSim (Mendeley Data, doi:10.17632/zhj366m53p.2, CC BY 4.0), SAML-D and IBM AML HI-Small (Kaggle). The datasets
are used as transaction workloads and arrival traces; they are not redistributed here.

### 3. Run

```bash
python -m experiments.run campaign --only E11 E12 E13 E14     # bursts, finality, policies, client-side tail
python -m experiments.idle scan                               # idle behaviour of the live network
python -m experiments.netdelay --delays 0 50 100 200 400 800 --trials 4 --n 9000 --rate 30
python experiments/passive_forks.py https://purechainnode.com results/passive/out --poll 0.5 --hours 168
python -m experiments.ots_baseline stamp                      # then: upgrade RUN_DIR, verify RUN_DIR
python -m experiments.coalition
```

Backends: `purechain` (live), `purechain_local`, `poa2_local` (Clique emulation), `besu`, `db_postgres`, `ethpos`.
The runner starts the required Docker network, stops the other local networks, and skips trials that already have a
`summary.json`, so a campaign can be interrupted and resumed. Live and local experiments must not run at the same
time. The passive monitor uses the standard library only and sends no transactions.

### 4. Analyse

```bash
python -m analysis.report      # statistics, figures, tables and macros -> results/, paper/generated, paper/figures
python -m analysis.passive     # E8 summary
python -m analysis.ots         # B8 summary
```

The per-record measurements (`items.parquet`) are attached to the `v1.0.0` GitHub release in two archives; extract
them into `results/raw/` to recompute per-record statistics.

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The datasets are not redistributed and remain under their original licenses.
