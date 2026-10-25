# Operations log (append-only)

Dated entries for the sindexer deployment. Newest last.

## 2026-10-24 — vector store moved to Strix; Mac stack retired to cold backup; B550 stack deprecated

Agent: ZCode (GLM-5.3-Flash), interactive session.

What was done:

- Surveyed the three-machine indexing topology live: Mac local Milvus
  (OrbStack compose at `~/milvus`, milvusdb/milvus:v2.5.3, 49 collections,
  17 GB volumes, the authoritative serving store on 127.0.0.1:19530);
  Strix embedding service (systemd `sindexer-embed-gpu.service`, RTX 5070
  Laptop GPU, cuda-mxfp4, `/home/kearm/services/rust-indexer`, verified
  through the embed tunnel); B550 orphan stack
  (`/home/kearm/sindexer-stack`, milvusdb/milvus:v2.6.4, one stale
  collection `rust_sindexer_b403215b705834f3` with 1,583 rows, plus a
  disabled pre-Strix embedding service `sindexer-embed-gpu@.service` and
  `llama-embedding.service`).
- Backed up the authoritative Mac store with milvus-backup v0.6.1:
  backup `mac_full_20261024`, 49 collections, 8,268,366,741 bytes of
  backup data, server-side copy inside the Mac MinIO (`a-bucket`,
  rootPath `backup`), created in 15.3 s. Backup warnings noted: RPC
  channel position not backed up (normal instance, CDC-only concern) and
  dynamic-field meta not recorded (no etcd client; `--backup_index_extra`
  would need etcd exposure — restore was verified end-to-end instead).
- Stood up Milvus on Strix at `/home/kearm/sindexer-stack`:
  `milvusdb/milvus:v2.5.3` + `quay.io/coreos/etcd:v3.5.16` (both pulled
  amd64 natively on Strix) and object store
  `bitnamilegacy/minio:2024.12.18` (substitution required: docker.io
  denies the `minio/minio` repository to this household IP for every
  tag — verified via docker pull, clean-config anonymous pull, and
  skopeo 1.24.1; dl.min.io returns 410 for all archives; quay.io,
  public.ecr.aws, and mirror.gcr.io do not carry the tag). All binds
  are loopback-only. `milvus-standalone` reached `healthy` and serves
  the Milvus v2 REST API on 127.0.0.1:19530.
- Added the Mac-side tunnel pair: `deploy/strix/launchd/
  sindexer-strix-milvus-tunnel` + plist
  `dev.resmp.sindexer-strix-milvus-tunnel` forwarding Mac
  127.0.0.1:19530 -> Strix 19530 (companion to the 1236 embed tunnel).
- Updated `deploy/local/sindexer-doctor` (milvus_container now reports
  the tunnel LaunchAgent state instead of local docker container
  health) and `deploy/local/sindexer-strix-rebuild` (ensure_milvus now
  opens the Strix tunnel instead of starting local OrbStack
  containers), both with syntax checks.

Non-claims and open items (updated as work completes):

- The restore of `mac_full_20261024` into the Strix store, the Mac
  cutover, end-to-end search verification, and the B550 backup /
  decommission are tracked in this session; see the final receipt for
  their outcome.
- Strix and B550 both intermittently reject NEW TCP connections over
  ZeroTier with ICMP destination-unreachable while established
  sessions keep flowing (verified by ICMP receive-counter deltas: one
  unreachable per attempt, and by the embed tunnel surviving every
  wedge). Wedge windows observed 11:04-11:26 and from ~12:10 onward.
  All retries during this work stayed at <=1 new connection per 10
  minutes per host to avoid feeding any connection-rate limiter.
- The Mac's local `~/milvus` stack data (17 GB volumes) is preserved
  in place as a cold backup; containers are stopped with restart
  policy `no` at cutover. B550's stack is preserved untouched until
  its single collection is backed up and the box is decommissioned.

Overnight blocker (2026-10-24 ~12:10 through 2026-10-25 09:00): both
compute boxes reject every NEW TCP connection over ZeroTier with an
immediate ICMP destination-unreachable (verified by receive-counter
delta: exactly one per attempt) while ESTABLISHED sessions keep
flowing — the Mac->Strix embed tunnel (connected 2026-10-24 11:07) has
stayed alive across the entire block, including a NAT re-derivation
on the boxes' WAN (peer UDP ports changed). Clean single attempts
after a 7.5-hour probe-free window (09:00 2026-10-25) also failed, so
a ban timer is ruled out; the block is structural at the boxes
(firewall REJECT rule or a wedged ZeroTier client on each box; both
boxes share fleet config management). All Mac-side lanes are
exhausted: ZeroTier overlay (new sessions refused), LAN 192.168.1.x
(separate subnet, unrouted), relays (no client-side control), GCP
fleet box (address unknown, loc-infra records no infra role),
mosh/b550-jump (need SSH first). Remaining paths need a human at the
boxes (inspect crowdsec/ufw/nftables, restart zerotier) or a
user-authorized restart of the Mac's ZeroTier daemon (root). The
restore watcher stays armed: it fires automatically when any Strix
tunnel attempt succeeds.

Repo note (2026-10-25): this checkout moved from
`~/AlphaHENG/contrib/rust_sindexer` to `~/rust_sindexer` mid-task by
the parallel branch-consolidation process (PR #22 merge); the deploy
commit was re-applied and amended at the new root
(`fb2f7b0` on `docs/agents-fullscale-setup`).
