# GPU rebuild operations (Strix RTX 5070)

Rebuilding a repository index is embedding-bound: every changed or rebuilt
chunk becomes an embedding request. The active endpoint is the pinned CUDA
MXFP4 service on Strix, not a Mac model and not the retired B550 dual-GPU
deployment.

## One command

```sh
sindexer-strix-rebuild /absolute/repo/path
sindexer-strix-rebuild /absolute/repo/path --update   # incremental instead
```

`sindexer-strix-rebuild` (source: `deploy/local/sindexer-strix-rebuild`)
fails closed unless it can prove, in order:

1. Strix is reachable through `strix-zt` and `sindexer-embed-gpu.service`
   is active.
2. Mac port 1236 reports the pinned Jina model, 1536 dimensions,
   `cuda-mxfp4`, GPU UUID 0, and the RTX 5070 Laptop GPU.
3. Local `milvus-standalone` is healthy.
4. A transient `sindexer-strix-keepawake.service` sleep inhibitor remains
   active for the indexing call.
5. The wrapper environment still points `EMBEDDING_URL` and `OPENAI_BASE_URL`
   at the loopback tunnel.

Full rebuild is the default. `--update` applies the same preflight to the
incremental path.

## Why a full rebuild is slower than an update

`sindexer index` rebuilds the whole codebase index from scratch; `sindexer
update` re-embeds only changed chunks and skips unchanged chunk IDs recorded
in the manifest. A manifest whose `inputs` no longer match the current
embedding identity forces exactly one full rebuild, then later updates return
to the cheap path.

## Proving the GPU served the run

Port 1236 alone is not proof of host identity. Check all three surfaces:

```sh
ssh strix-zt 'hostname && systemctl is-active sindexer-embed-gpu.service'
curl -sS http://127.0.0.1:1236/health
ssh strix-zt 'journalctl -u sindexer-embed-gpu.service --since "5 minutes ago" | grep -c "/v1/embeddings"'
```

The health payload must name the RTX 5070 Laptop GPU and `cuda-mxfp4`; the
Strix journal should show `POST /v1/embeddings` requests during a rebuild.

## Installing or repairing the endpoint

Follow `deploy/strix/README.md`. It installs the single systemd service, the
Mac `dev.resmp.sindexer-strix-embed-tunnel` LaunchAgent, and retires the old
B550-named Mac agent.

## Health and freshness

`sindexer-doctor` reports the wrapper, tunnel agent, endpoint model/runtime/
device/dimension, Milvus collections and container health, per-repo manifest
freshness, LM Studio fallback, and Strix reachability. The rebuild command
runs its own identity and Milvus preflight and never silently degrades to
lexical-only or a Mac model.

## Non-claims

This document does not claim a rebuild duration or that Strix's slower network
link affects embedding latency; measure both under the real workload. It also
does not claim B550 is unavailable for other compute: only its old sindexer
dual-GPU units are retired from this pipeline.
