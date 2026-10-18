# GPU rebuild operations (dual RTX 3090 Ti)

Rebuilding a repository index is embedding-bound: every chunk becomes one
`/v1/embeddings` request, so the deployment that serves that endpoint decides
rebuild wall-clock time. On this fleet that endpoint is B550, not the Mac.

## One command

```sh
sindexer-gpu-rebuild /absolute/repo/path
sindexer-gpu-rebuild /absolute/repo/path --update   # incremental instead
```

`sindexer-gpu-rebuild` (source: `deploy/local/sindexer-gpu-rebuild`) refuses to
start unless it can prove, in this order:

1. the Mac port 1236 embedding tunnel answers with dimension 1536,
2. both B550 `sindexer-embed-gpu@0` and `@1` systemd instances are active,
3. B550 nginx is active with the `sindexer_jina` upstream loaded,
4. the local `milvus-standalone` container reports healthy,
5. B550 wakes via its known Wake-on-LAN address when the host is asleep, and
   a transient `sindexer-gpu-keepawake.service` sleep inhibitor stays active
   for the indexing call (this matters: B550 can otherwise idle-sleep between
   embedding bursts),
6. the environment `EMBEDDING_URL` and `MILVUS_URL` point at the loopback
   services rather than a cloud or Mac-MLX endpoint.

It then runs `sindexer index` (full rebuild, the default) or, with `--update`,
the incremental path. Both paths read the same profile the `sindexer` wrapper
sources, so the two commands cannot silently diverge.

## Why a full rebuild is slower than an update

`sindexer index` rebuilds the whole codebase index from scratch; `sindexer
update` only re-embeds changed chunks and skips unchanged chunk IDs recorded in
the manifest. A manifest whose `inputs` no longer match the current embedding
identity forces exactly one full rebuild, so a later `update` returns to the
cheap path. Prefer `--update` unless a rebuild was explicitly requested.

## Proving both GPUs actually served the run

Port 1236 alone is not evidence of dual-GPU use; it is the balanced entry point.
During and after a rebuild, confirm both replicas received work:

```sh
ssh B550 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
ssh B550 'journalctl -u sindexer-embed-gpu@0.service -u sindexer-embed-gpu@1.service --since "5 minutes ago" | grep -c "/v1/embeddings"'
```

Both GPU utilization readings should rise, and both journals should show
`POST /v1/embeddings` lines. Per-replica throughput differs by design: GPU 0
was measured faster than GPU 1, and nginx balances by least connections.

## Installing or repairing the dual-GPU endpoint

```sh
repo=/Users/kearm/AlphaHENG/contrib/rust_sindexer/deploy/b550
scp "$repo/systemd/sindexer-embed-gpu@.service" B550:/tmp/
ssh B550 'sudo install -m 0644 /tmp/sindexer-embed-gpu@.service /etc/systemd/system/'
ssh B550 'sudo systemctl daemon-reload && sudo systemctl enable --now sindexer-embed-gpu@0 sindexer-embed-gpu@1'
scp "$repo/nginx/sindexer-jina-upstream.conf" B550:/tmp/
ssh B550 'sudo install -m 0644 /tmp/sindexer-jina-upstream.conf /etc/nginx/conf.d/ && sudo nginx -t && sudo systemctl reload nginx'
```

Mac-side tunnel (loopback 1236 to B550 1238), which is what the wrapper's
`EMBEDDING_URL` and `OPENAI_BASE_URL` both point at:

```sh
ln -sf "$repo/launchd/sindexer-b550-embed-tunnel" ~/.local/bin/sindexer-b550-embed-tunnel
cp "$repo/launchd/dev.resmp.sindexer-b550-embed-tunnel.plist" ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/dev.resmp.sindexer-b550-embed-tunnel 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.resmp.sindexer-b550-embed-tunnel.plist
curl -sS http://127.0.0.1:1236/health
```

## Health and freshness

`sindexer-doctor` (source: `deploy/local/sindexer-doctor`) reports the wrapper,
the tunnel agent and embedding endpoint, Milvus collections and container
health, per-repo manifest freshness, the LM Studio fallback, and B550
reachability. `sindexer-gpu-rebuild` runs the same embedding and Milvus probes
itself and fails closed when either is unhealthy, so a rebuild never silently
degrades to lexical-only or to a Mac model.

## Non-claims

This document does not claim a specific rebuild duration; measured time depends
on repository size, batch size, and GPU contention. It also does not claim that
`update` is free: unchanged chunks are skipped, but changed chunks are
re-embedded.
