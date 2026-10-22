# Strix embedding service

This deployment keeps repository walking, lexical indexes, Git authority,
and task state on the main machine. Strix hosts one pinned Jina Code MXFP4
embedding replica on its RTX 5070 Laptop GPU. The service is compute
infrastructure, not an authority: source repositories and index manifests
remain on the Mac, and the vector collection can be rebuilt.

The runtime binds only Strix loopback port 1236. The Mac LaunchAgent forwards
`127.0.0.1:1236` to that port over the `strix-zt` SSH identity. There is no
nginx balancer and no second GPU replica in this topology.

## Runtime contract

- Service: `systemd/sindexer-embed-gpu.service`
- Runtime root: `/home/kearm/services/rust-indexer/runtime`
- Model path: `/home/kearm/services/rust-indexer/models/jina-code-mxfp4-a328a7f`
- Model: `jina-code-embeddings-1.5b-block-gptq-mxfp4-32k`
- Dimensions: 1536
- Runtime: CUDA MXFP4 on GPU UUID 0
- Memory limits: 12 GiB high, 16 GiB maximum
- Mac tunnel: `launchd/dev.resmp.sindexer-strix-embed-tunnel.plist` plus
  `launchd/sindexer-strix-embed-tunnel`

The service contains no provider credentials. It accepts OpenAI-compatible
embedding requests on its private loopback interface and exposes `/health`.

## Install or update

```bash
repo=$PWD
scp "$repo/systemd/sindexer-embed-gpu.service" strix-zt:/tmp/
ssh strix-zt 'sudo install -m 0644 /tmp/sindexer-embed-gpu.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now sindexer-embed-gpu.service'
ln -sf "$repo/launchd/sindexer-strix-embed-tunnel" ~/.local/bin/sindexer-strix-embed-tunnel
cp "$repo/launchd/dev.resmp.sindexer-strix-embed-tunnel.plist" ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/dev.resmp.sindexer-strix-embed-tunnel 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.resmp.sindexer-strix-embed-tunnel.plist
```

Remove the retired B550-named agent when migrating an existing Mac:

```bash
launchctl bootout gui/$(id -u)/dev.resmp.sindexer-b550-embed-tunnel 2>/dev/null || true
launchctl disable gui/$(id -u)/dev.resmp.sindexer-b550-embed-tunnel 2>/dev/null || true
```

## Acceptance

1. `ssh strix-zt 'hostname && systemctl is-active sindexer-embed-gpu.service'`
   reports `strix` and `active`.
2. Strix `/health` reports the pinned model, 1536 dimensions,
   `cuda-mxfp4`, GPU UUID 0, and the RTX 5070 device.
3. Mac `/health` through port 1236 reports the same identity.
4. `lsof` shows the Mac port 1236 listener owned by the Strix tunnel SSH
   process, not by a local model server.
5. A real `sindexer search` completes through `~/.local/bin/sindexer`.
6. The old B550 embedding replicas remain inactive and disabled on B550.
