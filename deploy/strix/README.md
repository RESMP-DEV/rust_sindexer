# Strix embedding and vector-store services

This deployment keeps repository walking, lexical indexes, Git authority,
and task state on the main machine. Strix hosts one pinned Jina Code MXFP4
embedding replica on its RTX 5070 Laptop GPU and, since 2026-10-24, the
Milvus vector store for the whole fleet. The services are compute
infrastructure, not an authority: source repositories and index manifests
remain on the Mac, and the vector collection can be rebuilt or restored
from the Mac cold backup.

The embedding runtime binds only Strix loopback port 1236; Milvus binds
loopback ports 19530 (gRPC/REST), 9091 (management), and 9000/9001 (object
store). Mac LaunchAgents forward `127.0.0.1:1236` and `127.0.0.1:19530` to
those ports over the `strix-zt` SSH identity. There is no nginx balancer,
no second GPU replica, and no public exposure in this topology.

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

## Vector store (Milvus standalone)

- Runtime root: `/home/kearm/sindexer-stack` (compose file copied from
  `milvus/docker-compose.yml` in this directory)
- Images: `milvusdb/milvus:v2.5.3`, `quay.io/coreos/etcd:v3.5.16`, object
  store `bitnamilegacy/minio:2024.12.18`
- Credentials: `minioadmin`/`minioadmin`, bucket `a-bucket`, root path
  `files` (same layout as the retired Mac OrbStack stack, so backups move
  between them unchanged)
- Mac tunnel: `launchd/dev.resmp.sindexer-strix-milvus-tunnel.plist` plus
  `launchd/sindexer-strix-milvus-tunnel`, forwarding Mac `127.0.0.1:19530`

Object-store substitution note: `docker.io/minio/minio` is pull-blocked for
this household IP (every tag denied, both the docker daemon and skopeo), and
`dl.min.io` serves 410 for all archives from the same network. The compose
file therefore pins `bitnamilegacy/minio:2024.12.18` with its own paths and
`MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`; the S3 API, credentials, and bucket
layout are identical to the official image the Mac and B550 stacks ran.
amd64 `milvusdb/milvus` and etcd were pulled directly on Strix (quay and a
retrying Hub lane both work for those repositories); on a fresh host, pull
with `docker pull --platform linux/amd64` and retry transient CDN resets.

## Install or update (Milvus)

```bash
repo=$PWD
cat "$repo/milvus/docker-compose.yml" | ssh strix-zt 'mkdir -p /home/kearm/sindexer-stack && cat > /home/kearm/sindexer-stack/docker-compose.yml && mkdir -p /home/kearm/sindexer-stack/volumes/minio && sudo chown 1001:1001 /home/kearm/sindexer-stack/volumes/minio'
ssh strix-zt 'cd /home/kearm/sindexer-stack && docker compose up -d'
ln -sf "$repo/launchd/sindexer-strix-milvus-tunnel" ~/.local/bin/sindexer-strix-milvus-tunnel
cp "$repo/launchd/dev.resmp.sindexer-strix-milvus-tunnel.plist" ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/dev.resmp.sindexer-strix-milvus-tunnel 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.resmp.sindexer-strix-milvus-tunnel.plist
```

The Mac port 19530 must be free before the milvus tunnel agent loads: stop
the retired local OrbStack stack first (`docker stop milvus-standalone
milvus-minio milvus-etcd` and set their restart policies to `no`).

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
7. Strix `curl http://127.0.0.1:19530/v2/vectordb/collections/list` returns
   code 0 and the Mac `sindexer collections` sees the same collections
   through the milvus tunnel.
8. `sindexer-doctor --json` reports `milvus.ok: true` and
   `milvus_container.state: tunnel-up`.
