# B550 embedding and vector services

This deployment keeps repository walking, Tantivy, Git authority, and task
state on the Mac. B550 hosts the exact pinned Jina Code MXFP4 embedding model
and the Milvus/MinIO/etcd vector stack. All B550 ports bind to loopback and the
Mac reaches them through the persistent SSH tunnel.

The embedding launcher probes every working RTX 3090 Ti and selects the card
with the most free VRAM. The model revision is
`a328a7f269e0cc7eb7eaf977225f698940aa7d25`; its `model.safetensors` SHA-256 is
`9cadd568e6757e15358d960e4688bb0792a7edcd9889ef3d37f2361f4db6b756`.

The CUDA loader consumes the published U32 E2M1 packing and U8 E8M0 scales
directly. It does not dequantize or repack the checkpoint at load time.
PyTorch SDPA supplies attention; FlashAttention is neither installed nor built.
Native-build concurrency is capped at one worker, and systemd limits the
embedding service to 48 GiB so a failed warm-up cannot exhaust the host.

## Runtime paths

- Service root: `/home/kearm/services/rust-indexer`
- Model: `models/jina-code-mxfp4-a328a7f`
- Server sources: `runtime/`
- Vector stack and data: `milvus/`
- Mac tunnel: `~/Library/LaunchAgents/dev.resmp.rust-indexer-b550-tunnel.plist`

## Credentials

MinIO is reachable only over the compose network; no host ports are
published for it. Its credentials live in an untracked env file, not in the
compose file: copy `deploy/b550/.env.example` to `deploy/b550/.env`, set
real values for `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`, and keep the
file at mode `0600`. `docker compose` loads `.env` from the directory
holding the compose file, so the deployed `milvus/` copy on B550 needs the
same file.

## Acceptance

The move is complete only when:

1. `/health` reports `runtime=cuda-mxfp4` and the selected GPU UUID.
2. Matched MLX and CUDA embeddings pass the cosine-parity gate.
3. All source and destination Milvus collections have matching row counts.
4. A real `rust-indexer search` returns semantic and lexical hits through the
   loopback tunnel.
5. Both B550 services and the Mac tunnel recover after controlled restarts.
