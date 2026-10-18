# B550 embedding and vector services

This deployment keeps repository walking, Tantivy, Git authority, Milvus, and
task state on the Mac. B550 hosts the exact pinned Jina Code MXFP4 embedding
model on both RTX 3090 Ti GPUs. All B550 ports bind to loopback and the Mac
reaches the balanced embedding endpoint through the persistent SSH tunnel.

Two complete embedding replicas run on the two RTX 3090 Ti GPUs. nginx uses
least-connections balancing over loopback ports 1236 and 1237, and the Mac
tunnel exposes that balanced endpoint locally as port 1236. The model
revision is
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
- GPU systemd template: `systemd/sindexer-embed-gpu@.service`
- nginx upstream: `nginx/sindexer-jina-upstream.conf`
- Mac tunnel: `launchd/dev.resmp.sindexer-b550-embed-tunnel.plist` plus
  `launchd/sindexer-b550-embed-tunnel`

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
6. During a full rebuild, both GPU utilization readings rise and both systemd
   instance journals receive `/v1/embeddings` requests; port 1236 alone is
   not proof of dual-GPU operation.

The older `rust-indexer-embedding.service`, `rust-indexer-milvus.service`,
and `dev.resmp.rust-indexer-b550-tunnel.plist` files remain historical
recovery references; the active endpoint is the dual-GPU template, nginx
upstream, and `dev.resmp.sindexer-b550-embed-tunnel` described above.
