#!/usr/bin/env bash
set -euo pipefail

service_root="${RUST_INDEXER_SERVICE_ROOT:-/home/kearm/services/rust-indexer}"
best_uuid=""
best_free=-1

for gpu_index in $(seq 0 7); do
	if ! gpu_line=$(nvidia-smi -i "$gpu_index" \
		--query-gpu=uuid,name,memory.free \
		--format=csv,noheader,nounits 2>/dev/null); then
		continue
	fi
	IFS=, read -r gpu_uuid gpu_name gpu_free <<<"$gpu_line"
	gpu_uuid="${gpu_uuid//[[:space:]]/}"
	gpu_name="${gpu_name# }"
	gpu_free="${gpu_free//[[:space:]]/}"
	if [[ "$gpu_name" != *"RTX 3090 Ti"* ]]; then
		continue
	fi
	if ((gpu_free > best_free)); then
		best_uuid="$gpu_uuid"
		best_free="$gpu_free"
	fi
done

if [[ -z "$best_uuid" ]]; then
	echo "no usable RTX 3090 Ti found" >&2
	exit 1
fi

export CUDA_VISIBLE_DEVICES="$best_uuid"
export PYTHONPATH="$service_root/.venv/lib/python3.12/site-packages"
export JINA_CODE_MODEL_PATH="${JINA_CODE_MODEL_PATH:-$service_root/models/jina-code-mxfp4-a328a7f}"
export JINA_CODE_MODEL_NAME="${JINA_CODE_MODEL_NAME:-jina-code-embeddings-1.5b-block-gptq-mxfp4-32k}"
# Keep any first-run native compilation bounded. A separate FlashAttention
# source build previously launched one multi-gigabyte cicc process per core.
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

echo "selected_gpu_uuid=$best_uuid free_mib=$best_free"
exec /home/kearm/AlphaHENG-cuda-bringup/.venv/bin/python -m uvicorn \
	cuda_embed_server:app \
	--app-dir "$service_root/runtime" \
	--host 127.0.0.1 \
	--port 1235
