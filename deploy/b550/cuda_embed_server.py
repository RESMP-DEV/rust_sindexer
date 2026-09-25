"""OpenAI-compatible CUDA MXFP4 server for Jina Code Embeddings 1.5B."""

from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.integrations.hub_kernels import get_kernel

MODEL_PATH = Path(
    os.environ.get(
        "JINA_CODE_MODEL_PATH",
        "/home/kearm/services/rust-indexer/models/jina-code-mxfp4-a328a7f",
    )
)
MODEL_NAME = os.environ.get(
    "JINA_CODE_MODEL_NAME", "jina-code-embeddings-1.5b-block-gptq-mxfp4-32k"
)
MAX_TOKENS = int(os.environ.get("JINA_CODE_MAX_TOKENS", "8192"))
MAX_BATCH_SIZE = int(os.environ.get("JINA_CODE_MAX_BATCH_SIZE", "32"))
MAX_REQUEST_BYTES = int(
    os.environ.get("JINA_CODE_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))
)
MAX_INPUT_BYTES = int(
    os.environ.get("JINA_CODE_MAX_INPUT_BYTES", str(2 * 1024 * 1024))
)
QUERY_PREFIX = "Find the most relevant code snippet given the following query:\n"
PASSAGE_PREFIX = "Candidate code snippet:\n"
DIMENSIONS = 1536

_KERNEL = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
_MODEL: CudaMxfp4Encoder | None = None
_MODEL_LOCK = threading.Lock()


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embeddings request payload."""

    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: Literal["float"] = "float"
    input_type: Literal["query", "passage", "raw"] = Field(default="raw")


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    """Swap the module at the dotted attribute path for a replacement."""
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _replace_parameter(root: nn.Module, name: str, value: torch.Tensor) -> None:
    """Rebind the parameter tensor at the dotted path as a frozen parameter."""
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    parent._parameters[parts[-1]] = nn.Parameter(value, requires_grad=False)


class Mxfp4Linear(nn.Module):
    """Qwen2 linear layer backed by packed E2M1 values and E8M0 scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_u32: torch.Tensor,
        scales_u8: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        """Validate the packed shapes and register weights, scales, and bias."""
        super().__init__()
        if packed_u32.dtype != torch.uint32:
            raise TypeError(f"expected uint32 packed weights, got {packed_u32.dtype}")
        if scales_u8.dtype != torch.uint8:
            raise TypeError(f"expected uint8 scales, got {scales_u8.dtype}")
        if tuple(packed_u32.shape) != (out_features, in_features // 8):
            raise ValueError(f"invalid packed shape for {in_features}x{out_features}")
        if tuple(scales_u8.shape) != (out_features, in_features // 32):
            raise ValueError(f"invalid scale shape for {in_features}x{out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("packed_u32", packed_u32.contiguous())
        self.register_buffer("scales_u8", scales_u8.contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.float().contiguous()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the CUDA MXFP4 matmul over the de-packed E2M1 weights."""
        output_shape = (*inputs.shape[:-1], self.out_features)
        flat_inputs = inputs.reshape(-1, self.in_features)

        # Eight E2M1 nibbles are packed in each U32. Viewing as bytes and
        # transposing produces the column-major [K/2, N] CUDA operand without
        # changing a packed value.
        packed_bytes = self.packed_u32.view(torch.uint8).reshape(self.out_features, -1)
        cuda_weight = packed_bytes.t()
        cuda_scales = self.scales_u8.t()
        precision = _KERNEL.matmul_ogs.PrecisionConfig(
            weight_scale=cuda_scales,
            out_dtype=inputs.dtype,
        )
        output = _KERNEL.matmul_ogs.matmul_ogs(
            flat_inputs,
            cuda_weight,
            self.bias,
            precision_config=precision,
        )
        return output.reshape(output_shape)


class CudaMxfp4Encoder:
    """Canonical Qwen2 body with CUDA MXFP4 linears and last-token pooling."""

    def __init__(self, model_path: Path) -> None:
        """Build the Qwen2 body, load MXFP4 linears from safetensors, move to CUDA."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.hidden_size != DIMENSIONS:
            raise ValueError(
                f"expected {DIMENSIONS} dimensions, got {config.hidden_size}"
            )
        config.use_cache = False
        config._attn_implementation = "sdpa"

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = AutoModel.from_config(config)
        finally:
            torch.set_default_dtype(previous_dtype)

        weights_path = model_path / "model.safetensors"
        used_keys: set[str] = set()
        with safe_open(weights_path, framework="pt", device="cpu") as state:
            artifact_keys = set(state.keys())
            for name, module in list(model.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                weight_key = f"{name}.weight"
                scale_key = f"{name}.scales"
                bias_key = f"{name}.bias"
                bias = state.get_tensor(bias_key) if bias_key in artifact_keys else None
                replacement = Mxfp4Linear(
                    module.in_features,
                    module.out_features,
                    state.get_tensor(weight_key),
                    state.get_tensor(scale_key),
                    bias,
                )
                _replace_module(model, name, replacement)
                used_keys.update((weight_key, scale_key))
                if bias is not None:
                    used_keys.add(bias_key)

            for name, _parameter in list(model.named_parameters()):
                if name not in artifact_keys:
                    raise KeyError(f"checkpoint is missing parameter {name}")
                _replace_parameter(model, name, state.get_tensor(name))
                used_keys.add(name)

            unused = artifact_keys - used_keys
            if unused:
                raise ValueError(
                    f"checkpoint contains unused tensors: {sorted(unused)[:8]}"
                )

        self.model = model.eval().requires_grad_(False).to(self.device)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _prepare_input(text: str, input_type: str) -> str:
        """Prepend the type-specific instruction prefix for the text."""
        if input_type == "query":
            return QUERY_PREFIX + text
        if input_type == "passage":
            return PASSAGE_PREFIX + text
        return text

    def encode(
        self, texts: list[str], input_type: str
    ) -> tuple[list[list[float]], int]:
        """Tokenize, run last-token pooling, and return L2-normalized vectors."""
        prepared = [self._prepare_input(text, input_type) for text in texts]
        encoded = self.tokenizer(
            prepared,
            add_special_tokens=True,
            max_length=MAX_TOKENS,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        token_count = int(encoded["attention_mask"].sum().item())
        inputs = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            hidden = self.model(
                **inputs, use_cache=False, return_dict=True
            ).last_hidden_state
            positions = torch.arange(hidden.shape[1], device=self.device)
            last_indices = (inputs["attention_mask"] * positions).argmax(dim=1)
            vectors = hidden[
                torch.arange(hidden.shape[0], device=self.device), last_indices
            ]
            vectors = F.normalize(vectors.float(), p=2, dim=-1)
            torch.cuda.synchronize()
        return vectors.cpu().tolist(), token_count


def get_model() -> CudaMxfp4Encoder:
    """Return the process-wide encoder, constructing it on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = CudaMxfp4Encoder(MODEL_PATH)
    return _MODEL


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model at startup so requests never pay initialization cost."""
    get_model()
    yield


app = FastAPI(title="Jina Code CUDA MXFP4", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject requests whose declared body exceeds MAX_REQUEST_BYTES with 413."""
    raw_length = request.headers.get("content-length")
    if raw_length is not None and raw_length.isdigit():
        content_length = int(raw_length)
        if content_length > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"request body {content_length} bytes exceeds "
                        f"maximum {MAX_REQUEST_BYTES}"
                    )
                },
            )
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, object]:
    """Report runtime, model, and selected GPU identity."""
    model = get_model()
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "runtime": "cuda-mxfp4",
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", "cuda:0"),
        "device": torch.cuda.get_device_name(model.device),
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    """List the served model in OpenAI format."""
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> dict[str, object]:
    """Validate, encode, and return OpenAI-shaped embeddings with usage."""
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}",
        )
    for index, text in enumerate(texts):
        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"input {index} exceeds maximum {MAX_INPUT_BYTES} UTF-8 bytes",
            )

    started = time.perf_counter()
    with _MODEL_LOCK:
        vectors, token_count = get_model().encode(texts, request.input_type)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": index}
            for index, vector in enumerate(vectors)
        ],
        "model": MODEL_NAME,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
