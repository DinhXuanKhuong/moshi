#!/usr/bin/env bash
# Download the original checkpoint and trace it on the active Python environment's GPU.
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: bash scripts/run_moshiko_gpu_trace.sh CHECKPOINT_DIR OUTPUT_DIR [tracer options]" >&2
    exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoints="$1"
output="$2"
shift 2

python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see CUDA"; assert torch.cuda.is_bf16_supported(including_emulation=False), "Native CUDA BF16 is required"; print(torch.cuda.get_device_name(0), torch.__version__)'
if [[ -e "$output/trace.json" || -e "$output/explorer.html" ]]; then
    echo "Output already contains a trace or explorer. Choose a new output directory." >&2
    exit 2
fi
mkdir -p -- "$checkpoints" "$output"

python - "$checkpoints" <<'PY'
import sys
from huggingface_hub import hf_hub_download

for filename in (
    "tokenizer_spm_32k_3.model",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "model.safetensors",
):
    path = hf_hub_download(
        repo_id="kyutai/moshiko-pytorch-bf16",
        revision="2bfc9ae",
        filename=filename,
        local_dir=sys.argv[1],
    )
    print(f"Checkpoint asset: {path}", flush=True)
PY

python -u "$repo/scripts/trace_moshi_audio.py" \
    --sample --tail-frames 100 "$@" \
    --device cuda \
    --mimi "$checkpoints/tokenizer-e351c8d8-checkpoint125.safetensors" \
    --moshi "$checkpoints/model.safetensors" \
    --tokenizer "$checkpoints/tokenizer_spm_32k_3.model" \
    --output "$output" 2>&1 | tee "$output/terminal.txt"

python "$repo/scripts/build_moshi_explorer.py" "$output"
