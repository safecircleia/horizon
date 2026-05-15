#!/usr/bin/env bash
# Export merged model to GGUF variants using llama.cpp
#
# Usage:
#   bash scripts/export_gguf.sh models/horizon-full-merged models/horizon-full-gguf
#
# Requires llama.cpp cloned and built at $LLAMA_CPP_DIR (default: ~/llama.cpp)

set -euo pipefail

MERGED_MODEL="${1:-models/horizon-full-merged}"
GGUF_DIR="${2:-models/horizon-full-gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Install with:"
    echo "  git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR"
    echo "  cd $LLAMA_CPP_DIR && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"
    exit 1
fi

mkdir -p "$GGUF_DIR"

echo "==> Converting to F16 GGUF..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_MODEL" \
    --outfile "$GGUF_DIR/horizon-full-f16.gguf" \
    --outtype f16

echo "==> Quantizing to Q4_K_M..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$GGUF_DIR/horizon-full-f16.gguf" \
    "$GGUF_DIR/horizon-full-Q4_K_M.gguf" \
    Q4_K_M

echo "==> Quantizing to Q5_K_M..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$GGUF_DIR/horizon-full-f16.gguf" \
    "$GGUF_DIR/horizon-full-Q5_K_M.gguf" \
    Q5_K_M

echo "==> Quantizing to Q8_0..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$GGUF_DIR/horizon-full-f16.gguf" \
    "$GGUF_DIR/horizon-full-Q8_0.gguf" \
    Q8_0

echo ""
echo "GGUF variants saved to $GGUF_DIR:"
ls -lh "$GGUF_DIR"/*.gguf
