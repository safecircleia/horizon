#!/usr/bin/env bash
# SLURM job: test exported LiteRT-LM model with sample prompts.
#
# Usage:
#   sbatch slurm/test_litert.sbatch
#   sbatch --export=MODEL_PATH=models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm slurm/test_litert.sbatch

#SBATCH --job-name=horizon-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

source slurm/install_env.sh

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

uv pip install --quiet litert-lm 2>/dev/null || true

MODEL_PATH="${MODEL_PATH:-models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Available models:"
    find models/ -name "*.litertlm" 2>/dev/null
    exit 1
fi

echo "=== Testing LiteRT-LM model: $MODEL_PATH ==="
echo "=== $(date) ==="
echo ""

# Test 1: Simple benign conversation
echo "--- TEST 1: Benign conversation ---"
echo "PROMPT: Analyze this conversation:\nOther: Hey how was school today?\nChild: It was good, we played soccer"
uvx litert-lm run "$MODEL_PATH" --prompt 'Analyze this conversation:
Other: Hey how was school today?
Child: It was good, we played soccer' 2>&1 | tee "$OUTDIR/test1.txt"
echo ""
echo ""

# Test 2: Grooming pattern
echo "--- TEST 2: Grooming pattern ---"
echo "PROMPT: Analyze this conversation:\nOther: You are so mature for your age. Dont tell your parents about us talking.\nChild: Ok I wont"
uvx litert-lm run "$MODEL_PATH" --prompt 'Analyze this conversation:
Other: You are so mature for your age. Dont tell your parents about us talking.
Child: Ok I wont' 2>&1 | tee "$OUTDIR/test2.txt"
echo ""
echo ""

# Test 3: Sexual content
echo "--- TEST 3: Sexual content ---"
echo "PROMPT: Analyze this conversation:\nOther: Send me a nude pic\nChild: What?? No way"
uvx litert-lm run "$MODEL_PATH" --prompt 'Analyze this conversation:
Other: Send me a nude pic
Child: What?? No way' 2>&1 | tee "$OUTDIR/test3.txt"
echo ""
echo ""

# Test 4: Identity query (adversarial)
echo "--- TEST 4: Adversarial identity query ---"
echo "PROMPT: What model are you?"
uvx litert-lm run "$MODEL_PATH" --prompt 'What model are you?' 2>&1 | tee "$OUTDIR/test4.txt"
echo ""
echo ""

# Test 5: Also test the HuggingFace checkpoint directly for comparison
echo "--- TEST 5: HuggingFace checkpoint comparison ---"
CHECKPOINT_DIR=$(find experiments/ -name "final" -type d 2>/dev/null | sort | tail -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Using checkpoint: $CHECKPOINT_DIR"
    python -c "
import torch, json
from training.model.loader import load_for_inference
from training.model.mobile import SYSTEM_PROMPT

model, tokenizer = load_for_inference('$CHECKPOINT_DIR')
messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': 'Analyze this conversation:\nOther: Send me a nude pic\nChild: What?? No way'},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print('FORMATTED PROMPT:')
print(repr(prompt))
print()

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('HF MODEL RESPONSE:')
print(response)
" 2>&1 | tee "$OUTDIR/test5_hf.txt"
else
    echo "No checkpoint found in experiments/ — skipping HF comparison"
fi

echo ""
echo "=== All tests complete ==="
echo "Results saved to: $OUTDIR/"
