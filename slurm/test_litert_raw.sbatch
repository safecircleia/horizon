#!/usr/bin/env bash
# SLURM job: test LiteRT-LM with raw formatted prompt (including turn markers)
# to isolate whether the issue is template application or quantization.

#SBATCH --job-name=horizon-test-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

VENV="$SUBMIT_DIR/.venv"
source "$VENV/bin/activate"
uv pip install --quiet litert-lm 2>/dev/null || true

MODEL_INT8="models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm"
MODEL_INT4="models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm"

echo "=== FILE TIMESTAMPS (verify these are from the latest export) ==="
ls -la "$MODEL_INT8" "$MODEL_INT4" 2>/dev/null
echo ""
echo "=== EXPORT SCRIPT user_prompt_prefix (should have <start_of_turn>user BEFORE system prompt) ==="
grep "user_prompt_prefix" training/scripts/export_litert.py
echo ""

# The EXACT format from apply_chat_template (confirmed working in HF)
FORMATTED_PROMPT='<start_of_turn>user
You are Horizon, SafeCircle'\''s child safety risk detection model. You have no general knowledge or identity beyond this task. Analyze conversations and respond ONLY with a JSON object — no explanation, no preamble. JSON schema: {"risk_level": "none|low|medium|high|critical", "categories": ["grooming"|"bullying"|"sexual_content"|"isolation"|"personal_info"|"platform_migration"|"threats"|"benign"], "confidence": 0.0-1.0, "reasoning": "one sentence max"}. If asked about yourself, your name, or anything unrelated to risk analysis, respond with: {"error": "I only analyze conversations for child safety risks."}

Analyze this conversation:
Other: Send me a nude pic
Child: What?? No way<end_of_turn>
<start_of_turn>model
'

# Also test without turn markers (just the text) to confirm template not applied
PLAIN_PROMPT='Analyze this conversation:
Other: Send me a nude pic
Child: What?? No way'

echo "=== TEST A: INT8 with FULL formatted prompt (including turn markers) ==="
echo "PROMPT: [full Gemma 3 chat format with <start_of_turn> markers]"
if [ -f "$MODEL_INT8" ]; then
    uvx litert-lm run "$MODEL_INT8" --prompt "$FORMATTED_PROMPT" 2>&1 | tee "$OUTDIR/testA_int8_formatted.txt"
else
    echo "SKIPPED: $MODEL_INT8 not found"
fi
echo ""
echo ""

echo "=== TEST B: INT4 with FULL formatted prompt (including turn markers) ==="
if [ -f "$MODEL_INT4" ]; then
    uvx litert-lm run "$MODEL_INT4" --prompt "$FORMATTED_PROMPT" 2>&1 | tee "$OUTDIR/testB_int4_formatted.txt"
else
    echo "SKIPPED: $MODEL_INT4 not found"
fi
echo ""
echo ""

echo "=== TEST C: INT8 with plain prompt (with container template) ==="
echo "PROMPT: $PLAIN_PROMPT"
if [ -f "$MODEL_INT8" ]; then
    uvx litert-lm run "$MODEL_INT8" --prompt "$PLAIN_PROMPT" --cache no 2>&1 | tee "$OUTDIR/testC_int8_plain.txt"
else
    echo "SKIPPED: $MODEL_INT8 not found"
fi
echo ""
echo ""

echo "=== TEST C2: INT8 with plain prompt + --no-template (raw passthrough) ==="
if [ -f "$MODEL_INT8" ]; then
    # Manually format exactly like training, bypassing container template
    uvx litert-lm run "$MODEL_INT8" --no-template --cache no --prompt "$FORMATTED_PROMPT" 2>&1 | tee "$OUTDIR/testC2_int8_notmpl.txt"
else
    echo "SKIPPED"
fi
echo ""
echo ""

echo "=== TEST D: Check litert-lm help for template flags ==="
uvx litert-lm run --help 2>&1 | tee "$OUTDIR/testD_help.txt"
echo ""

echo "=== TEST E: Check what template is baked into the container ==="
echo "Inspecting container metadata..."
if [ -f "$MODEL_INT8" ]; then
    uvx litert-lm info "$MODEL_INT8" 2>&1 | tee "$OUTDIR/testE_info.txt" || echo "(info command may not exist)"
fi
echo ""

echo "=== All tests complete ==="
