MODEL_NAME="Qwen/Qwen3-8B"
MODEL_DIR="/mnt/ssd/liaw/hf_cache/Qwen/Qwen3-8B"
RESULT_NAME="Qwen/Qwen3-8B"

source .venv/bin/activate && python -u heuristic_benchmark.py \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME"
