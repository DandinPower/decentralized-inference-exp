MODEL_NAME="Qwen/Qwen3-32B"
MODEL_DIR="Qwen/Qwen3-32B"
RESULT_NAME="Qwen/Qwen3-32B"

source .venv/bin/activate && python -u lctx_robustness_benchmark.py \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME"

MODEL_NAME="Qwen/Qwen3.5-27B"
MODEL_DIR="Qwen/Qwen3.5-27B"
RESULT_NAME="Qwen/Qwen3.5-27B"

source .venv/bin/activate && python -u lctx_robustness_benchmark.py \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME"

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_DIR="Qwen/Qwen3.5-9B"
RESULT_NAME="Qwen/Qwen3.5-9B"

source .venv/bin/activate && python -u lctx_robustness_benchmark.py \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME"

MODEL_NAME="Qwen/Qwen3-8B"
MODEL_DIR="/mnt/ssd/liaw/Qwen/Qwen3-8B"
RESULT_NAME="Qwen/Qwen3-8B"

source .venv/bin/activate && python -u lctx_robustness_benchmark.py \
    --model-name "$MODEL_NAME" \
    --model-dir "$MODEL_DIR" \
    --result-name "$RESULT_NAME"
    