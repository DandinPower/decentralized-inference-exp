uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install huggingface-hub numpy packaging psutil pyyaml safetensors transformers datasets accelerate bitsandbytes

git clone https://github.com/DandinPower/PyBitSqueeze.git
cd PyBitSqueeze
uv pip install -r requirements.txt
uv pip install . --no-build-isolation
cd ..