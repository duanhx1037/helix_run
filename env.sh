uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install vllm --torch-backend=auto