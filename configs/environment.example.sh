# Copy to a private environment file; do not commit credentials.
export SAM3_CHECKPOINT_PATH="$PWD/assets/sam3.pt"
export PYRUALEAN_SAM3_URL=http://127.0.0.1:8765
# Set UUIDs from nvidia-smi -L on YOUR machine.
export SAM3_GPU_UUID=GPU-REPLACE-SAM3
export POLICY_GPU_UUID=GPU-REPLACE-POLICY
export OPENAI_BASE_URL=https://YOUR-ENDPOINT/v1
export PYRUALEAN_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY=REPLACE-ME
export PYRUALEAN_API_KEY_ENV=OPENAI_API_KEY
export PYRUALEAN_MODEL=qwen3.8-flash
