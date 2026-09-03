#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-./checkpoints/Breeze-TTS-2}"

python -m pip install -q "huggingface_hub>=0.26.0"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BreezeBlue/Breeze-TTS-2",
    local_dir="${DEST}",
    local_dir_use_symlinks=False,
)
print("downloaded ${DEST}")
PY

if [[ ! -d "${DEST}/audio_tokenizer" ]]; then
  echo "error: ${DEST}/audio_tokenizer missing — the HF snapshot is incomplete" >&2
  exit 1
fi
