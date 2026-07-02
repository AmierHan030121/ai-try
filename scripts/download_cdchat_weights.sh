#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${CDCHAT_REVISION:-bf08270f943114eee92c5fcd93daf5009d460af4}"
BASE_URL="${CDCHAT_BASE_URL:-https://huggingface.co/mubashir04/cdchat/resolve/${REVISION}}"

export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export all_proxy="${all_proxy:-socks5://127.0.0.1:7890}"

FILES=(
  "pretrain_mm_projector/mm_projector.bin"
  "model_weights_cdchat/pytorch_model-00001-of-00002.bin"
  "model_weights_cdchat/pytorch_model-00002-of-00002.bin"
)

declare -A EXPECTED_SIZE=(
  ["pretrain_mm_projector/mm_projector.bin"]="50349693"
  ["model_weights_cdchat/pytorch_model-00001-of-00002.bin"]="9976634558"
  ["model_weights_cdchat/pytorch_model-00002-of-00002.bin"]="4158750994"
)

declare -A EXPECTED_SHA256=(
  ["pretrain_mm_projector/mm_projector.bin"]="649112857bc8683b508749e53bf09843afc2725487dd397b13799e260822b85e"
  ["model_weights_cdchat/pytorch_model-00001-of-00002.bin"]="6f6e941c126d913a889a7a6ed255ed130b27444c2268e35fb907a5a1a67e882d"
  ["model_weights_cdchat/pytorch_model-00002-of-00002.bin"]="42c4bf4528254a0544894673eecd2fbed6a26d6705740a8483c2aabaca1e3e18"
)

download_file() {
  local rel_path="$1"
  local out_path="${ROOT_DIR}/checkpoints/cdchat/${rel_path}"
  local expected_size="${EXPECTED_SIZE[${rel_path}]:-}"
  local expected_sha="${EXPECTED_SHA256[${rel_path}]:-}"
  mkdir -p "$(dirname "${out_path}")"

  if [[ -f "${out_path}" && -n "${expected_size}" ]]; then
    local current_size
    current_size="$(stat -c "%s" "${out_path}")"
    if [[ "${current_size}" -eq "${expected_size}" ]]; then
      local current_sha
      current_sha="$(sha256sum "${out_path}" | awk '{print $1}')"
      if [[ "${current_sha}" == "${expected_sha}" ]]; then
        echo "Already verified: ${rel_path}"
        return 0
      fi
      local bad_path="${out_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
      echo "Quarantining SHA-mismatched file: ${out_path} -> ${bad_path}" >&2
      mv "${out_path}" "${bad_path}"
    elif [[ "${current_size}" -gt "${expected_size}" ]]; then
      local bad_path="${out_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
      echo "Quarantining oversized file: ${out_path} -> ${bad_path}" >&2
      mv "${out_path}" "${bad_path}"
    fi
  fi

  wget -c --timeout=60 --tries=20 --waitretry=10 --progress=dot:giga \
    -O "${out_path}" \
    "${BASE_URL}/${rel_path}"
}

echo "CDChat revision: ${REVISION}"
echo "Base URL: ${BASE_URL}"
echo "http_proxy=${http_proxy}"
echo "https_proxy=${https_proxy}"

if [[ "${1:-}" == "--verify-only" ]]; then
  python "${ROOT_DIR}/scripts/verify_cdchat_weights.py"
  exit $?
fi

if [[ "$#" -gt 0 ]]; then
  FILES=("$@")
fi

for rel_path in "${FILES[@]}"; do
  download_file "${rel_path}"
done

python "${ROOT_DIR}/scripts/verify_cdchat_weights.py"
