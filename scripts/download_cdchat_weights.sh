#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${CDCHAT_REVISION:-bf08270f943114eee92c5fcd93daf5009d460af4}"
BASE_URL="${CDCHAT_BASE_URL:-https://huggingface.co/mubashir04/cdchat/resolve/${REVISION}}"
PREFLIGHT_URL="${CDCHAT_PREFLIGHT_URL:-${BASE_URL}/model_weights_cdchat/config.json}"

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
  local part_path="${out_path}.part"
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
    elif [[ "${current_size}" -lt "${expected_size}" ]]; then
      echo "Moving incomplete file to resumable partial: ${out_path} -> ${part_path}" >&2
      if [[ -f "${part_path}" ]]; then
        local bad_path="${out_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
        mv "${out_path}" "${bad_path}"
      else
        mv "${out_path}" "${part_path}"
      fi
    fi
  fi

  if [[ -f "${part_path}" && -n "${expected_size}" ]]; then
    local part_size
    part_size="$(stat -c "%s" "${part_path}")"
    if [[ "${part_size}" -eq "${expected_size}" ]]; then
      local part_sha
      part_sha="$(sha256sum "${part_path}" | awk '{print $1}')"
      if [[ "${part_sha}" == "${expected_sha}" ]]; then
        mv "${part_path}" "${out_path}"
        echo "Verified partial promoted: ${rel_path}"
        return 0
      fi
      local bad_path="${part_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
      echo "Quarantining SHA-mismatched partial: ${part_path} -> ${bad_path}" >&2
      mv "${part_path}" "${bad_path}"
    elif [[ "${part_size}" -gt "${expected_size}" ]]; then
      local bad_path="${part_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
      echo "Quarantining oversized partial: ${part_path} -> ${bad_path}" >&2
      mv "${part_path}" "${bad_path}"
    fi
  fi

  if ! wget -c --timeout=60 --tries=20 --waitretry=10 --progress=dot:giga \
    -O "${part_path}" \
    "${BASE_URL}/${rel_path}"; then
    echo "Download interrupted; resumable partial kept at ${part_path}" >&2
    return 1
  fi

  if [[ -n "${expected_size}" ]]; then
    local part_size
    part_size="$(stat -c "%s" "${part_path}")"
    if [[ "${part_size}" -ne "${expected_size}" ]]; then
      echo "Downloaded partial has wrong size: ${rel_path} ${part_size}/${expected_size}" >&2
      return 1
    fi

    local part_sha
    part_sha="$(sha256sum "${part_path}" | awk '{print $1}')"
    if [[ "${part_sha}" != "${expected_sha}" ]]; then
      local bad_path="${part_path}.bad_$(date -u +%Y%m%d_%H%M%S)"
      echo "Downloaded partial has wrong SHA-256; quarantining ${part_path} -> ${bad_path}" >&2
      mv "${part_path}" "${bad_path}"
      return 1
    fi
  fi

  mv "${part_path}" "${out_path}"
  echo "Downloaded and verified: ${rel_path}"
}

preflight_network() {
  if [[ "${CDCHAT_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    echo "Skipping network preflight because CDCHAT_SKIP_PREFLIGHT=1"
    return 0
  fi

  echo "Checking network/proxy access: ${PREFLIGHT_URL}"
  if ! curl -fsSIL --max-time 20 "${PREFLIGHT_URL}" >/dev/null; then
    cat >&2 <<EOF
Network preflight failed before downloading large checkpoint files.
Check that the required proxy is listening and exported:
  export http_proxy=http://127.0.0.1:7890
  export https_proxy=http://127.0.0.1:7890
  export all_proxy=socks5://127.0.0.1:7890
EOF
    return 2
  fi
}

echo "CDChat revision: ${REVISION}"
echo "Base URL: ${BASE_URL}"
echo "http_proxy=${http_proxy}"
echo "https_proxy=${https_proxy}"

if [[ "${1:-}" == "--verify-only" ]]; then
  python "${ROOT_DIR}/scripts/verify_cdchat_weights.py"
  exit $?
fi

preflight_network

if [[ "$#" -gt 0 ]]; then
  FILES=("$@")
fi

for rel_path in "${FILES[@]}"; do
  download_file "${rel_path}"
done

python "${ROOT_DIR}/scripts/verify_cdchat_weights.py"
