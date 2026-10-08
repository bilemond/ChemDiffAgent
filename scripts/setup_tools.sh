#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
THIRD_PARTY="${ROOT}/third_party"
mkdir -p "${THIRD_PARTY}"

clone_at() {
  local url=$1
  local directory=$2
  local revision=$3
  if [[ ! -d "${directory}/.git" ]]; then
    git clone "${url}" "${directory}"
  fi
  git -C "${directory}" fetch --all --tags
  git -C "${directory}" checkout --detach "${revision}"
}

clone_at https://github.com/OSU-NLP-Group/ChemToolAgent.git \
  "${THIRD_PARTY}/ChemToolAgent" 172f6b82d20f8ca0d325d5c5b1f1f32a7cb42077
clone_at https://github.com/AI4Chem/ChemistryAgent.git \
  "${THIRD_PARTY}/ChemistryAgent" 8d8ab2d4a5bc0dbc184b563cc273963a301004d1
clone_at https://github.com/HelloJocelynLu/t5chem.git \
  "${THIRD_PARTY}/t5chem" 06efc59edeace3005da435d1a118965c6b9c84e0
clone_at https://github.com/dptech-corp/Uni-Core.git \
  "${THIRD_PARTY}/Uni-Core" ace6fae1c8479a9751f2bb1e1d6e4047427bc134

echo "Optional chemistry tool sources are available under ${THIRD_PARTY}."
