#!/bin/bash
# SAM 2 and Sapiens2 source code at the exact commits used for the paper, installed in editable mode.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p external
if [ ! -d external/sam2 ]; then
  git clone https://github.com/facebookresearch/sam2.git external/sam2
  git -C external/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
fi
if [ ! -d external/sapiens2 ]; then
  git clone https://github.com/facebookresearch/sapiens2.git external/sapiens2
  git -C external/sapiens2 checkout 7e5bae88456ac418ff0e58e74106c9fe192055d4
fi
SAM2_BUILD_CUDA=0 $PY -m pip install -e external/sam2
$PY -m pip install -e external/sapiens2
