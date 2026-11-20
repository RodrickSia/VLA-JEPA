#!/usr/bin/env bash
# Download LIBERO for VLA-JEPA.
#   1. the four LeRobot-format LIBERO datasets used for training (mixture "libero_all"):
#        IPEC-COMMUNITY/libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot
#      and copy examples/LIBERO/modality.json into each dataset's meta/ (required by the data loader)
#   2. optionally the LIBERO benchmark code, which is needed to run the simulator evaluation
#
# Usage (from anywhere):
#   bash scripts/tools/download_libero.sh                          # datasets -> <repo>/data/libero_lerobot
#   bash scripts/tools/download_libero.sh --suites "spatial 10"    # only some suites
#   bash scripts/tools/download_libero.sh --with-code              # also clone LIBERO -> <repo>/../LIBERO
#   bash scripts/tools/download_libero.sh --dry-run                # print what would happen, download nothing
#
# Options (or the matching environment variable):
#   --data-dir DIR   (DATA_DIR)  where the datasets go            [default: <repo>/data/libero_lerobot]
#   --suites "..."   (SUITES)    subset of: spatial object goal 10 [default: all four]
#   --with-code                  also clone the LIBERO benchmark code
#   --code-dir DIR   (CODE_DIR)  where to clone LIBERO            [default: <repo>/../LIBERO]
#   --include GLOB               only fetch files matching GLOB (e.g. "meta/*"); the dataset is then NOT marked complete
#   --force                      re-check datasets that are already marked complete
#   --dry-run                    only print the plan
# Environment: PYTHON (interpreter with huggingface_hub, default python), RETRIES (default 5),
#              HF_TOKEN, HF_ENDPOINT, HF_HOME (Hugging Face cache) are honoured as usual.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/libero_lerobot}
SUITES=${SUITES:-"spatial object goal 10"}
CODE_DIR=${CODE_DIR:-$REPO_ROOT/../LIBERO}
PYTHON=${PYTHON:-python}
RETRIES=${RETRIES:-5}
WITH_CODE=0; FORCE=0; DRY=0; INCLUDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir)  DATA_DIR=$2; shift 2 ;;
    --suites)    SUITES=$2; shift 2 ;;
    --with-code) WITH_CODE=1; shift ;;
    --code-dir)  CODE_DIR=$2; shift 2 ;;
    --include)   INCLUDE=$2; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

log() { echo "[download_libero] $*"; }

for s in $SUITES; do
  case "$s" in spatial|object|goal|10) ;; *) echo "unknown suite '$s' (use: spatial object goal 10)" >&2; exit 2 ;; esac
done

if [ "$DRY" = 0 ]; then
  "$PYTHON" -c "import huggingface_hub" 2>/dev/null || { echo "huggingface_hub not found for '$PYTHON' (pip install huggingface_hub, or set PYTHON=...)" >&2; exit 1; }
fi

[ "$DRY" = 1 ] || mkdir -p "$DATA_DIR"
log "datasets -> $DATA_DIR | suites: $SUITES${INCLUDE:+ | only files matching '$INCLUDE'}"

fetch() {   # fetch REPO_ID LOCAL_DIR  (retries; resumes partial downloads)
  local repo=$1 dir=$2 attempt
  for attempt in $(seq 1 "$RETRIES"); do
    if REPO=$repo DIR=$dir INCLUDE=$INCLUDE "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
inc = os.environ.get("INCLUDE") or None
snapshot_download(os.environ["REPO"], repo_type="dataset", local_dir=os.environ["DIR"], max_workers=8,
                  allow_patterns=[inc] if inc else None)
PY
    then return 0; fi
    log "  attempt $attempt/$RETRIES failed for $repo, retrying in 30s"; sleep 30
  done
  return 1
}

for s in $SUITES; do
  name=libero_${s}_no_noops_1.0.0_lerobot
  dir=$DATA_DIR/$name
  marker=$dir/.vla_jepa_download_complete
  if [ "$DRY" = 1 ]; then log "[dry-run] would download IPEC-COMMUNITY/$name -> $dir and add meta/modality.json"; continue; fi
  if [ -f "$marker" ] && [ "$FORCE" = 0 ] && [ -z "$INCLUDE" ]; then log "$name: already downloaded (use --force to re-check)"; else
    log "$name: downloading ..."
    fetch "IPEC-COMMUNITY/$name" "$dir" || { echo "FAILED: $name after $RETRIES attempts" >&2; exit 1; }
    [ -z "$INCLUDE" ] && touch "$marker"
  fi
  mkdir -p "$dir/meta"
  if [ ! -f "$dir/meta/modality.json" ]; then
    cp "$REPO_ROOT/examples/LIBERO/modality.json" "$dir/meta/modality.json" && log "$name: added meta/modality.json"
  elif ! cmp -s "$REPO_ROOT/examples/LIBERO/modality.json" "$dir/meta/modality.json"; then
    log "$name: WARNING meta/modality.json differs from examples/LIBERO/modality.json (left unchanged)"
  fi
done

if [ "$WITH_CODE" = 1 ]; then
  if [ "$DRY" = 1 ]; then log "[dry-run] would clone https://github.com/Lifelong-Robot-Learning/LIBERO.git -> $CODE_DIR"
  elif [ -d "$CODE_DIR/libero" ]; then log "LIBERO code already present: $CODE_DIR"
  else
    command -v git >/dev/null || { echo "git is required for --with-code" >&2; exit 1; }
    log "cloning LIBERO code -> $CODE_DIR"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$CODE_DIR"
  fi
  log "LIBERO code needs its own environment; follow https://github.com/Lifelong-Robot-Learning/LIBERO, then:"
  log "  export LIBERO_HOME=$CODE_DIR     # used by examples/LIBERO/eval_libero.sh"
fi

[ "$DRY" = 1 ] && log "dry run finished" || log "done. Training data root: $DATA_DIR (data_mix: libero_all)"
