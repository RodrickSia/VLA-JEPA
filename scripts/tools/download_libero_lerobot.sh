#!/bin/bash
# Download the 4 LeRobot LIBERO datasets used by mixtures.py "libero_all" into the workspace (no home-dir cache).
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
W=$(pwd)
export HF_HOME=$W/.hf_cache HF_HUB_CACHE=$W/.hf_cache/hub HF_DATASETS_CACHE=$W/.hf_cache/datasets
export XDG_CACHE_HOME=$W/.hf_cache/xdg TMPDIR=$W/.hf_cache/tmp HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p $TMPDIR
for s in spatial object goal 10; do
  name=libero_${s}_no_noops_1.0.0_lerobot
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    /home/developer/miniconda3/envs/VLA_JEPA/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('IPEC-COMMUNITY/$name', repo_type='dataset', local_dir='data/libero_lerobot/$name', max_workers=8)
" && { echo "DONE $name"; break; } || { echo "attempt $attempt failed for $name, retrying in 60s"; sleep 60; }
  done
done
du -sh data/libero_lerobot/* ; ls data/libero_lerobot
echo "ALL DATASETS DOWNLOADED at $(date)"
exec bash
