#!/bin/bash
# Waits for (1) the LIBERO LeRobot download and (2) >= MIN_FREE_GB free on each training GPU (GPUS, default 6,7),
# then runs a 2-step smoke test of the VLA_JEPA_future training, and only if it passes, the full training.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
MIN_FREE_GB=${MIN_FREE_GB:-13}
# Paper settings (steps 30K, warmup, LRs, batch 2/GPU) live in scripts/configs/vlajepa_libero_ft_future.yaml;
# global batch = 2/GPU x 2 GPUs x 64 accumulation = 256 (DeepSpeed config picked by GPU count in run_vlajepa_libero_ft_future.sh).
export CUDA_VISIBLE_DEVICES=${GPUS:-6,7}
RUN_ID=${RUN_ID:-vlajepa_libero_ft_future_paper}
COMMON=""
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a logs/future/training_waiter.txt; }

log "waiting for dataset download"
until grep -aq "ALL DATASETS DOWNLOADED" logs/data/download_libero_lerobot.txt 2>/dev/null; do sleep 60; done
log "datasets ready: $(ls data/libero_lerobot | tr '\n' ' ')"

log "waiting for >= ${MIN_FREE_GB} GB free on each of GPUs ${CUDA_VISIBLE_DEVICES}"
n=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} | tr '\n' ' ')
  ok=1; for f in $free; do [ "$f" -ge $((MIN_FREE_GB * 1024)) ] || ok=0; done
  [ $ok = 1 ] && break
  [ $((n % 6)) = 0 ] && log "  free MiB on ${CUDA_VISIBLE_DEVICES}: $free"
  n=$((n + 1)); sleep 300
done
log "GPUs free enough (MiB on ${CUDA_VISIBLE_DEVICES}: $free)"

log "SMOKE TEST: 2 optimizer steps"
RUN_ID=smoke_vlajepa_libero_ft_future bash scripts/run_vlajepa_libero_ft_future.sh $COMMON \
  --trainer.max_train_steps 2 --trainer.num_warmup_steps 0 --trainer.save_interval 1000000 --trainer.eval_interval 1000000 \
  > logs/future/smoke_train.txt 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -aqiE "final_model|Training complete|completed_steps|100%" logs/future/smoke_train.txt; then
  log "SMOKE TEST FAILED (exit $rc) -> not starting full training; see logs/future/smoke_train.txt"
  exec bash
fi
log "SMOKE TEST PASSED"

log "FULL TRAINING: run $RUN_ID, settings from scripts/configs/vlajepa_libero_ft_future.yaml"
RUN_ID=$RUN_ID bash scripts/run_vlajepa_libero_ft_future.sh $COMMON \
  > logs/future/train_${RUN_ID}.txt 2>&1
log "FULL TRAINING EXITED with code $? ; see logs/future/train_${RUN_ID}.txt"
exec bash
