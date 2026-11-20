#!/bin/bash
# Waits for (1) the LIBERO LeRobot download and (2) >= MIN_FREE_GB free on each of GPUs 5,6,7,
# then runs a 2-step smoke test of the VLA_JEPA_future training, and only if it passes, the full training.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
MIN_FREE_GB=${MIN_FREE_GB:-13}
MAX_STEPS=${MAX_STEPS:-5000}          # optimizer steps; global batch = 2/GPU x 3 GPUs x 10 accumulation = 60
COMMON="--datasets.vla_data.per_device_batch_size 2"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a logs/future/training_waiter.txt; }

log "waiting for dataset download"
until grep -aq "ALL DATASETS DOWNLOADED" logs/data/download_libero_lerobot.txt 2>/dev/null; do sleep 60; done
log "datasets ready: $(ls data/libero_lerobot | tr '\n' ' ')"

log "waiting for >= ${MIN_FREE_GB} GB free on each of GPUs 5,6,7"
n=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5,6,7 | tr '\n' ' ')
  ok=1; for f in $free; do [ "$f" -ge $((MIN_FREE_GB * 1024)) ] || ok=0; done
  [ $ok = 1 ] && break
  [ $((n % 6)) = 0 ] && log "  free MiB on 5/6/7: $free"
  n=$((n + 1)); sleep 300
done
log "GPUs free enough (MiB 5/6/7: $free)"

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

log "FULL TRAINING: ${MAX_STEPS} optimizer steps"
RUN_ID=vlajepa_libero_ft_future bash scripts/run_vlajepa_libero_ft_future.sh $COMMON \
  --trainer.max_train_steps ${MAX_STEPS} --trainer.num_warmup_steps 500 --trainer.save_interval 1000 --trainer.eval_interval 500 \
  > logs/future/train.txt 2>&1
log "FULL TRAINING EXITED with code $? ; see logs/future/train.txt"
exec bash
