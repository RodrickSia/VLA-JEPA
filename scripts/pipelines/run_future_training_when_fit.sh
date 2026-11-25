#!/bin/bash
# Waits until the VLA_JEPA_future training fits on 3, 2 or 1 of the CANDIDATES GPUs (as many as fit, 3 preferred),
# then runs a 2-update smoke test on those GPUs and, if it passes, the full training (settings in
# $CONFIG_FILE, default scripts/configs/vlajepa_libero_ft_future_from_libero.yaml). Global batch stays ~256: 3x43 / 2x64 / 1x128 accumulation.
# If the smoke test runs out of GPU memory (someone took memory meanwhile), it goes back to waiting.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future
CANDIDATES=${CANDIDATES:-5,6,7}
MULTI_GB=${MULTI_GB:-14}     # free GB per GPU with 2-3 GPUs (full finetune peaked ~19.6 GB; frozen backbone needs less. OOM in the smoke test -> back to waiting)
SINGLE_GB=${SINGLE_GB:-16}   # free GB on 1 GPU (full finetune would need ~26; only the small WM blocks have gradients here)
RUN_ID=${RUN_ID:-vlajepa_libero_ft_future_from_libero}
export CONFIG_FILE=${CONFIG_FILE:-$(pwd)/scripts/configs/vlajepa_libero_ft_future_from_libero.yaml}
WLOG=logs/future/training_waiter.txt
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$WLOG"; }

pick_gpus() {  # prints the chosen GPU list ("6,7"), or nothing
  local fits_multi=() best_single="" best_free=0 idx free
  while IFS=', ' read -r idx free; do
    [ "$free" -ge $((MULTI_GB * 1024)) ] && fits_multi+=("$idx")
    if [ "$free" -ge $((SINGLE_GB * 1024)) ] && [ "$free" -gt "$best_free" ]; then best_single=$idx; best_free=$free; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i "$CANDIDATES")
  if [ ${#fits_multi[@]} -ge 2 ]; then (IFS=,; echo "${fits_multi[*]:0:3}"); return; fi
  [ -n "$best_single" ] && echo "$best_single"
}

log "waiting until training fits on 3, 2 or 1 of GPUs $CANDIDATES (>= ${MULTI_GB} GB each for 2-3 GPUs, >= ${SINGLE_GB} GB for 1)"
n=0
while true; do
  GPUS=$(pick_gpus)
  if [ -z "$GPUS" ]; then
    [ $((n % 6)) = 0 ] && log "  no fit yet; free MiB on $CANDIDATES: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CANDIDATES" | tr '\n' ' ')"
    n=$((n + 1)); sleep 300; continue
  fi
  export CUDA_VISIBLE_DEVICES=$GPUS
  log "fits on GPU(s) $GPUS -> SMOKE TEST: 2 optimizer updates"
  rm -rf runs/smoke_vlajepa_libero_ft_future
  RUN_ID=smoke_vlajepa_libero_ft_future bash scripts/run_vlajepa_libero_ft_future.sh \
    --trainer.max_train_steps 2 --trainer.num_warmup_steps 0 --trainer.save_interval 1000000 --trainer.eval_interval 1000000 \
    > logs/future/smoke_train.txt 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -aqiE "final_model|Training complete|100%" logs/future/smoke_train.txt; then
    log "SMOKE TEST PASSED on GPU(s) $GPUS"; break
  fi
  if grep -aqiE "out of memory|CUDA error: out of memory|OutOfMemoryError" logs/future/smoke_train.txt; then
    log "SMOKE TEST ran out of GPU memory on $GPUS -> back to waiting"; sleep 300; continue
  fi
  log "SMOKE TEST FAILED (exit $rc) on $GPUS -> stopping; see logs/future/smoke_train.txt"
  exec bash
done

log "FULL TRAINING on GPU(s) $GPUS: run $RUN_ID, settings from $CONFIG_FILE"
RUN_ID=$RUN_ID bash scripts/run_vlajepa_libero_ft_future.sh > logs/future/train_${RUN_ID}.txt 2>&1
log "FULL TRAINING EXITED with code $? ; see logs/future/train_${RUN_ID}.txt"
exec bash
