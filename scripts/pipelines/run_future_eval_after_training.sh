#!/bin/bash
# Evaluation pipeline for the VLA_JEPA_future model. It does NOT start or stop anything that is running.
#   1. waits for the training started by run_future_training_when_ready.sh (reads its log) to finish
#   2. evaluates the final checkpoint, only on GPUs 5-7 and only with memory that is actually free:
#        libero_async  LIBERO 4 suites, naive async, delay 3 / horizon 4          (default)
#        plus_async    LIBERO-Plus (7 perturbation categories), async, delay 3 / horizon 4   (default)
#   The future model is ALWAYS evaluated with delay 3 / execution horizon 4 (no no-delay stages exist). STAGES="libero_async plus_async" (default).
#   `--check` only prints the free-memory slots and the checkpoint it would use, then exits.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
source /home/developer/miniconda3/bin/activate VLA_JEPA
export TMPDIR=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/tmp
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/developer/miniconda3/envs/libero/lib:$LD_LIBRARY_PATH

RUN_ID=${RUN_ID:-vlajepa_libero_ft_future}
CKPT=${CKPT:-$(pwd)/runs/$RUN_ID/final_model/pytorch_model.pt}
STAGES=${STAGES:-"libero_async plus_async"}
SLOT_MB=${SLOT_MB:-11264}            # memory reserved per model server (future model also runs V-JEPA at inference)
WLOG=logs/future/training_waiter.txt
LOG=logs/future/eval_pipeline.txt
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

# one entry per free SLOT_MB on GPUs 5,6,7, e.g. "7 7 6"
slots() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 5,6,7 \
          | awk -F', *' -v s="$SLOT_MB" '{n=int($2/s); for(k=0;k<n;k++) printf "%s ", $1}'; }

if [ "${1:-}" = "--check" ]; then echo "slots now: [$(slots)]  (one per ${SLOT_MB} MiB free)"; echo "checkpoint: $CKPT"; echo "stages: $STAGES"; exit 0; fi

# ---------------------------------------------------------------- 1. wait for training
base_lines=$(wc -l < "$WLOG")
log "pipeline started; waiting for training to finish (waiter log: $WLOG, ckpt: $CKPT)"
while true; do
  new=$(tail -n +$((base_lines + 1)) "$WLOG")
  if echo "$new" | grep -q "SMOKE TEST FAILED"; then log "ABORT: smoke test failed, no training -> nothing to evaluate"; exec bash; fi
  if echo "$new" | grep -q "FULL TRAINING EXITED"; then break; fi
  sleep 120
done
log "training exited: $(echo "$new" | grep 'FULL TRAINING EXITED' | tail -1)"
if [ ! -f "$CKPT" ]; then log "ABORT: $CKPT not found (training did not finish) -> check logs/future/train.txt"; exec bash; fi

# ---------------------------------------------------------------- 2. evaluation on free GPU memory
LIBERO_SUITES=(libero_10 libero_goal libero_object libero_spatial)
port=19100

run_libero() {   # $1 = delay, $2 = execution horizon
  local remaining=("${LIBERO_SUITES[@]}") sl n batch gpus
  while [ ${#remaining[@]} -gt 0 ]; do
    read -r -a sl <<< "$(slots)"
    if [ ${#sl[@]} -eq 0 ]; then log "  no GPU among 5-7 has $SLOT_MB MiB free yet, waiting"; sleep 300; continue; fi
    n=${#sl[@]}; [ $n -gt ${#remaining[@]} ] && n=${#remaining[@]}
    batch=("${remaining[@]:0:$n}"); remaining=("${remaining[@]:$n}"); gpus="${sl[*]:0:$n}"
    log "  d=$1 s=$2 suites [${batch[*]}] on GPUs [$gpus]"
    CKPT=$CKPT SUITES="${batch[*]}" GPUS="$gpus" DELAY=$1 EXEC_HORIZON=$2 RESULTS_ROOT=results_future BASE_PORT=$port \
      MUJOCO_EGL_DEVICE_ID=${sl[0]} bash examples/LIBERO/eval_libero_async.sh >> logs/future/eval.txt 2>&1
    port=$((port + 10))
  done
}

run_plus() {     # $1 = delay, $2 = execution horizon
  local sl
  while true; do
    read -r -a sl <<< "$(slots)"; [ ${#sl[@]} -gt 0 ] && break
    log "  no GPU among 5-7 has $SLOT_MB MiB free yet, waiting"; sleep 300
  done
  [ ${#sl[@]} -gt 3 ] && sl=("${sl[@]:0:3}")
  log "  LIBERO-Plus d=$1 s=$2 servers on GPUs [${sl[*]}]"
  CKPT=$CKPT SERVER_GPUS="${sl[*]}" EGL_GPU=${sl[0]} DELAY=$1 EXEC_HORIZON=$2 RESULTS_ROOT=results_future_plus BASE_PORT=$port NO_EXEC=1 \
    bash scripts/eval/run_eval_libero_plus_async.sh >> logs/future/eval.txt 2>&1
  port=$((port + 10))
}

for st in $STAGES; do
  log "STAGE $st START"
  case $st in
    libero_async) run_libero 3 4 ;;
    plus_async)   run_plus 3 4 ;;
    *) log "unknown stage $st" ;;
  esac
  log "STAGE $st DONE"
done
log "FUTURE EVALUATION PIPELINE FINISHED (results: results_future/, results_future_plus/)"
exec bash
