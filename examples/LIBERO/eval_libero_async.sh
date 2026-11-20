#!/bin/bash
# LIBERO eval with (naive) async action chunking: examples/LIBERO/eval_libero_async.py
# Run from the repo root:  bash examples/LIBERO/eval_libero_async.sh
# Override settings from the command line, e.g.:
#   DELAY=0 EXEC_HORIZON=7 bash examples/LIBERO/eval_libero_async.sh   # sync sanity check (= eval_libero.sh)
#   DELAY=3 EXEC_HORIZON=4 GPU=6 bash examples/LIBERO/eval_libero_async.sh

export PYTHONDONTWRITEBYTECODE=1
export LIBERO_HOME=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/LIBERO # your LIBERO code path
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export sim_python=/home/developer/miniconda3/envs/libero/bin/python # your LIBERO conda path

your_ckpt=${CKPT:-/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt}
RESULTS_ROOT=${RESULTS_ROOT:-results_async}
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# ---- async action chunking settings (DELAY + EXEC_HORIZON must be <= 7, the chunk size) ----
delay=${DELAY:-3}                 # d: inference latency in control steps
execution_horizon=${EXEC_HORIZON:-4}  # s: new actions executed per chunk
if [ $((delay + execution_horizon)) -gt 7 ]; then
    echo "DELAY ($delay) + EXEC_HORIZON ($execution_horizon) must be <= 7"; exit 1
fi
if [ "$delay" -gt "$execution_horizon" ]; then
    echo "DELAY ($delay) must be <= EXEC_HORIZON ($execution_horizon)"; exit 1
fi

if [ -n "${SUITES:-}" ]; then read -r -a items <<< "$SUITES"; else items=("libero_10" "libero_goal" "libero_object" "libero_spatial"); fi
host="127.0.0.1"
base_port=${BASE_PORT:-16083}     # different from eval_libero.sh (15083) and LIBERO-Plus (14083+)
index=0
gpu=${GPU:-5}                     # default GPU
read -r -a gpus <<< "${GPUS:-$gpu}"  # GPUS="5 6": suites are spread round-robin over these GPUs
num_trials_per_task=${TRIALS:-50}
with_state="true"

echo "async chunking: delay=${delay} execution_horizon=${execution_horizon} | suites: ${items[*]} | GPUs ${gpus[*]}"

# start one server + one client per task suite
for task_suite_name in "${items[@]}"
do
    index=$((index+1))
    port=$((base_port+index))
    suite_gpu=${gpus[$(( (index-1) % ${#gpus[@]} ))]}
    echo "  ${task_suite_name}: server port ${port} on GPU ${suite_gpu}"

    CUDA_VISIBLE_DEVICES=${suite_gpu} python ./deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        --cuda 0 &
    SERVERS="$SERVERS $!"

    video_out_path="${RESULTS_ROOT}/d${delay}_s${execution_horizon}/${task_suite_name}/${folder_name}"
    mkdir -p ${video_out_path}

    ${sim_python} ./examples/LIBERO/eval_libero_async.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port ${port} \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.with_state "$with_state" \
        --args.delay ${delay} \
        --args.execution-horizon ${execution_horizon} > "${video_out_path}/eval.log" 2>&1 &
    CLIENTS="$CLIENTS $!"
done

# wait for all clients, then stop the servers this script started
wait $CLIENTS
echo "all suites finished; stopping servers:$SERVERS"
kill $SERVERS 2>/dev/null
grep -h "Total success rate" -A1 ${RESULTS_ROOT}/d${delay}_s${execution_horizon}/*/${folder_name}/eval.log 2>/dev/null
