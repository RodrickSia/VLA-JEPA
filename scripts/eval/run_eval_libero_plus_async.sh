#!/bin/bash
# LIBERO-Plus eval with async action chunking (examples/LIBERO/eval_libero_async.py).
#   DELAY=3 EXEC_HORIZON=4 SERVER_GPUS="6 6 5" bash scripts/eval/run_eval_libero_plus_async.sh
# One model server per entry in SERVER_GPUS; the 7 perturbation categories are spread round-robin over them.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
source /home/developer/miniconda3/bin/activate VLA_JEPA
AC=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking
ckpt=${CKPT:-$AC/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt}
folder=$(echo "$ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
delay=${DELAY:-3}; execution_horizon=${EXEC_HORIZON:-4}
if [ $((delay + execution_horizon)) -gt 7 ] || [ "$delay" -gt "$execution_horizon" ]; then
  echo "need DELAY <= EXEC_HORIZON and DELAY + EXEC_HORIZON <= 7 (got $delay, $execution_horizon)"; exit 1; fi
read -r -a server_gpus <<< "${SERVER_GPUS:-6 6 5}"
base_port=${BASE_PORT:-17083}

ports=()
for i in "${!server_gpus[@]}"; do
  port=$((base_port + i)); ports+=($port)
  CUDA_VISIBLE_DEVICES=${server_gpus[$i]} python ./deployment/model_server/server_policy.py --ckpt_path $ckpt --port $port --use_bf16 --cuda 0 > logs/libero_plus/async_server_$port.txt 2>&1 &
  SERVERS="$SERVERS $!"
  echo "server port $port on GPU ${server_gpus[$i]}"
done

export LIBERO_HOME=$AC/LIBERO-plus LIBERO_CONFIG_PATH=$AC/LIBERO-plus/libero
export PYTHONPATH=$(pwd):$LIBERO_HOME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${EGL_GPU:-${server_gpus[0]}} MAGICK_HOME=$AC/envs/magick
export LD_LIBRARY_PATH=$AC/envs/liberoplus/lib:$LD_LIBRARY_PATH
sim_python=$AC/envs/liberoplus/bin/python

i=0
for cat in "Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise"; do
  port=${ports[$(( i % ${#ports[@]} ))]}; i=$((i+1))
  out="${RESULTS_ROOT:-results_async_plus}/d${delay}_s${execution_horizon}/${cat// /_}/$folder"; mkdir -p "$out"
  echo "  $cat -> port $port"
  $sim_python ./examples/LIBERO/eval_libero_async.py --args.pretrained-path $ckpt --args.host 127.0.0.1 --args.port $port \
    --args.task-suite-name libero_mix --args.num-trials-per-task 1 --args.video-out-path "$out" \
    --args.category_value "$cat" --args.with_state true \
    --args.delay $delay --args.execution-horizon $execution_horizon > "$out/eval.log" 2> "$out/eval.stderr.log" &
  CLIENTS="$CLIENTS $!"
done
wait $CLIENTS
echo "ALL ASYNC LIBERO-PLUS CLIENTS FINISHED at $(date) (delay=$delay, execution_horizon=$execution_horizon)"
kill $SERVERS 2>/dev/null
[ -n "${NO_EXEC:-}" ] || exec bash
