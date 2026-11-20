#!/bin/bash
# Resume the interrupted LIBERO-Plus runs (sync Sensor Noise + async d3/s4 for all 7 categories) from where they stopped.
# Start points come from resume_points.json (completed-task counts read from the old logs). Old eval.log files are kept;
# each resumed chunk writes its own resume_<start>_<end>.log. Slow Sensor Noise is split into 3 parallel chunks per run.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
source /home/developer/miniconda3/bin/activate VLA_JEPA
AC=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking
export TMPDIR=$AC/tmp            # the container's / is full
ckpt=$AC/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
folder=LIBERO_checkpoints_VLA-JEPA-LIBERO.pt
read -r -a gpus <<< "${SERVER_GPUS:-7}"; ports=(); for i in "${!gpus[@]}"; do ports+=($((18083+i))); done
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} python ./deployment/model_server/server_policy.py --ckpt_path $ckpt --port ${ports[$i]} --use_bf16 --cuda 0 > logs/libero_plus/resume1_server_${ports[$i]}.txt 2>&1 &
  SERVERS="$SERVERS $!"; echo "server ${ports[$i]} on GPU ${gpus[$i]}"
done
export LIBERO_HOME=$AC/LIBERO-plus LIBERO_CONFIG_PATH=$AC/LIBERO-plus/libero
export PYTHONPATH=$(pwd):$LIBERO_HOME
export MUJOCO_GL=egl MAGICK_HOME=$AC/envs/magick
export LD_LIBRARY_PATH=$AC/envs/liberoplus/lib:$LD_LIBRARY_PATH
sim_python=$AC/envs/liberoplus/bin/python
read -r -a egl <<< "${EGL_GPUS:-6 7}"; n=0
# mode  category                 start  end(-1=to end)
JOBS=(
"sync  Sensor_Noise            1171 1314"
"sync  Sensor_Noise            1314 1458"
"sync  Sensor_Noise            1458 -1"
"async Background_Textures     699  -1"
"async Camera_Viewpoints       434  -1"
"async Language_Instructions   841  -1"
"async Light_Conditions        722  -1"
"async Objects_Layout          627  -1"
"async Robot_Initial_States    555  -1"
"async Sensor_Noise            175  650"
"async Sensor_Noise            650  1125"
"async Sensor_Noise            1125 -1"
)
for job in "${JOBS[@]}"; do
  read -r mode cat start end <<< "$job"
  port=${ports[$(( n % ${#ports[@]} ))]}; export MUJOCO_EGL_DEVICE_ID=${egl[$(( n % ${#egl[@]} ))]}; n=$((n+1))
  catname=${cat//_/ }
  if [ $mode = sync ]; then out=results/plus_libero_mix/$cat/$folder; script=eval_libero.py; extra=""
  else out=results_async_plus/d3_s4/$cat/$folder; script=eval_libero_async.py; extra="--args.delay 3 --args.execution-horizon 4"; fi
  mkdir -p $out
  echo "  $mode $cat [$start,$end) -> port $port, EGL $MUJOCO_EGL_DEVICE_ID"
  $sim_python ./examples/LIBERO/$script --args.pretrained-path $ckpt --args.host 127.0.0.1 --args.port $port \
    --args.task-suite-name libero_mix --args.num-trials-per-task 1 --args.video-out-path "$out" \
    --args.category_value "$catname" --args.with_state true --args.start-index $start --args.end-index $end $extra \
    > "$out/resume_${start}_${end}.log" 2> "$out/resume_${start}_${end}.stderr.log" &
  CLIENTS="$CLIENTS $!"
done
wait $CLIENTS
echo "ALL RESUMED CLIENTS FINISHED at $(date)"
kill $SERVERS 2>/dev/null
exec bash
