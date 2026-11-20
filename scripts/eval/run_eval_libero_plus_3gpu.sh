#!/bin/bash
# LIBERO-Plus eval: 3 model servers (one per GPU 5/6/7), 7 perturbation clients sharing them.
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
mkdir -p logs/future logs/libero_plus logs/libero logs/data
source /home/developer/miniconda3/bin/activate VLA_JEPA
AC=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking
ckpt=$AC/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
declare -A GPU_OF_PORT=( [14083]=5 [14085]=6 [14088]=7 )
for port in 14083 14085 14088; do
  CUDA_VISIBLE_DEVICES=${GPU_OF_PORT[$port]} python ./deployment/model_server/server_policy.py --ckpt_path $ckpt --port $port --use_bf16 --cuda 0 > logs/libero_plus/sync_server_$port.txt 2>&1 &
done
export LIBERO_HOME=$AC/LIBERO-plus LIBERO_CONFIG_PATH=$AC/LIBERO-plus/libero
export PYTHONPATH=$(pwd):$LIBERO_HOME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 MAGICK_HOME=$AC/envs/magick
export LD_LIBRARY_PATH=$AC/envs/liberoplus/lib:$LD_LIBRARY_PATH
sim_python=$AC/envs/liberoplus/bin/python
declare -A PORT=( ["Background Textures"]=14083 ["Camera Viewpoints"]=14083 ["Language Instructions"]=14085 ["Light Conditions"]=14085 ["Objects Layout"]=14088 ["Robot Initial States"]=14088 ["Sensor Noise"]=14088 )
for cat in "Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise"; do
  out="results/plus_libero_mix/${cat// /_}/LIBERO_checkpoints_VLA-JEPA-LIBERO.pt"; mkdir -p "$out"
  $sim_python ./examples/LIBERO/eval_libero.py --args.pretrained-path $ckpt --args.host 127.0.0.1 --args.port ${PORT[$cat]} \
    --args.task-suite-name libero_mix --args.num-trials-per-task 1 --args.video-out-path "$out" \
    --args.category_value "$cat" --args.with_state true > "$out/eval.log" 2> "$out/eval.stderr.log" &
  CLIENTS="$CLIENTS $!"
done
wait $CLIENTS
echo "ALL CLIENTS FINISHED at $(date)"; exec bash
