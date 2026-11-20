#!/bin/bash
# Relaunch the 7 LIBERO-Plus clients against the servers that are running (Objects Layout shares port 14085).
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
export LIBERO_HOME=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$(pwd):${LIBERO_HOME}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6
export MAGICK_HOME=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/magick
export LD_LIBRARY_PATH=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/liberoplus/lib:$LD_LIBRARY_PATH
sim_python=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/liberoplus/bin/python
ckpt=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
folder=LIBERO_checkpoints_VLA-JEPA-LIBERO.pt
declare -A PORT=( ["Background Textures"]=14083 ["Camera Viewpoints"]=14084 ["Language Instructions"]=14085 ["Light Conditions"]=14086 ["Objects Layout"]=14085 ["Robot Initial States"]=14088 ["Sensor Noise"]=14089 )
for cat in "Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise"; do
  out="results/plus_libero_mix/${cat// /_}/${folder}"; mkdir -p "$out"
  $sim_python ./examples/LIBERO/eval_libero.py --args.pretrained-path $ckpt --args.host 127.0.0.1 --args.port ${PORT[$cat]} \
    --args.task-suite-name libero_mix --args.num-trials-per-task 1 --args.video-out-path "$out" \
    --args.category_value "$cat" --args.with_state true > "$out/eval.log" 2> "$out/eval.stderr.log" &
done
wait
echo "ALL CLIENTS FINISHED"; exec bash
