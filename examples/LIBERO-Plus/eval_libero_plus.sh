#!/bin/bash


export LIBERO_HOME=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export sim_python=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/liberoplus/bin/python

your_ckpt=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise")
task_suite_name=libero_mix

host="127.0.0.1"
base_port=14082
unnorm_key="franka"
index=0
with_state="true"
gpus=(5 5 6 6 6 7 7) # one entry per perturbation server

for perturbation_name in "${items[@]}"
do
perturbation_file_name=${perturbation_name// /_}
index=$((index+1))
port=$((base_port+index))

CUDA_VISIBLE_DEVICES=${gpus[$((index-1))]} python ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda 0 &

# TODO pertubations should be modified in /home/dataset-assist-0/algorithm/ginwind/LIBERO-plus/libero/libero/benchmark/__init__.py
num_trials_per_task=1 # must be 1 for perturbation evaluation
video_out_path="results/plus_${task_suite_name}/${perturbation_file_name}/${folder_name}"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
mkdir -p ${video_out_path}

# export DEBUG=true

${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port ${port} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" > "${video_out_path}/eval.log" \
    --args.category_value "$perturbation_name" \
    --args.with_state "$with_state" &
done
    