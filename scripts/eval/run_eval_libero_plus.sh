#!/bin/bash
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
source /home/developer/miniconda3/bin/activate VLA_JEPA
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6
export MAGICK_HOME=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/magick
export LD_LIBRARY_PATH=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/envs/liberoplus/lib:$LD_LIBRARY_PATH
bash ./examples/LIBERO-Plus/eval_libero_plus.sh
wait
echo "EVAL FINISHED"; exec bash
