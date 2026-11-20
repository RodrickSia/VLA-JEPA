#!/bin/bash
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
source /home/developer/miniconda3/bin/activate VLA_JEPA
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5
export LD_LIBRARY_PATH=/home/developer/miniconda3/envs/libero/lib:$LD_LIBRARY_PATH
bash ./examples/LIBERO/eval_libero.sh
wait
echo "EVAL FINISHED"; exec bash
