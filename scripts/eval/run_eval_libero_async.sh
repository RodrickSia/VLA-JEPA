#!/bin/bash
# Async action chunking LIBERO eval. Settings pass through to examples/LIBERO/eval_libero_async.sh, e.g.:
#   DELAY=3 EXEC_HORIZON=4 SUITES="libero_spatial" GPU=5 bash scripts/eval/run_eval_libero_async.sh
cd /mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
source /home/developer/miniconda3/bin/activate VLA_JEPA
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU:-5}
export LD_LIBRARY_PATH=/home/developer/miniconda3/envs/libero/lib:$LD_LIBRARY_PATH
bash ./examples/LIBERO/eval_libero_async.sh
wait
echo "ASYNC EVAL FINISHED (delay=${DELAY:-3}, execution_horizon=${EXEC_HORIZON:-4})"; exec bash
