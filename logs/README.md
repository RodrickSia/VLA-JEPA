# logs/
- `future/`      VLA_JEPA_future: `training_waiter.txt` (queue + smoke test + training status), `smoke_train.txt`, `train.txt`, `eval_pipeline.txt`, `eval.txt`
- `libero_plus/` LIBERO-Plus runs: `sync_server_*`, `async_server_*`, `resume1_*` logs, `eval_*.txt`, `resume_points.json`
- `libero/`      plain LIBERO eval logs
- `data/`        dataset download logs
- `<timestamp>/` older per-run folders from eval_libero.sh
Per-episode eval logs and videos are NOT here: they live next to the results (`results*/`).
