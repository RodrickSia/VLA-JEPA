# scripts/
- `run_vlajepa_libero_ft.sh`, `run_vlajepa_libero_ft_future.sh`, `vlajepa_pretrain.sh`, `configs/` — training (original layout, unchanged)
- `pipelines/` — queued end-to-end jobs: `run_future_training_when_ready.sh` (wait for GPUs -> smoke test -> full training of the future model),
  `run_future_eval_after_training.sh` (after training: LIBERO + LIBERO-Plus, async delay 3 / horizon 4, on free GPU memory)
- `eval/` — evaluation launchers (LIBERO, LIBERO-Plus sync/async, resume helpers). Run from the repo root.
- `tools/` — data helpers: `download_libero.sh` (LeRobot LIBERO training datasets + `modality.json`, optionally the LIBERO code with `--with-code`; see `--help`), `download_libero_lerobot.sh` (older, machine-specific version)
- `_backups/` — `.bak_*` copies of edited scripts
- `MOVED_FILES.txt` — where every former top-level file went (and what was intentionally left at the top level)
Scripts write their own logs under `logs/` (see logs/README.md); launch them without `| tee`.
