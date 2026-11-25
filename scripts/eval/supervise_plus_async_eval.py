"""LIBERO-Plus async eval (delay 3 / execution horizon 4) of one checkpoint, with crash recovery.

Starts one model server per entry of SERVER_GPUS, splits every perturbation category into CHUNKS index ranges,
runs one sim client per chunk, and every 2 minutes relaunches any client that died, from the task after the last
completed one (same idea as supervise_async_plus.py). Stops the servers when every chunk has its final summary.

  CKPT=/path/run/checkpoints/x.pt SERVER_GPUS="5 5 6 6 7" EGL_GPUS="5 6 7" CHUNKS=2 \
    python3 -I scripts/eval/supervise_plus_async_eval.py

Results: results_future_plus/d3_s4/<Category>/<run>_checkpoints_<file>/ (eval logs + rollout videos).
"""
import glob, os, re, socket, subprocess, time

R = "/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA/"
AC = "/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking"
CKPT = os.environ["CKPT"]
parts = CKPT.rstrip("/").split("/")
FOLDER = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"          # same naming as run_eval_libero_plus_async.sh
SERVER_GPUS = os.environ.get("SERVER_GPUS", "5 5 6 6 7").split()
EGL = os.environ.get("EGL_GPUS", "5 6 7").split()
CHUNKS = int(os.environ.get("CHUNKS", "2"))
BASE_PORT = int(os.environ.get("BASE_PORT", "19200"))
RESULTS = R + os.environ.get("RESULTS_ROOT", "results_future_plus") + "/d3_s4"
SUPERVISE_ONLY = os.environ.get("SUPERVISE_ONLY") == "1"   # take over a running eval: no new servers, no initial launch
# CPU threads per model server: default PyTorch threading made V-JEPA preprocessing ~100x slower on this host
SERVER_THREADS = os.environ.get("SERVER_THREADS", "4")
MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS", "0"))   # >0: at most this many clients at once; other chunks wait in a queue
DELAY, HORIZON = 3, 4
MAX_RESTARTS = 5
# task counts per category in libero_mix (measured on the full released-model eval)
SIZES = {"Background_Textures": 1076, "Camera_Viewpoints": 1599, "Language_Instructions": 1537,
         "Light_Conditions": 1138, "Objects_Layout": 1525, "Robot_Initial_States": 1550, "Sensor_Noise": 1601}
N_RE = re.compile(r"# episodes completed so (?:\S+\.py:\d+ )?far: (?:\S+\.py:\d+ )?(\d+)")
PORTS = [BASE_PORT + i for i in range(len(SERVER_GPUS))]

env = dict(os.environ)
env.update({
    "TMPDIR": AC + "/tmp", "LIBERO_HOME": AC + "/LIBERO-plus", "LIBERO_CONFIG_PATH": AC + "/LIBERO-plus/libero",
    "PYTHONPATH": R + ":" + AC + "/LIBERO-plus", "MUJOCO_GL": "egl", "MAGICK_HOME": AC + "/envs/magick",
    "LD_LIBRARY_PATH": AC + "/envs/liberoplus/lib:" + env.get("LD_LIBRARY_PATH", ""),
})
SIM_PY = AC + "/envs/liberoplus/bin/python"
SERVER_PY = "/home/developer/miniconda3/envs/VLA_JEPA/bin/python"


def log(msg):
    print(time.strftime("%m-%d %H:%M ") + msg, flush=True)


def cmdlines():
    out = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            out.append((int(p.split("/")[2]), open(p, "rb").read().replace(b"\0", b" ").decode(errors="replace")))
        except OSError:
            pass
    return out


def port_open(port):
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def completed(path):
    t = re.sub(r"\s+", " ", open(path, "rb").read().decode(errors="replace"))
    n = N_RE.findall(t)
    return int(n[-1]) if n else 0, "Total episodes" in t


def launch(cat, start, end, port, egl):
    out = f"{RESULTS}/{cat}/{FOLDER}"
    os.makedirs(out, exist_ok=True)
    cmd = [SIM_PY, "./examples/LIBERO/eval_libero_async.py", "--args.pretrained-path", CKPT, "--args.host", "127.0.0.1",
           "--args.port", str(port), "--args.task-suite-name", "libero_mix", "--args.num-trials-per-task", "1",
           "--args.video-out-path", out, "--args.category_value", cat.replace("_", " "), "--args.with_state", "true",
           "--args.start-index", str(start), "--args.end-index", str(end),
           "--args.delay", str(DELAY), "--args.execution-horizon", str(HORIZON)]
    subprocess.Popen(cmd, cwd=R, env=dict(env, MUJOCO_EGL_DEVICE_ID=str(egl)), stdin=subprocess.DEVNULL,
                     stdout=open(f"{out}/resume_{start}_{end}.log", "w"), stderr=open(f"{out}/resume_{start}_{end}.stderr.log", "w"))


# ---- 1. model servers
os.makedirs(R + "logs/future", exist_ok=True)
servers = []
for gpu, port in ([] if SUPERVISE_ONLY else zip(SERVER_GPUS, PORTS)):
    if port_open(port):
        raise SystemExit(f"port {port} already in use; set BASE_PORT")
    p = subprocess.Popen([SERVER_PY, "./deployment/model_server/server_policy.py", "--ckpt_path", CKPT, "--port", str(port),
                          "--use_bf16", "--cuda", "0"], cwd=R, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=R, OMP_NUM_THREADS=SERVER_THREADS, MKL_NUM_THREADS=SERVER_THREADS),
                         stdin=subprocess.DEVNULL, stdout=open(f"{R}logs/future/plus_eval_server_{port}.txt", "w"),
                         stderr=subprocess.STDOUT)
    servers.append(p)
    log(f"server port {port} on GPU {gpu} (pid {p.pid})")
log(f"checkpoint {CKPT} -> results {RESULTS}/<category>/{FOLDER}")
while not SUPERVISE_ONLY and not all(port_open(p) for p in PORTS):
    dead = [s for s in servers if s.poll() is not None]
    if dead:
        raise SystemExit(f"a model server exited (code {dead[0].returncode}); see logs/future/plus_eval_server_*.txt")
    time.sleep(30)
log("all model servers are up")

# ---- 2. chunks: (category, start, end, port, egl)
jobs = []
if os.environ.get("JOBS_FILE"):   # explicit chunk list, e.g. after splitting slow chunks
    for line in open(os.environ["JOBS_FILE"]):
        if line.strip() and not line.startswith("#"):
            c, a, b, pt, eg = line.split()
            jobs.append((c, int(a), int(b), int(pt), eg))
else:
    for cat, n in SIZES.items():
        edges = [round(n * k / CHUNKS) for k in range(CHUNKS + 1)]
        jobs += [(cat, edges[k], edges[k + 1]) for k in range(CHUNKS)]
    jobs = [(c, a, b, PORTS[i % len(PORTS)], EGL[i % len(EGL)]) for i, (c, a, b) in enumerate(jobs)]
PORTS = sorted({j[3] for j in jobs}) if SUPERVISE_ONLY else PORTS
if SUPERVISE_ONLY:
    log(f"supervise-only mode: taking over {len(jobs)} chunks on ports {PORTS}")
else:
    for cat, s, e, port, egl in jobs:
        launch(cat, s, e, port, egl)
    log(f"started {len(jobs)} clients ({CHUNKS} chunks per category) on {len(PORTS)} servers")

# ---- 3. supervise
fails = {}
first = True
while True:
    if not first:
        time.sleep(120)
    first = False
    t0 = time.time()
    # only THIS checkpoint's clients: another eval with the same chunk ranges (e.g. a paused epoch-2 eval) must not count
    procs = [c for _, c in cmdlines() if c.split(" ", 1)[0].endswith("python") and "eval_libero_async.py" in c
             and f"--args.pretrained-path {CKPT} " in c]
    finished = 0
    running = len(procs)
    for cat, s, e, port, egl in jobs:
        d = f"{RESULTS}/{cat}/{FOLDER}/"
        cands = [(int(re.search(r"resume_(\d+)_", f).group(1)), f) for f in glob.glob(f"{d}resume_*_{e}.log")]
        cands = [(x, f) for x, f in cands if s <= x < e]
        if not cands:   # chunk never started (e.g. a new piece of a split chunk): start it
            if MAX_CLIENTS and running >= MAX_CLIENTS:
                continue   # queued: starts when a client slot frees up
            running += 1
            log(f"{cat} [{s},{e}): no log yet -> starting on port {port}")
            launch(cat, s, e, port, egl)
            continue
        x, f = max(cands)
        n, done = completed(f)
        if done or x + n >= e:
            finished += 1
            continue
        if any(f"--args.category_value {cat.replace('_', ' ')} " in c and f"--args.start-index {x} --args.end-index {e}" in c
               for c in procs):
            continue
        if MAX_CLIENTS and running >= MAX_CLIENTS:
            continue   # queued; do not count waiting as a failed relaunch
        key = (cat, s, e)
        fails[key] = fails.get(key, 0) + (1 if n == 0 else 0)
        if fails[key] > MAX_RESTARTS:
            if fails[key] == MAX_RESTARTS + 1:
                log(f"GIVING UP on {cat} [{s},{e}) at task {x + n}: {MAX_RESTARTS} relaunches made no progress")
                fails[key] += 1
            continue
        running += 1
        log(f"{cat} [{s},{e}): client from {x} died after {n} tasks -> relaunching from {x + n} on port {port}")
        launch(cat, x + n, e, port, egl)
    log(f"cycle: {finished}/{len(jobs)} chunks finished, {running} clients running (max {MAX_CLIENTS or '-'}), scan took {time.time() - t0:.0f}s")
    if finished == len(jobs):
        log("all chunks finished")
        break
    if servers and not any(s.poll() is None for s in servers):
        log("all model servers died -> stopping (results so far are kept; rerun to resume is not automatic)")
        break
for s in servers:
    if s.poll() is None:
        s.terminate()
        log(f"stopped model server pid {s.pid}")
