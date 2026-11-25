"""Per-module inference latency of VLA_JEPA_future (one predict_action call, batch 1, 2 camera views).

Times the same steps as VLA_JEPA_future.predict_action, separately:
  1. VLA          : Qwen3-VL input building + forward -> embodied tokens [B,32,H] and latent-action tokens [B,24,H]
  2. World model  : (a) V-JEPA preprocessing on CPU, (b) V-JEPA2 encoder on the current frame,
                    (c) predictor rollout over the future steps, conditioned on the latent-action tokens from step 1
  3. Action head  : DiT flow-matching denoising (with the world-model cross-attention blocks)
Usage: CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python scripts/eval/bench_module_latency.py <ckpt.pt> <frame.mp4> [iters]
"""
import sys, time, statistics as st
import numpy as np, torch, imageio, cv2
sys.path.insert(0, "/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA")
from starVLA.model.framework.base_framework import baseframework
from starVLA.training.trainer_utils.trainer_tools import resize_images  # noqa: F401  (same helper predict_action uses)

ckpt, video = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
dev = torch.device("cuda:0")

t0 = time.time()
vla = baseframework.from_pretrained(ckpt).to(torch.bfloat16).to(dev).eval()   # same as server_policy.py --use_bf16
print(f"model loaded in {time.time() - t0:.0f} s; uses_world_model={getattr(vla.action_model, 'uses_world_model', False)}")

frame = cv2.resize(np.asarray(imageio.mimread(video, memtest=False)[20])[..., :3], (224, 224)).astype(np.uint8)
batch_images = [[frame, frame]]                       # B=1, V=2 (agent view + wrist; same frame is fine for timing)
instructions = ["put the orange juice in the basket"]
state = [np.array([[-0.05, 0.0, 1.05, 3.1, 0.0, 0.0, 0.039, -0.039]], dtype=np.float32)]

def sync():
    torch.cuda.synchronize(dev)

@torch.inference_mode()
def run_once():
    t = {}
    # ---- 1. VLA
    sync(); a = time.perf_counter()
    current_frames = [np.asarray(img).transpose(2, 0, 1) for views in batch_images for img in views]
    qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, instructions=instructions,
        prompt_replace_dict={"{actions}": vla.replace_prompt, "{e_actions}": vla.embodied_replace_prompt})
    ids = qwen_inputs["input_ids"]
    emb_idx = torch.isin(ids, torch.tensor([vla.embodied_action_token_id], device=ids.device)).nonzero(as_tuple=True)
    act_idx = torch.isin(ids, torch.tensor(vla.action_token_ids, device=ids.device)).nonzero(as_tuple=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        B, _, H = h.shape
        embodied = h[emb_idx[0], emb_idx[1], :].view(B, -1, H)
        action_tokens = h[act_idx[0], act_idx[1], :].view(B, -1, H)
    sync(); t["1 VLA (Qwen3-VL)"] = time.perf_counter() - a

    # ---- 2. world model, split into its 3 parts (same code as _rollout_future_states)
    wm = vla
    tub = wm.vj_encoder.config.tubelet_size
    a = time.perf_counter()
    clips = [np.repeat(np.asarray(f)[None], tub, axis=0) for f in current_frames]
    pix = torch.cat([wm.vj_processor(videos=c, return_tensors="pt")["pixel_values_videos"] for c in clips])
    t["2a WM: V-JEPA preprocessing (CPU)"] = time.perf_counter() - a
    sync(); a = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = wm.vj_encoder.get_vision_features(pixel_values_videos=pix.to(device=dev, dtype=torch.bfloat16))
    sync(); t["2b WM: V-JEPA2 encoder"] = time.perf_counter() - a
    a = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        HW, C = emb.shape[1:]
        V = len(current_frames) // B
        x = emb.reshape(B, V, HW, C).permute(0, 2, 1, 3).reshape(B, HW, V * C)
        K = wm.config.framework.vj2_model.num_action_tokens_per_timestep
        n_future = wm.config.framework.vj2_model.num_frames // tub - 1
        preds = []
        for k in range(1, n_future + 1):
            o = wm.vj_predictor(x, action_tokens[:, : k * K])
            nxt = o[:, -HW:]
            preds.append(nxt)
            x = torch.cat([x, nxt], dim=1)
        wm_states = torch.stack(preds, 1).reshape(B, n_future, wm.vj_predictor.grid_height, wm.vj_predictor.grid_width, -1)
    sync(); t["2c WM: predictor rollout"] = time.perf_counter() - a

    # ---- 3. action head (DiT with world-model cross-attention)
    a = time.perf_counter()
    st_t = torch.from_numpy(np.array(state)).to(h.device, dtype=h.dtype)
    with torch.autocast("cuda", dtype=torch.float32):
        acts = vla.action_model.predict_action(embodied, st_t, wm_states=wm_states)
    sync(); t["3 Action head (DiT + WM cross-attn)"] = time.perf_counter() - a
    return t, tuple(wm_states.shape), tuple(acts.shape), n_future

for _ in range(3):
    run_once()                                        # warm-up (cuDNN autotune, allocator)
runs = [run_once() for _ in range(iters)]
res = {k: [r[0][k] * 1000 for r in runs] for k in runs[0][0]}
print(f"\nGPU: {torch.cuda.get_device_name(dev)}  | iters={iters} | wm_states {runs[0][1]} | actions {runs[0][2]} | rollout steps {runs[0][3]}")
total = sum(st.median(v) for v in res.values())
for k, v in res.items():
    print(f"  {k:38s} median {st.median(v):7.1f} ms   p90 {np.percentile(v, 90):7.1f} ms   ({100 * st.median(v) / total:4.1f}%)")
wm_tot = sum(st.median(v) for k, v in res.items() if k.startswith("2"))
print(f"  {'WORLD MODEL total (2a+2b+2c)':38s} median {wm_tot:7.1f} ms   ({100 * wm_tot / total:4.1f}%)")
print(f"  {'END-TO-END (sum of medians)':38s} median {total:7.1f} ms")
