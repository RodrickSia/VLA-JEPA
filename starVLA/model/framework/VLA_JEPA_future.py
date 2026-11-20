# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor, AutoConfig
from omegaconf import open_dict

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("VLA_JEPA_future")
class VLA_JEPA_Future(baseframework):
    """
    VLA-JEPA + world-model-conditioned action head (framework.name: VLA_JEPA_future).
    The action head cross-attends to future latents rolled out by the JEPA world model from the current frame.

    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        # Adding special learnable tokens ids
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        wm_attn_cfg = self.config.framework.action_model.get("world_model_attention", None)
        if wm_attn_cfg is not None:  # K/V dim of the head's world-model attention = vj_predictor output dim
            vj_cfg = AutoConfig.from_pretrained(self.config.framework.vj2_model.base_encoder)
            with open_dict(wm_attn_cfg):
                wm_attn_cfg.kdim = vj_cfg.hidden_size * 2  # 2 camera views, same as vj_predictor embed_dim

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def _rollout_future_states(self, current_frames, action_tokens):
        """Future world-model latents for the action head, computed from the CURRENT frame only.

        Used identically in training and inference, so the head never sees information that is unavailable at test
        time (the training clip contains the next frames; they are only used for wm_loss).
          current_frames: list of B*V arrays [3, H, W] (current frame of each view, sample-major: b*V + v)
          action_tokens:  [B, (T-1)*K, D] latent action tokens from Qwen
        The current frame is repeated tubelet_size times (= one V-JEPA step 0), then the world model is rolled out
        autoregressively: step k+1 is predicted from steps 0..k and latent-action groups 0..k.
        Returns [B, T-1, grid_h, grid_w, V*C].
        """
        tub = self.vj_encoder.config.tubelet_size
        B = action_tokens.shape[0]
        V = len(current_frames) // B
        clips = [np.repeat(np.asarray(f)[None], tub, axis=0) for f in current_frames]  # [tub, 3, H, W] each
        pix = torch.cat([self.vj_processor(videos=c, return_tensors="pt")["pixel_values_videos"] for c in clips])
        with torch.no_grad():
            emb = self.vj_encoder.get_vision_features(pixel_values_videos=pix.to(self.vj_encoder.device))  # [B*V, HW, C]
        HW, C = emb.shape[1:]
        x = emb.reshape(B, V, HW, C).permute(0, 2, 1, 3).reshape(B, HW, V * C)  # same view layout as in forward
        K = self.config.framework.vj2_model.num_action_tokens_per_timestep
        n_future = self.config.framework.vj2_model.num_frames // tub - 1
        preds = []
        for k in range(1, n_future + 1):
            out = self.vj_predictor(x, action_tokens[:, : k * K])  # [B, k*HW, V*C], frame-causal
            nxt = out[:, -HW:]                                     # prediction of step k
            preds.append(nxt)
            x = torch.cat([x, nxt], dim=1)
        future = torch.stack(preds, dim=1)                         # [B, T-1, HW, V*C]
        return future.reshape(B, n_future, self.vj_predictor.grid_height, self.vj_predictor.grid_width, -1)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                input_videos.append(self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                num_temporal_steps = T // self.vj_encoder.config.tubelet_size
                tokens_per_clip, embed_dim = video_embeddings.shape[1:]
                tokens_per_step = tokens_per_clip // num_temporal_steps
                video_embeddings = video_embeddings.reshape(B, V, num_temporal_steps, tokens_per_step, embed_dim)
                video_embeddings = video_embeddings.permute(0, 2, 3, 1, 4).contiguous().reshape(
                    B, num_temporal_steps * tokens_per_step, V * embed_dim
                )
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )
            # Future latents for the action head: rolled out from the CURRENT frame only (no future frames), so
            # training gives the head exactly what inference can compute. wm_loss stays teacher-forced on the clip.
            wm_for_head = None
            if getattr(self.action_model, "uses_world_model", False):
                keep_grad = torch.is_grad_enabled() and not getattr(self.action_model, "wm_detach", True)
                with torch.set_grad_enabled(keep_grad):
                    wm_for_head = self._rollout_future_states(
                        [batch_videos[i][0] for i in range(B * V)], action_tokens)  # frame 0 of each view's clip

            # TODO(rollout loss, not applied yet): train the world model the way the action head uses it.
            #   Now: teacher_forcing_wm_loss predicts steps 1..3 from the REAL steps 0..2 (one parallel pass), but the head
            #   reads the ROLLOUT (_rollout_future_states: from the current frame only, each step fed its own prediction).
            #   The WM is never trained in rollout mode, so its errors compound at test time and are never penalised.
            #   Plan: rollout_wm_loss = L1([f1, f2, f3], [z1, z2, z3]) with f = wm_for_head (rollout from frame t) and
            #   z = gt_states (V-JEPA latents of the real frames t+2..t+7); compute the rollout WITH grad for this loss
            #   (the head can still read a detached copy if detach_wm), then use rollout_wm_loss instead of / next to
            #   teacher_forcing_wm_loss. Same cost: the rollout is already computed for the head.
            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )
        
        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):

            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            wm_kwargs = {}
            if wm_for_head is not None:
                wm_kwargs["wm_states"] = wm_for_head.repeat(repeated_diffusion_steps, 1, 1, 1, 1)
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated, **wm_kwargs)  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        uses_wm = getattr(self.action_model, "uses_world_model", False)
        if uses_wm:  # current frame of each view (sample-major), as [3, H, W] arrays for V-JEPA
            current_frames = [np.asarray(img.convert("RGB") if hasattr(img, "convert") else img).transpose(2, 0, 1)
                              for views in batch_images for img in views]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)
            wm_states = None
            if uses_wm:  # same rollout as in training: current frame -> predicted future latents
                action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)
                wm_states = self._rollout_future_states(current_frames, action_tokens)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # TODO(prefix-conditioned async chunking, not applied yet): freeze the prefix of the action chunk.
            #   In async execution with delay d, the robot runs the d leftover actions of the previous chunk while this
            #   chunk is computed, so rows 0..d-1 of the new chunk are already decided. Plan:
            #   - take the d leftover actions (NORMALIZED) as `prefix_actions` [B, d, action_dim], sent by the client
            #     (M1Inference.step) and passed in through **kwargs of this function;
            #   - before any denoising, write them into rows 0..d-1 of the initial noise and keep them frozen at every
            #     Euler step (clamp after each update), with per-token flow time = clean (t = 1 in this head) for those rows;
            #   - denoise only rows d..H-1, and reuse the frozen prefix rows as-is (they are the actions actually executed);
            #   - training must match: sample d ~ U{0..d_max}, give GT rows 0..d-1 as the clean prefix, loss only on rows d..
            #   Needs: per-token timesteps in the head's ActionEncoder, a prefix-clamped predict_action in
            #   GR00T_ActionHeader_Future.py, and the prefix sent by the client in async mode.
            if uses_wm:
                pred_actions = self.action_model.predict_action(embodied_action_tokens, state, wm_states=wm_states)
            else:
                pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
