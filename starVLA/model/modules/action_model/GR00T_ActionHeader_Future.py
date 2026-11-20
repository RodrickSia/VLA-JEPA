# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class _AdaLN(nn.Module):
    """LayerNorm whose scale/shift come from the diffusion-timestep embedding (same idea as the DiT's ada_norm)."""

    def __init__(self, dim, temb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(temb_dim, 2 * dim))

    def forward(self, x, temb):
        scale, shift = self.proj(temb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class WorldModelCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block. Queries = action-side tokens, keys/values = world-model latents.

    attention -> (optional) feed-forward, each residual. With gate="tanh_zero" every sub-layer output is scaled by
    tanh(g) with g = 0 at init, so a block added to a pretrained head is an exact no-op at step 0 and is opened up
    by fine-tuning (Flamingo-style); it does not limit the block's capacity.
    """

    def __init__(self, dim, kdim, num_heads, dropout=0.0, norm_type="ada_norm", use_ffn=True, ffn_ratio=4,
                 gate="tanh_zero", temb_dim=None):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"world_model_attention.num_heads ({num_heads}) must divide the query dim ({dim})")
        if norm_type == "ada_norm" and temb_dim is None:
            raise ValueError("ada_norm needs the timestep-embedding dim")

        def make_norm():
            if norm_type == "ada_norm":
                return _AdaLN(dim, temb_dim)
            if norm_type == "layer_norm":
                return nn.LayerNorm(dim)
            if norm_type == "none":
                return nn.Identity()
            raise ValueError(f"unknown world_model_attention.norm_type: {norm_type}")

        self.norm_q = make_norm()
        self.norm_kv = nn.LayerNorm(kdim) if norm_type != "none" else nn.Identity()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, kdim=kdim, vdim=kdim, batch_first=True)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ff = make_norm()
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_ratio * dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                nn.Linear(ffn_ratio * dim, dim),
            )
        if gate == "tanh_zero":
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1)) if use_ffn else None
        elif gate == "none":
            self.gate_attn = self.gate_ffn = None
        else:
            raise ValueError(f"unknown world_model_attention.gate: {gate}")

    @staticmethod
    def _norm(norm, x, temb):
        return norm(x, temb) if isinstance(norm, _AdaLN) else norm(x)

    @staticmethod
    def _gated(g, y):
        return y if g is None else torch.tanh(g) * y

    def forward(self, x, ctx, temb=None):
        q = self._norm(self.norm_q, x, temb)
        kv = self.norm_kv(ctx)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + self._gated(self.gate_attn, a)
        if self.use_ffn:
            x = x + self._gated(self.gate_ffn, self.ffn(self._norm(self.norm_ff, x, temb)))
        return x


def _cfg(c, key, default=None):
    return c.get(key, default) if hasattr(c, "get") else getattr(c, key, default)


class FlowmatchingActionHeadFuture(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size # 是不要和 Q对齐？
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        # World-model cross-attention, configured by framework.action_model.world_model_attention.
        #   Q   = action-side tokens: sa_embs (input_embedding_dim) before the DiT, or the DiT output (output_dim) after it
        #   K/V = world-model predicted latents; kdim is filled in by VLA_JEPA.__init__ from the V-JEPA encoder
        wm_cfg = _cfg(config, "world_model_attention", None)
        if wm_cfg is None:
            raise ValueError("head_type groot_future needs a framework.action_model.world_model_attention block in the config")
        wm_type = _cfg(wm_cfg, "type", "mha")
        if wm_type != "mha":
            raise ValueError(f"world_model_attention.type '{wm_type}' not implemented (only 'mha')")
        self.wm_position = _cfg(wm_cfg, "position", "pre_dit")
        if self.wm_position == "pre_dit":
            q_dim = self.input_embedding_dim
        elif self.wm_position == "post_dit":
            q_dim = diffusion_model_cfg["output_dim"]
        else:
            raise ValueError(f"world_model_attention.position must be pre_dit or post_dit, got {self.wm_position}")
        kdim = _cfg(wm_cfg, "kdim", None)
        if kdim is None:
            raise ValueError("world_model_attention.kdim is not set (VLA_JEPA.__init__ fills it from the V-JEPA encoder)")
        self.wm_norm_type = _cfg(wm_cfg, "norm_type", "ada_norm")
        self.wm_pool_size = _cfg(wm_cfg, "pool_size", None)
        self.wm_detach = bool(_cfg(wm_cfg, "detach_wm", True))
        self.world_model_attention = nn.ModuleList([
            WorldModelCrossAttentionBlock(
                dim=q_dim, kdim=int(kdim), num_heads=int(_cfg(wm_cfg, "num_heads", 12)),
                dropout=float(_cfg(wm_cfg, "dropout", 0.0)), norm_type=self.wm_norm_type,
                use_ffn=bool(_cfg(wm_cfg, "use_ffn", True)), ffn_ratio=int(_cfg(wm_cfg, "ffn_ratio", 4)),
                gate=_cfg(wm_cfg, "gate", "tanh_zero"), temb_dim=self.model.inner_dim,
            )
            for _ in range(int(_cfg(wm_cfg, "num_layers", 4)))
        ])
        self.uses_world_model = True
        
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def _prepare_wm(self, wm_states, dtype):
        """World-model latents [B, T, H, W, C] (or already-flat [B, N, C]) -> K/V tokens [B, N, C].
        With pool_size p, each step's H x W patch grid is average-pooled to p x p (N = T * p * p)."""
        if wm_states is None:  # e.g. inference before the world model is wired in
            return None
        if self.wm_detach:
            wm_states = wm_states.detach()
        if wm_states.dim() == 5:
            B, T, H, W, C = wm_states.shape
            if self.wm_pool_size:
                x = wm_states.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
                x = F.adaptive_avg_pool2d(x.float(), int(self.wm_pool_size))
                wm_states = x.reshape(B, T, C, -1).permute(0, 1, 3, 2).reshape(B, -1, C)
            else:
                wm_states = wm_states.reshape(B, T * H * W, C)
        return wm_states.to(dtype)

    def _apply_wm(self, x, wm, timestep):
        if wm is None:
            return x
        temb = self.model.timestep_encoder(timestep).to(x.dtype) if self.wm_norm_type == "ada_norm" else None
        for blk in self.world_model_attention:
            x = blk(x, wm, temb)
        return x

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, wm_states: torch.Tensor = None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        # World-model cross-attention (pre_dit: on sa_embs before the DiT)
        wm = self._prepare_wm(wm_states, sa_embs.dtype)
        if self.wm_position == "pre_dit":
            sa_embs = self._apply_wm(sa_embs, wm, t_discretized)

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        if self.wm_position == "post_dit":
            model_output = self._apply_wm(model_output, wm, t_discretized)
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, wm_states: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        state_features = self.state_encoder(state) if state is not None else None
        wm = self._prepare_wm(wm_states, vl_embs.dtype)  # constant across denoising steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
                if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
            if self.wm_position == "pre_dit":
                sa_embs = self._apply_wm(sa_embs, wm, timesteps_tensor)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            if self.wm_position == "post_dit":
                model_output = self._apply_wm(model_output, wm, timesteps_tensor)
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHeadFuture(
        full_config=config
    )


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass