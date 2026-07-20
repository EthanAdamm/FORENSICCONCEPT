import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip, set_lora_config


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
    "PE-Lang-L14-448": 1024,
    "PE-Lang-G14-448": 1536,
    "PE-Core-G14-448": 1536,
}


def _load_codebook(path: str) -> torch.Tensor:
    """Load codebook from .npy or .pt/.pth"""
    if path is None:
        raise ValueError("codebook_path is None")
    if path.endswith(".npy"):
        arr = np.load(path)
        cb = torch.from_numpy(arr).float()
    else:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["codebook", "centroids", "C"]:
                if k in obj:
                    obj = obj[k]
                    break
        cb = obj.float() if torch.is_tensor(obj) else torch.tensor(obj, dtype=torch.float32)
    if cb.ndim != 2:
        raise ValueError(f"codebook must be 2D, got {tuple(cb.shape)} from {path}")
    return cb


class CodebookTopKHead(nn.Module):
    """
    Top-k patch pooling guided by codebook similarity.
    tokens:   [B,N,dim_x]
    codebook: [K,dim_c] (assumed L2 normalized if clip_codebook_l2=True at load)
    """

    def __init__(
        self,
        dim_x: int,
        dim_c: int,
        num_concepts: int,
        tau: float = 0.1,
        tau_w: float = 0.1,
        topk_m: int = 20,
        top_r: int = 8,
        weight_mode: str = "score",
        l2_codebook: bool = True,
        freeze_codebook: bool = True,
        mlp_hidden: int = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_c = dim_c
        self.num_concepts = num_concepts
        self.tau = tau
        self.tau_w = tau_w
        self.topk_m = topk_m
        self.top_r = top_r
        self.weight_mode = (weight_mode or "score").lower()
        self.l2_codebook = l2_codebook
        self.freeze_codebook = freeze_codebook

        self.w_q = nn.Linear(dim_x, dim_c, bias=False)
        self.register_buffer("codebook", torch.empty(num_concepts, dim_c), persistent=True)

        hidden = mlp_hidden if mlp_hidden is not None else dim_x
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_x),
            nn.Linear(dim_x, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @torch.no_grad()
    def set_codebook(self, codebook: torch.Tensor):
        if codebook.shape != self.codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: got {tuple(codebook.shape)} expected {tuple(self.codebook.shape)}"
            )
        cb = codebook.float()
        if self.l2_codebook:
            cb = F.normalize(cb, dim=-1)
        self.codebook.copy_(cb)

    def forward(self, tokens: torch.Tensor, return_feature: bool = False):
        """
        returns:
          logits_cb: [B,1]
          score:     [B,N]
          idx:       [B,m]
        """
        bsz, num_patches, dim_x = tokens.shape
        if dim_x != self.dim_x:
            raise ValueError(f"tokens dim mismatch: {dim_x} vs {self.dim_x}")

        q_raw = self.w_q(tokens)
        q = F.normalize(q_raw, dim=-1)
        k = self.codebook.detach() if self.freeze_codebook else self.codebook

        s = torch.matmul(q, k.t()) / max(self.tau, 1e-8)
        r = int(self.top_r) if self.top_r else 1
        r = min(r, s.shape[-1])
        top_vals, top_ids = s.topk(k=r, dim=-1)
        score = top_vals.mean(dim=-1)

        m = int(self.topk_m) if self.topk_m else num_patches
        m = min(m, num_patches)
        score_sel, idx = torch.topk(score, k=m, dim=1)
        top_ids_sel = top_ids.gather(1, idx.unsqueeze(-1).expand(-1, -1, top_ids.shape[-1]))
        concept_id = s.argmax(dim=-1)
        topk_concept = concept_id.gather(1, idx)

        x_sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, dim_x))
        if self.weight_mode == "score":
            patch_weight = score_sel
        elif self.weight_mode == "concept":
            s_sel = s.gather(1, idx.unsqueeze(-1).expand(-1, -1, s.shape[-1]))
            concept_prob = F.softmax(s_sel, dim=-1)
            patch_weight = concept_prob.max(dim=-1).values
        else:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

        w = F.softmax(patch_weight / max(self.tau_w, 1e-8), dim=1)
        g = (w.unsqueeze(-1) * x_sel).sum(dim=1)

        logits_cb = self.mlp(g)
        if return_feature:
            return logits_cb, score, idx, top_ids_sel, g
        return logits_cb, score, idx, top_ids_sel


class CLIPModel(nn.Module):
    """CLIP wrapper: CLS head + codebook top-k head."""

    def __init__(self, name, num_classes=1, opt=None):
        super().__init__()

        clip_rank = getattr(opt, "clip_lora_rank", 4) if opt is not None else 4
        clip_alpha = getattr(opt, "clip_lora_alpha", 8) if opt is not None else 8
        set_lora_config(rank=clip_rank, alpha=clip_alpha)

        load_init = True if opt is None else bool(getattr(opt, "clip_load_init_weights", True))
        os.environ["CLIP_USE_INIT"] = "1" if load_init else "0"

        weight_path = getattr(opt, "clip_model_path", None)
        load_target = weight_path if weight_path else name
        self.model, self.preprocess = clip.load(load_target, device="cpu")

        # If explicit checkpoint path is provided, trust loaded model metadata first.
        in_features = None
        if weight_path:
            in_features = getattr(self.model.visual, "output_dim", None)
        if in_features is None:
            in_features = CHANNELS.get(name)
        if in_features is None:
            in_features = getattr(self.model.visual, "output_dim", None)
        if in_features is None:
            with torch.no_grad():
                res = getattr(self.model.visual, "input_resolution", 224)
                dummy = torch.zeros(1, 3, res, res)
                in_features = self.model.encode_image(dummy).shape[-1]

        self.dim_x = int(in_features)
        self.opt = opt
        self.use_main_head = bool(getattr(opt, "clip_use_main_head", True))
        if not self.use_main_head:
            raise ValueError("clip_use_main_head must be True for this CLIP head.")

        num_patches = getattr(opt, "clip_num_patches", None) if opt is not None else None
        if not num_patches:
            patch_size = getattr(self.model.visual, "patch_size", None)
            crop_size = getattr(opt, "cropSize", None) if opt is not None else None
            if patch_size and crop_size:
                grid = int(crop_size) // int(patch_size)
                num_patches = grid * grid
        if not num_patches:
            raise ValueError("clip_num_patches is required (or provide cropSize to infer it).")
        self.num_patches = int(num_patches)

        self.codebook_path = getattr(opt, "clip_codebook_path", None) if opt is not None else None
        self.use_codebook_inject = bool(getattr(opt, "clip_use_codebook_inject", True)) if opt is not None else True
        self.cb_detach_tokens = bool(getattr(opt, "clip_cb_detach_tokens", False)) if opt is not None else False

        self.dim_c = int(getattr(opt, "clip_codebook_dim", 1280)) if opt is not None else 1280
        self.num_concepts = int(getattr(opt, "clip_num_concepts", 200)) if opt is not None else 200

        tau = float(getattr(opt, "clip_tau", 0.1)) if opt is not None else 0.1

        l2_codebook = bool(getattr(opt, "clip_codebook_l2", True)) if opt is not None else True
        freeze_codebook = bool(getattr(opt, "clip_freeze_codebook", True)) if opt is not None else True

        self.codebook_head = None
        if self.use_codebook_inject:
            if not self.codebook_path:
                raise ValueError("clip_use_codebook_inject=True but clip_codebook_path is None")

            topk_m = int(getattr(opt, "clip_cb_topk", 20)) if opt is not None else 20
            top_r = int(getattr(opt, "clip_cb_topr", 8)) if opt is not None else 8
            tau_w = float(getattr(opt, "clip_cb_tau_w", 0.1)) if opt is not None else 0.1
            weight_mode = getattr(opt, "clip_cb_weight_mode", "score") if opt is not None else "score"
            mlp_hidden = getattr(opt, "clip_cb_mlp_hidden", None) if opt is not None else None

            self.codebook_head = CodebookTopKHead(
                dim_x=self.dim_x,
                dim_c=self.dim_c,
                num_concepts=self.num_concepts,
                tau=tau,
                tau_w=tau_w,
                topk_m=topk_m,
                top_r=top_r,
                weight_mode=weight_mode,
                l2_codebook=l2_codebook,
                freeze_codebook=freeze_codebook,
                mlp_hidden=mlp_hidden,
            )

            cb = _load_codebook(self.codebook_path)
            if cb.shape[0] != self.num_concepts or cb.shape[1] != self.dim_c:
                raise ValueError(
                    f"Loaded codebook shape {tuple(cb.shape)} != ({self.num_concepts},{self.dim_c})"
                )
            self.codebook_head.set_codebook(cb)

        self.fc = nn.Linear(self.dim_x, num_classes)

    def forward(self, x, return_feature=False, return_debug=False):
        """
        returns:
          logits: [B,num_classes]
          tokens: original patch tokens (CLS removed)
          logits_cb: [B,1] (if enabled)
        """
        cls, tokens = self.model.encode_image_with_tokens(x)
        tokens = tokens[:, 1:, :]
        if tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"num_patches mismatch: tokens has {tokens.shape[1]} patches, expected {self.num_patches}"
            )

        logits_cls = self.fc(cls)
        if self.use_codebook_inject:
            if self.codebook_head is None:
                raise RuntimeError("Codebook head is not initialized.")
            cb_tokens = tokens.detach() if self.cb_detach_tokens else tokens
            logits_cb, score, idx, topk_concept = self.codebook_head(cb_tokens)
        else:
            logits_cb, score, idx, topk_concept = None, None, None, None

        if return_feature:
            return cls

        logits = logits_cls

        out = {
            "logits": logits,
            "cb_only": logits_cb,
            "tokens": tokens,
        }
        if return_debug and score is not None:
            out.update({
                "codebook_score": score,
                "codebook_topk_idx": idx,
                "codebook_topk_concept": topk_concept,
            })
        return out
