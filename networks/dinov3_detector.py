import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import re
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, safe_open

try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when tqdm missing
    _TQDM_AVAILABLE = False

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(kwargs.get("total", 0))

_DINOV3_REPOSITORY_URL = "https://github.com/facebookresearch/dinov3"


@dataclass
class Dinov3Config:
    variant: str = "vitl16"
    backend: str = "official"  # official | legacy-compatible external source
    repo_path: Optional[str] = None
    weights_path: Optional[str] = None
    freeze_backbone: bool = True
    trainable_blocks: int = 0
    max_blocks: Optional[int] = None
    head_hidden_dim: Optional[int] = None
    head_dropout: float = 0.1
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: Optional[List[str]] = None
    lora_target_last_n: Optional[int] = None
    use_concept_head: bool = False
    concept_matrix_path: Optional[str] = None
    concept_mapping_trainable: bool = False
    concept_mapping_bias: bool = False
    concept_head_hidden_dim: Optional[int] = None
    concept_head_dropout: Optional[float] = None
    concept_sparsity_ratio: float = 0.0
    concept_loss_weight: float = 1.0
    concept_eval_sparsity: Optional[List[float]] = None
    train_lora: bool = True
    disable_rope_train_jitter: bool = False
    gradient_checkpointing: bool = False


def _normalize_backend(backend: str) -> str:
    value = str(backend or "").strip().lower()
    if value in {"official", "legacy"}:
        return value
    raise ValueError(f"Unsupported DINOv3 backend: {backend}. Expected one of: official, legacy.")


def _candidate_dinov3_roots(config: Dinov3Config) -> List[Path]:
    _normalize_backend(config.backend)
    if not config.repo_path:
        return []
    return [Path(config.repo_path).expanduser().resolve()]


def _ensure_dinov3_importable(config: Dinov3Config) -> Optional[Path]:
    for root in _candidate_dinov3_roots(config):
        if not (root / "dinov3").is_dir():
            continue
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        return root

    try:
        import dinov3 as dinov3_pkg  # type: ignore

        return Path(dinov3_pkg.__file__).resolve().parents[1]
    except Exception as exc:  # noqa: BLE001
        configured = _candidate_dinov3_roots(config)
        checked = "\n".join(f"  - {p}" for p in configured) if configured else "  - no repository path configured"
        raise ImportError(
            "Could not import the official DINOv3 package.\n"
            f"Clone {_DINOV3_REPOSITORY_URL} and set training.dinov3_repo_path, "
            "or install DINOv3 in the active Python environment.\n"
            f"Checked:\n{checked}"
        ) from exc


def _resolve_weights_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct_file = os.path.join(path, "model.safetensors")
        if os.path.isfile(direct_file):
            return direct_file
        index_file = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_file):
            return index_file
    raise FileNotFoundError(f"Could not resolve DINOv3 weights from: {path}")


def _is_pth_like(path: str) -> bool:
    lower = str(path).lower()
    return lower.endswith(".pth") or lower.endswith(".pt")


def _is_url(path: str) -> bool:
    try:
        parsed = urlparse(str(path))
    except Exception:
        return False
    return parsed.scheme in {"http", "https", "file"}


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".index.json"):
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
        directory = os.path.dirname(path)
        state_dict: Dict[str, torch.Tensor] = {}
        total_tensors = len(index["weight_map"])
        unique_shards = sorted(set(index["weight_map"].values()))
        progress = tqdm(total=total_tensors, desc="Loading DINOv3 weights", unit="tensor") if _TQDM_AVAILABLE else None
        for shard_file in unique_shards:
            shard_path = os.path.join(directory, shard_file)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                    if progress is not None:
                        progress.update(1)
        if progress is not None:
            progress.close()
        return state_dict
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        if _TQDM_AVAILABLE:
            progress = tqdm(keys, desc="Loading DINOv3 weights", unit="tensor")
            state_dict = {key: f.get_tensor(key) for key in progress}
            progress.close()
            return state_dict
        return {key: f.get_tensor(key) for key in keys}


def _dinov3_constructors(config: Dinov3Config):
    source_root = _ensure_dinov3_importable(config)
    from dinov3.hub import backbones

    backend = _normalize_backend(config.backend)
    if source_root is not None:
        print(f"[DINOv3] backend={backend}, source={source_root}")

    return {
        "vits16": backbones.dinov3_vits16,
        "vits16plus": backbones.dinov3_vits16plus,
        "vitb16": backbones.dinov3_vitb16,
        "vitl16": backbones.dinov3_vitl16,
        "vith16plus": backbones.dinov3_vith16plus,
        "vit7b16": backbones.dinov3_vit7b16,
        "convnext_tiny": backbones.dinov3_convnext_tiny,
        "convnext_small": backbones.dinov3_convnext_small,
        "convnext_base": backbones.dinov3_convnext_base,
        "convnext_large": backbones.dinov3_convnext_large,
    }


def _sparse_shrink(
    x: torch.Tensor, ratio: float, return_mask: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if ratio <= 0.0:
        if return_mask:
            mask = torch.ones_like(x, dtype=torch.bool)
            return x, mask
        return x

    width = x.shape[-1]
    if width == 0:
        if return_mask:
            mask = torch.zeros_like(x, dtype=torch.bool)
            return x, mask
        return x

    keep = int(width * (1 - ratio))
    keep = max(1, min(width, keep))
    if keep >= width:
        if return_mask:
            mask = torch.ones_like(x, dtype=torch.bool)
            return x, mask
        return x

    # kthvalue returns the k-th smallest element, so convert to the keep-th largest
    threshold_index = width - keep + 1
    threshold, _ = x.abs().kthvalue(threshold_index, dim=-1, keepdim=True)
    mask = x.abs() >= threshold
    shrunk = x * mask
    if return_mask:
        return shrunk, mask
    return shrunk


def _map_hf_to_dinov3(backbone: nn.Module, hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    if "embeddings.cls_token" in hf_state:
        cls_token = hf_state["embeddings.cls_token"]
        mapped["cls_token"] = cls_token if cls_token.ndim == 3 else cls_token.unsqueeze(1)
    if "embeddings.mask_token" in hf_state:
        mask_token = hf_state["embeddings.mask_token"]
        if mask_token.ndim == 3:
            mask_token = mask_token.squeeze(1)
        mapped["mask_token"] = mask_token
    if "embeddings.register_tokens" in hf_state and hasattr(backbone, "n_storage_tokens") and backbone.n_storage_tokens:
        mapped["storage_tokens"] = hf_state["embeddings.register_tokens"]
    if "embeddings.patch_embeddings.weight" in hf_state:
        mapped["patch_embed.proj.weight"] = hf_state["embeddings.patch_embeddings.weight"]
    if "embeddings.patch_embeddings.bias" in hf_state:
        mapped["patch_embed.proj.bias"] = hf_state["embeddings.patch_embeddings.bias"]

    n_blocks = getattr(backbone, "n_blocks", 0)
    ref_tensor = next(iter(hf_state.values()))
    blocks = getattr(backbone, "blocks", None)
    progress = None
    iterator: Iterable[int]
    if _TQDM_AVAILABLE and n_blocks > 0:
        progress = tqdm(range(n_blocks), desc="Mapping DINOv3 blocks", unit="block")
        iterator = progress
    else:
        iterator = range(n_blocks)

    for i in iterator:
        prefix = f"layer.{i}."
        block_prefix = f"blocks.{i}."
        block = blocks[i] if blocks is not None else None
        mapped[f"{block_prefix}norm1.weight"] = hf_state[f"{prefix}norm1.weight"]
        mapped[f"{block_prefix}norm1.bias"] = hf_state[f"{prefix}norm1.bias"]
        mapped[f"{block_prefix}norm2.weight"] = hf_state[f"{prefix}norm2.weight"]
        mapped[f"{block_prefix}norm2.bias"] = hf_state[f"{prefix}norm2.bias"]

        q_w = hf_state[f"{prefix}attention.q_proj.weight"]
        k_w = hf_state[f"{prefix}attention.k_proj.weight"]
        v_w = hf_state[f"{prefix}attention.v_proj.weight"]
        mapped[f"{block_prefix}attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        attn_qkv = getattr(block, "attn", None)
        qkv_bias = getattr(attn_qkv, "qkv", None)
        has_qkv_bias = bool(qkv_bias is not None and getattr(qkv_bias, "bias", None) is not None)
        if has_qkv_bias:
            dim = q_w.shape[0]
            zeros = torch.zeros(dim, dtype=ref_tensor.dtype, device=ref_tensor.device)
            q_b = hf_state.get(f"{prefix}attention.q_proj.bias", zeros)
            k_b = hf_state.get(f"{prefix}attention.k_proj.bias", zeros)
            v_b = hf_state.get(f"{prefix}attention.v_proj.bias", zeros)
            mapped[f"{block_prefix}attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        mapped[f"{block_prefix}attn.proj.weight"] = hf_state[f"{prefix}attention.o_proj.weight"]
        mapped[f"{block_prefix}attn.proj.bias"] = hf_state[f"{prefix}attention.o_proj.bias"]

        if hasattr(block, "mlp") and hasattr(block.mlp, "fc1"):
            mapped[f"{block_prefix}mlp.fc1.weight"] = hf_state[f"{prefix}mlp.up_proj.weight"]
            mapped[f"{block_prefix}mlp.fc1.bias"] = hf_state[f"{prefix}mlp.up_proj.bias"]
            mapped[f"{block_prefix}mlp.fc2.weight"] = hf_state[f"{prefix}mlp.down_proj.weight"]
            mapped[f"{block_prefix}mlp.fc2.bias"] = hf_state[f"{prefix}mlp.down_proj.bias"]
        elif hasattr(block, "mlp") and all(hasattr(block.mlp, attr) for attr in ("w1", "w2", "w3")):
            mapped[f"{block_prefix}mlp.w1.weight"] = hf_state[f"{prefix}mlp.gate_proj.weight"]
            mapped[f"{block_prefix}mlp.w1.bias"] = hf_state[f"{prefix}mlp.gate_proj.bias"]
            mapped[f"{block_prefix}mlp.w2.weight"] = hf_state[f"{prefix}mlp.up_proj.weight"]
            mapped[f"{block_prefix}mlp.w2.bias"] = hf_state[f"{prefix}mlp.up_proj.bias"]
            mapped[f"{block_prefix}mlp.w3.weight"] = hf_state[f"{prefix}mlp.down_proj.weight"]
            mapped[f"{block_prefix}mlp.w3.bias"] = hf_state[f"{prefix}mlp.down_proj.bias"]
        else:
            raise RuntimeError(
                f"Unsupported MLP structure for block {i}: {type(getattr(block, 'mlp', None))}"
            )

        if f"{prefix}layer_scale1.lambda1" in hf_state:
            mapped[f"{block_prefix}ls1.gamma"] = hf_state[f"{prefix}layer_scale1.lambda1"]
        if f"{prefix}layer_scale2.lambda1" in hf_state:
            mapped[f"{block_prefix}ls2.gamma"] = hf_state[f"{prefix}layer_scale2.lambda1"]

    if progress is not None:
        progress.close()

    if "norm.weight" in hf_state:
        mapped["norm.weight"] = hf_state["norm.weight"]
    if "norm.bias" in hf_state:
        mapped["norm.bias"] = hf_state["norm.bias"]

    return mapped


def _load_backbone(config: Dinov3Config) -> nn.Module:
    constructors = _dinov3_constructors(config)
    if config.variant not in constructors:
        raise ValueError(f"Unsupported DINOv3 variant: {config.variant}")
    constructor = constructors[config.variant]

    raw_weights = config.weights_path
    resolved_weights: Optional[str] = None
    if raw_weights:
        try:
            resolved_weights = _resolve_weights_path(raw_weights)
        except FileNotFoundError:
            # Allow direct URL/file-like .pth path for official DINOv3 checkpoints.
            if _is_url(raw_weights) and _is_pth_like(raw_weights):
                resolved_weights = raw_weights
            else:
                raise

    if resolved_weights and _is_pth_like(resolved_weights):
        return constructor(pretrained=True, weights=resolved_weights)

    backbone = constructor(pretrained=False)
    if resolved_weights:
        if not (
            resolved_weights.endswith(".safetensors")
            or resolved_weights.endswith(".index.json")
        ):
            raise RuntimeError(
                f"Unsupported DINOv3 weight format: {resolved_weights}. "
                "Expected .safetensors/.index.json or .pth/.pt."
            )
        hf_state = _load_safetensors(resolved_weights)
        mapped_state = _map_hf_to_dinov3(backbone, hf_state)
        missing, unexpected = backbone.load_state_dict(mapped_state, strict=False)
        unexpected = [k for k in unexpected if not k.endswith("bias_mask")]
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading DINOv3 weights: {unexpected}")
        if missing:
            diff = [
                m
                for m in missing
                if not (
                    m.startswith("rope_embed.")
                    or m.endswith("bias_mask")
                    or m.startswith("local_cls_norm.")
                )
            ]
            if diff:
                raise RuntimeError(f"Missing keys when loading DINOv3 weights: {diff}")
    return backbone


class Dinov3Detector(nn.Module):
    def __init__(self, config: Dinov3Config, num_classes: int = 1):
        super().__init__()
        self.config = config
        self.backbone = _load_backbone(config)
        self._truncate_backbone_blocks_if_needed()
        self._disable_rope_train_jitter_if_needed()
        self.patch_size = getattr(self.backbone, "patch_size", None)
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Backbone does not expose embed_dim")

        hidden_dim = config.head_hidden_dim or embed_dim
        layers = []
        if hidden_dim != embed_dim:
            layers.extend([
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(hidden_dim, num_classes),
            ])
        else:
            layers.extend([
                nn.LayerNorm(embed_dim),
                nn.Dropout(config.head_dropout),
                nn.Linear(embed_dim, num_classes),
            ])
        self.head = nn.Sequential(*layers)

        self.use_concept_head = bool(config.use_concept_head)
        self.concept_mapping: Optional[nn.Linear] = None
        self.concept_head: Optional[nn.Module] = None
        if self.use_concept_head:
            self._init_concept_modules(embed_dim, num_classes)

        self.freeze_backbone = config.freeze_backbone
        self.trainable_blocks = max(0, config.trainable_blocks)
        self.using_lora = bool(config.use_lora and config.lora_rank > 0)
        self.train_lora = bool(config.train_lora)
        self.lora_target_modules: List[str] = []
        if self.using_lora:
            self._apply_lora_adapters()
        self._configure_trainable_params()
        self._enable_gradient_checkpointing_if_needed()

    def _truncate_backbone_blocks_if_needed(self) -> None:
        max_blocks = self.config.max_blocks
        if max_blocks is None:
            return

        try:
            max_blocks = int(max_blocks)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dinov3_max_blocks: {self.config.max_blocks}") from exc

        if max_blocks <= 0:
            raise ValueError(f"dinov3_max_blocks must be > 0, got {max_blocks}.")

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            print("[DINOv3] dinov3_max_blocks is set but backbone has no `blocks`; ignoring.")
            return

        total_blocks = len(blocks)
        if total_blocks <= 0:
            return
        keep_blocks = min(max_blocks, total_blocks)

        if keep_blocks < total_blocks:
            self.backbone.blocks = nn.ModuleList(list(blocks)[:keep_blocks])
            if hasattr(self.backbone, "n_blocks"):
                self.backbone.n_blocks = keep_blocks
            print(f"[DINOv3] Truncated transformer blocks: keeping first {keep_blocks}/{total_blocks}.")
        else:
            print(
                f"[DINOv3] dinov3_max_blocks={max_blocks} >= total_blocks={total_blocks}; "
                "using all blocks."
            )

    def _enable_gradient_checkpointing_if_needed(self) -> None:
        if not bool(self.config.gradient_checkpointing):
            return

        def _try_enable(module) -> bool:
            if module is None:
                return False
            for method_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
                method = getattr(module, method_name, None)
                if callable(method):
                    method()
                    return True
            setter = getattr(module, "set_gradient_checkpointing", None)
            if callable(setter):
                try:
                    setter(True)
                    return True
                except TypeError:
                    return False
            return False

        enabled = _try_enable(self.backbone)
        if not enabled:
            enabled = _try_enable(getattr(self.backbone, "base_model", None))
        if enabled:
            print("[FullFinetune] DINOv3 gradient checkpointing enabled.")
        else:
            print("[FullFinetune] gradient_checkpointing requested, but backbone API does not expose it.")

    def _disable_rope_train_jitter_if_needed(self) -> None:
        if not bool(getattr(self.config, "disable_rope_train_jitter", False)):
            return
        rope = getattr(self.backbone, "rope_embed", None)
        if rope is None:
            return
        changed = []
        for attr in ("shift_coords", "jitter_coords", "rescale_coords"):
            if getattr(rope, attr, None) is not None:
                setattr(rope, attr, None)
                changed.append(attr)
        if changed:
            print(f"[DINOv3] Disabled RoPE train-time coordinate jitter: {', '.join(changed)}.")

    def _init_concept_modules(self, embed_dim: int, num_classes: int) -> None:
        if not self.config.concept_matrix_path:
            raise ValueError("use_concept_head is True but concept_matrix_path is not provided.")

        matrix_path = Path(self.config.concept_matrix_path)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Concept matrix file not found: {matrix_path}")

        concept_array = np.load(str(matrix_path), allow_pickle=False)
        if concept_array.ndim != 2:
            raise ValueError(
                f"Concept matrix must be 2D, got shape {concept_array.shape} from {matrix_path}."
            )

        concept_array = concept_array.astype(np.float32, copy=False)
        concept_tensor = torch.from_numpy(concept_array)
        concept_dim, concept_embed_dim = concept_tensor.shape

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Loaded concept matrix of shape {concept_tensor.shape} "
            f"(concept_dim={concept_dim}, embed_dim={concept_embed_dim}) from {matrix_path}"
        )

        if concept_embed_dim != embed_dim:
            raise ValueError(
                "Concept matrix embedding dimension does not match backbone output: "
                f"expected {embed_dim}, got {concept_embed_dim}."
            )

        concept_mapping = nn.Linear(
            embed_dim,
            concept_dim,
            bias=bool(self.config.concept_mapping_bias),
        )
        with torch.no_grad():
            concept_mapping.weight.copy_(concept_tensor)
            if concept_mapping.bias is not None:
                concept_mapping.bias.zero_()

        if not self.config.concept_mapping_trainable:
            concept_mapping.weight.requires_grad = False
            if concept_mapping.bias is not None:
                concept_mapping.bias.requires_grad = False

        concept_hidden_dim = self.config.concept_head_hidden_dim or concept_dim
        concept_dropout = (
            self.config.concept_head_dropout
            if self.config.concept_head_dropout is not None
            else self.config.head_dropout
        )

        head_layers: List[nn.Module] = []
        if concept_hidden_dim != concept_dim:
            head_layers.extend([
                nn.LayerNorm(concept_dim),
                nn.Linear(concept_dim, concept_hidden_dim),
                nn.GELU(),
                nn.Dropout(concept_dropout),
                nn.Linear(concept_hidden_dim, num_classes),
            ])
        else:
            head_layers.extend([
                nn.LayerNorm(concept_dim),
                nn.Dropout(concept_dropout),
                nn.Linear(concept_dim, num_classes),
            ])

        self.concept_mapping = concept_mapping
        self.concept_head = nn.Sequential(*head_layers)

    def _apply_lora_adapters(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            try:
                from importlib.metadata import version as pkg_version

                peft_ver = pkg_version("peft")
            except Exception:
                peft_ver = "not-installed"
            try:
                from importlib.metadata import version as pkg_version

                transformers_ver = pkg_version("transformers")
            except Exception:
                transformers_ver = "not-installed"
            raise ImportError(
                "LoRA dependency import failed. This is usually a peft/transformers version mismatch. "
                f"Current versions: peft={peft_ver}, transformers={transformers_ver}. "
                "Reinstall the compatible ranges from requirements.txt: "
                "`python3 -m pip install -U \"transformers>=4.56,<5\" \"peft>=0.17,<0.19\"`."
            ) from exc

        target_modules = self.config.lora_target_modules
        if not target_modules:
            target_modules = self._collect_linear_module_names()
        if not target_modules:
            raise RuntimeError("Could not find any linear modules in DINOv3 backbone for LoRA.")

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.lora_target_modules = target_modules

    def _collect_linear_module_names(self) -> List[str]:
        names: List[str] = []
        last_n = self.config.lora_target_last_n
        total_blocks = getattr(self.backbone, "n_blocks", None)
        min_block_index = None
        if last_n and total_blocks:
            min_block_index = max(0, total_blocks - last_n)

        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear) and name:
                if min_block_index is not None:
                    if "blocks." in name:
                        match = re.search(r"blocks\.(\d+)", name)
                        if match and int(match.group(1)) < min_block_index:
                            continue
                    else:
                        # Skip non-block modules when restricting to last N blocks
                        continue
                names.append(name)
        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for name in names:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    def _configure_trainable_params(self):
        lora_trainable = self.train_lora if self.using_lora else False
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if self.using_lora and "lora_" in name:
                    param.requires_grad = lora_trainable
                else:
                    param.requires_grad = False
            if self.trainable_blocks > 0 and hasattr(self.backbone, "blocks"):
                for block in self.backbone.blocks[-self.trainable_blocks:]:
                    for name, param in block.named_parameters():
                        if self.using_lora and "lora_" in name:
                            param.requires_grad = lora_trainable
                        elif not self.using_lora:
                            param.requires_grad = True
                if hasattr(self.backbone, "norm"):
                    for name, param in self.backbone.norm.named_parameters():
                        if self.using_lora and "lora_" in name:
                            param.requires_grad = lora_trainable
                        elif not self.using_lora:
                            param.requires_grad = True
        else:
            for name, param in self.backbone.named_parameters():
                if self.using_lora and "lora_" in name and not lora_trainable:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Use __call__/forward path first so FSDP hooks are triggered.
        # Calling forward_features() directly can bypass composable-FSDP hooks.
        features_dict = None
        try:
            # Use inference-style forward path for stability in supervised fine-tuning.
            features_dict = self.backbone(x, is_training=False)
        except TypeError:
            # Fallback for backbones that do not expose is_training.
            features_dict = self.backbone.forward_features(x)
        if not isinstance(features_dict, dict) or "x_norm_clstoken" not in features_dict:
            # Final fallback in case wrapper returns unexpected type.
            features_dict = self.backbone.forward_features(x)
        features = features_dict["x_norm_clstoken"]
        logits = self.head(features)
        if self.concept_mapping is None or self.concept_head is None:
            return logits

        concept_features_raw = self.concept_mapping(features)
        concept_features = concept_features_raw
        if self.config.concept_sparsity_ratio > 0.0:
            concept_features = _sparse_shrink(concept_features, self.config.concept_sparsity_ratio)
        concept_logits = self.concept_head(concept_features)
        return {
            "logits": logits,
            "concept_logits": concept_logits,
            "concept_features": concept_features,
            "concept_features_raw": concept_features_raw,
        }

    def compute_concept_logits_with_ratio(
        self,
        concept_features_raw: torch.Tensor,
        sparsity_ratio: float,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.concept_head is None:
            raise RuntimeError("Concept head not initialized.")
        concept_features = concept_features_raw
        if return_mask:
            concept_features, mask = _sparse_shrink(concept_features, sparsity_ratio, return_mask=True)
        else:
            concept_features = _sparse_shrink(concept_features, sparsity_ratio)
            mask = None
        logits = self.concept_head(concept_features)
        if return_mask:
            return logits, mask
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            if self.using_lora:
                self.backbone.train(mode)
                base_model = getattr(self.backbone, "base_model", None)
                if base_model is not None:
                    base_model.eval()
            else:
                self.backbone.eval()
        return self

    def get_trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def get_lora_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.using_lora:
            return None
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.backbone)

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not self.using_lora:
            raise RuntimeError("Attempting to load LoRA state dict while LoRA is disabled.")
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        set_peft_model_state_dict(self.backbone, state_dict, adapter_name="default")

        # PEFT loads adapter weights non-strictly because the frozen backbone is
        # intentionally absent. Re-export the adapter to make mismatches fatal.
        loaded_state = get_peft_model_state_dict(self.backbone, adapter_name="default")
        expected_keys = set(state_dict)
        loaded_keys = set(loaded_state)
        missing = sorted(expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "LoRA checkpoint key mismatch after loading: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        mismatched = []
        for name in sorted(expected_keys):
            expected = state_dict[name].detach()
            actual = loaded_state[name].detach()
            if expected.shape != actual.shape or not torch.equal(expected.cpu(), actual.cpu()):
                mismatched.append(name)
        if mismatched:
            raise RuntimeError(
                "LoRA checkpoint tensor mismatch after loading: "
                f"{mismatched[:5]} (total={len(mismatched)})"
            )
        print(f"[Checkpoint] Strictly verified {len(expected_keys)} LoRA tensors.")

    def supports_resolution(self, size: int) -> bool:
        if self.patch_size is None:
            return True
        return size % self.patch_size == 0
