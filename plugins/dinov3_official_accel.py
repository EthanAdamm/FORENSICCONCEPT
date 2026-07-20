from __future__ import annotations

import json
import sys
import types
import importlib
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn


def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_official_repo_from_opt(opt) -> Optional[Path]:
    if getattr(opt, "dinov3_repo_path", None):
        p = Path(opt.dinov3_repo_path).expanduser().resolve()
        if (p / "dinov3").is_dir():
            return p
    return None


def _ensure_official_import(opt) -> Path:
    repo_path = _resolve_official_repo_from_opt(opt)
    if repo_path is not None:
        repo_str = str(repo_path)
        if repo_str in sys.path:
            sys.path.remove(repo_str)
        sys.path.insert(0, repo_str)
        return repo_path

    try:
        import dinov3 as dinov3_pkg

        return Path(dinov3_pkg.__file__).resolve().parents[1]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Official DINOv3 repository not found. Clone "
            "https://github.com/facebookresearch/dinov3 and set "
            "training.dinov3_repo_path, or install DINOv3 in the active environment."
        ) from exc


def _effective_precision(opt) -> str:
    # Reuse existing full-finetune precision key if provided.
    p = str(getattr(opt, "full_finetune_precision", "fp32") or "fp32").strip().lower()
    if p not in {"fp16", "bf16", "fp32"}:
        p = "fp32"
    return p


def _install_fsdp_compat_shim_for_official(*, is_main_process: bool) -> None:
    """
    Compatibility shim for older PyTorch builds where official DINOv3 import
    fails due to missing torch.distributed.fsdp.register_fsdp_forward_method.
    """
    try:
        import torch.distributed.fsdp as torch_fsdp
    except Exception:  # noqa: BLE001
        return

    if hasattr(torch_fsdp, "register_fsdp_forward_method"):
        return

    def _noop_register_fsdp_forward_method(module, method_name):  # noqa: ANN001
        return module

    setattr(torch_fsdp, "register_fsdp_forward_method", _noop_register_fsdp_forward_method)
    if is_main_process:
        print(
            "[AccelPlugin] torch.distributed.fsdp.register_fsdp_forward_method is missing; "
            "installed no-op compatibility shim."
        )

    # torch<2.6 may expose FSDPState under _composable path only.
    legacy_parent = "torch.distributed.fsdp._fully_shard"
    legacy_state_mod = f"{legacy_parent}._fsdp_state"
    if legacy_state_mod not in sys.modules:
        try:
            src_state_mod = importlib.import_module("torch.distributed._composable.fsdp._fsdp_state")
            if legacy_parent not in sys.modules:
                parent_mod = types.ModuleType(legacy_parent)
                parent_mod.__path__ = []  # type: ignore[attr-defined]
                sys.modules[legacy_parent] = parent_mod
            sys.modules[legacy_state_mod] = src_state_mod
            if is_main_process:
                print(
                    "[AccelPlugin] installed compatibility alias "
                    "torch.distributed.fsdp._fully_shard._fsdp_state -> "
                    "torch.distributed._composable.fsdp._fsdp_state"
                )
        except Exception:  # noqa: BLE001
            pass


def _patch_official_fsdp_transformer_for_torch25(is_main_process: bool) -> None:
    """
    Torch 2.5.x may fail when fully_shard() is also applied at the backbone root
    (patch_embed conv sees mixed Tensor/DTensor). Keep official block-level FSDP
    wrapping but skip root-level fully_shard for transformer backbones.
    """
    if not str(torch.__version__).startswith("2.5"):
        return
    try:
        import dinov3.fsdp.ac_compile_parallelize as acp_mod
    except Exception:  # noqa: BLE001
        return

    fully_shard = getattr(acp_mod, "fully_shard", None)
    nn_mod = getattr(acp_mod, "nn", None)
    if fully_shard is None or nn_mod is None:
        return

    def _fsdp_transformer_torch25(fsdp_config, model):  # noqa: ANN001
        blocks = model.blocks
        assert isinstance(blocks, nn_mod.ModuleList)
        for block_id, block in enumerate(blocks):
            blocks[block_id] = fully_shard(block, **fsdp_config, reshard_after_forward=True)
        # On torch 2.5.x, block prefetch hooks can trigger
        # `'FSDPCommContext' has no attribute 'all_gather_copy_in_stream'`.
        # Keep sharding, but disable forward/backward prefetch wiring.
        # Intentionally skip `fully_shard(model, ...)` on torch 2.5.x.

    acp_mod.fsdp_transformer = _fsdp_transformer_torch25
    if is_main_process:
        print(
            "[AccelPlugin] Applied torch2.5 compatibility patch: "
            "official fsdp_transformer now shards blocks only (no root fully_shard)."
        )


@contextlib.contextmanager
def _patch_to_empty_to_preserve_loaded_weights(*, enabled: bool, is_main_process: bool):
    """
    Official ac_compile_parallelize() calls model.to_empty(device="cuda") at the end.
    In this project we call it after loading checkpoints, so to_empty would drop weights.
    """
    if not enabled:
        yield
        return

    original_to_empty = nn.Module.to_empty

    def _to_empty_keep_weights(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and len(args) > 0:
            device = args[0]
        if device is None:
            return self
        return self.to(device)

    nn.Module.to_empty = _to_empty_keep_weights
    if is_main_process:
        print(
            "[AccelPlugin] Patched nn.Module.to_empty during official_fsdp apply "
            "to preserve already-loaded checkpoint weights."
        )
    try:
        yield
    finally:
        nn.Module.to_empty = original_to_empty


def _apply_official_fsdp(trainer_model, opt, *, is_main_process: bool) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("official_fsdp requires distributed process group. Launch with torchrun.")
    if str(getattr(opt, "trainmode", "")).lower() != "dinov3":
        raise RuntimeError("official_fsdp plugin currently supports only training.trainmode=dinov3.")

    repo_path = _ensure_official_import(opt)
    _install_fsdp_compat_shim_for_official(is_main_process=is_main_process)
    _patch_official_fsdp_transformer_for_torch25(is_main_process=is_main_process)
    try:
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import official DINOv3 FSDP2 wrapper. "
            "This typically means your PyTorch version is too old for official FSDP2 APIs "
            "(e.g., missing torch.distributed.fsdp.register_fsdp_forward_method). "
            "Use a newer PyTorch build compatible with the official DINOv3 code."
        ) from exc

    model = trainer_model.model
    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        raise RuntimeError("Dinov3Detector backbone/head not found; cannot apply official FSDP plugin.")

    precision = _effective_precision(opt)
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            checkpointing=_as_bool(getattr(opt, "dinov3_accel_checkpointing", True), True),
            checkpointing_full=_as_bool(getattr(opt, "dinov3_accel_checkpointing_full", False), False),
            compile=_as_bool(getattr(opt, "dinov3_accel_compile", False), False),
            cudagraphs=_as_bool(getattr(opt, "dinov3_accel_cudagraphs", False), False),
        ),
        compute_precision=SimpleNamespace(
            param_dtype=str(getattr(opt, "dinov3_accel_param_dtype", precision) or precision).lower(),
            reduce_dtype=str(getattr(opt, "dinov3_accel_reduce_dtype", precision) or precision).lower(),
            sharding_strategy="SHARD_GRAD_OP",
        ),
    )

    for attr in ("param_dtype", "reduce_dtype"):
        val = getattr(cfg.compute_precision, attr)
        if val not in {"fp16", "bf16", "fp32"}:
            setattr(cfg.compute_precision, attr, precision)

    trained_model = nn.ModuleDict(
        {
            "backbone": model.backbone,
            "head": model.head,
        }
    )
    if getattr(model, "concept_mapping", None) is not None:
        trained_model["concept_mapping"] = model.concept_mapping
    if getattr(model, "concept_head", None) is not None:
        trained_model["concept_head"] = model.concept_head

    if is_main_process:
        print(
            "[AccelPlugin] Applying official FSDP2 from DINOv3 "
            f"(repo={repo_path}, param_dtype={cfg.compute_precision.param_dtype}, "
            f"reduce_dtype={cfg.compute_precision.reduce_dtype}, "
            f"checkpointing={cfg.train.checkpointing}, compile={cfg.train.compile})."
        )

    preserve_loaded_weights = _as_bool(
        getattr(opt, "dinov3_accel_preserve_loaded_weights", True),
        True,
    )
    with _patch_to_empty_to_preserve_loaded_weights(
        enabled=preserve_loaded_weights,
        is_main_process=is_main_process,
    ):
        ac_compile_parallelize(trained_model=trained_model, inference_only_models=[], cfg=cfg)

    # Put wrapped modules back into detector.
    model.backbone = trained_model["backbone"]
    model.head = trained_model["head"]
    if "concept_mapping" in trained_model:
        model.concept_mapping = trained_model["concept_mapping"]
    if "concept_head" in trained_model:
        model.concept_head = trained_model["concept_head"]

    trainer_model.reset_optimizer_for_wrapped_model()
    if str(torch.__version__).startswith("2.5") and float(getattr(trainer_model, "max_grad_norm", 0.0) or 0.0) > 0.0:
        # torch2.5 composable FSDP gradients may mix Tensor/DTensor objects,
        # and clip_grad_norm_ can fail on mixed groups.
        trainer_model.max_grad_norm = 0.0
        if is_main_process:
            print("[AccelPlugin] Disabled max_grad_norm clipping for official_fsdp compatibility.")
    if is_main_process:
        print("[AccelPlugin] official_fsdp applied and optimizer rebuilt.")


def _build_deepspeed_config(opt) -> dict:
    cfg_path = getattr(opt, "deepspeed_config_path", None)
    if cfg_path:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    stage = int(getattr(opt, "deepspeed_zero_stage", 2) or 2)
    stage = max(0, min(3, stage))
    grad_accum = int(getattr(opt, "deepspeed_grad_accum_steps", getattr(opt, "full_finetune_grad_accum_steps", 1)) or 1)
    grad_accum = max(1, grad_accum)
    precision = _effective_precision(opt)
    grad_clip = float(getattr(opt, "max_grad_norm", 0.0) or 0.0)

    ds_cfg = {
        "train_micro_batch_size_per_gpu": int(getattr(opt, "batch_size", 1) or 1),
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "allgather_bucket_size": int(2e8),
            "reduce_bucket_size": int(2e8),
        },
    }
    if grad_clip > 0.0:
        ds_cfg["gradient_clipping"] = grad_clip
    if precision == "bf16":
        ds_cfg["bf16"] = {"enabled": True}
        ds_cfg["fp16"] = {"enabled": False}
    elif precision == "fp16":
        ds_cfg["fp16"] = {"enabled": True}
        ds_cfg["bf16"] = {"enabled": False}
    else:
        ds_cfg["fp16"] = {"enabled": False}
        ds_cfg["bf16"] = {"enabled": False}

    offload_opt = str(getattr(opt, "deepspeed_offload_optimizer_device", "none") or "none").lower()
    offload_param = str(getattr(opt, "deepspeed_offload_param_device", "none") or "none").lower()
    if offload_opt in {"cpu", "nvme"}:
        ds_cfg["zero_optimization"]["offload_optimizer"] = {"device": offload_opt}
    if offload_param in {"cpu", "nvme"}:
        ds_cfg["zero_optimization"]["offload_param"] = {"device": offload_param}

    return ds_cfg


def _apply_deepspeed(trainer_model, opt, *, is_main_process: bool) -> None:
    if str(getattr(opt, "trainmode", "")).lower() != "dinov3":
        raise RuntimeError("deepspeed plugin currently supports only training.trainmode=dinov3.")
    _ensure_official_import(opt)

    try:
        import deepspeed
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "DeepSpeed is not available. Install it first (matching your CUDA/PyTorch build)."
        ) from exc

    ds_cfg = _build_deepspeed_config(opt)
    model = trainer_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for DeepSpeed initialization.")

    if is_main_process:
        stage = ds_cfg.get("zero_optimization", {}).get("stage", 0)
        print(f"[AccelPlugin] Initializing DeepSpeed (ZeRO stage={stage}).")

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=params,
        optimizer=trainer_model.optimizer,
        config=ds_cfg,
    )

    trainer_model.model = engine
    trainer_model.optimizer = optimizer
    trainer_model._deepspeed_engine = engine
    trainer_model.use_amp = False
    trainer_model.scaler = None
    if is_main_process:
        print("[AccelPlugin] deepspeed initialized.")


def apply_dinov3_acceleration_plugin(trainer_model, opt, *, is_main_process: bool) -> str:
    """Apply optional acceleration plugin. Default path is no-op."""
    backend = str(getattr(opt, "dinov3_accel_backend", "none") or "none").strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return "none"

    if backend == "official_fsdp":
        _apply_official_fsdp(trainer_model, opt, is_main_process=is_main_process)
        return backend
    if backend == "deepspeed":
        _apply_deepspeed(trainer_model, opt, is_main_process=is_main_process)
        return backend

    raise ValueError(
        f"Unsupported training.dinov3_accel_backend={backend}. "
        "Expected one of: none, official_fsdp, deepspeed."
    )
