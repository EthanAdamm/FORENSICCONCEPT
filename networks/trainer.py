import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel
from networks.dinov3_detector import Dinov3Config, Dinov3Detector
from models import get_model


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.num_classes = getattr(opt, "num_classes", 1)
        self._deepspeed_engine = None
        self.augmix_js_loss = None
        self._best_loss = float("inf")
        self._loss_plateau_counter = 0
        self.full_finetune_enable = bool(getattr(opt, "full_finetune_enable", False))
        self.model = self._build_model(opt)

        if opt.trainmode == "lora" and opt.modelname.startswith("CLIP:"):
            if self.isTrain and not opt.continue_train:
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            self._configure_clip_trainable(opt)

        if self.isTrain:
            pos_weight = getattr(opt, "pos_weight", None)
            if self.num_classes == 1:
                if pos_weight is not None:
                    pos_tensor = torch.tensor(float(pos_weight), device=self.device)
                    self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
                else:
                    self.loss_fn = nn.BCEWithLogitsLoss()
            elif self.num_classes == 2:
                self.loss_fn = nn.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer = self._build_optimizer()
            self._init_full_finetune_plugin()
            self.optimizer.zero_grad(set_to_none=True)

        self._init_augmix_js_branch()

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(self.device)

    def _build_model(self, opt):
        if opt.trainmode == "dinov3":
            model = Dinov3Detector(self._build_dinov3_config(opt), num_classes=self.num_classes)
            self.model = model
            self._ensure_valid_resolution(opt)
            return model
        if opt.trainmode == "lora" and str(getattr(opt, "modelname", "")).startswith("CLIP:"):
            return get_model(opt.modelname, opt)
        raise ValueError(
            f"Unsupported trainmode={opt.trainmode!r}. "
            "This release supports 'dinov3' and CLIP 'lora' only."
        )

    def _configure_clip_trainable(self, opt):
        train_lora = bool(getattr(opt, "clip_train_lora", True))
        train_heads = bool(getattr(opt, "clip_train_fc", True))
        for name, param in self.model.named_parameters():
            is_visual_qkv_lora = (
                train_lora and "lora_" in name and "visual" in name and "qkv" in name
            )
            is_head = train_heads and (
                name.startswith("fc.") or name.startswith("codebook_head.")
            )
            param.requires_grad = is_visual_qkv_lora or is_head

    def _build_dinov3_config(self, opt):
        freeze_backbone = bool(getattr(opt, "freeze_backbone", True))
        trainable_blocks = int(getattr(opt, "trainable_blocks", 0) or 0)
        use_lora = bool(getattr(opt, "use_lora", False))
        gradient_checkpointing = bool(getattr(opt, "full_finetune_gradient_checkpointing", False))

        if self.full_finetune_enable:
            if freeze_backbone or use_lora or trainable_blocks != 0:
                print(
                    "[FullFinetune] Enforcing DINOv3 full fine-tune: "
                    "freeze_backbone=false, use_lora=false, trainable_blocks=0."
                )
            freeze_backbone = False
            use_lora = False
            trainable_blocks = 0
        else:
            gradient_checkpointing = False

        return Dinov3Config(
            variant=getattr(opt, "dinov3_variant", "vitl16"),
            backend=getattr(opt, "dinov3_backend", "official"),
            repo_path=getattr(opt, "dinov3_repo_path", None),
            weights_path=getattr(opt, "dinov3_weights", None),
            freeze_backbone=freeze_backbone,
            trainable_blocks=trainable_blocks,
            max_blocks=getattr(opt, "dinov3_max_blocks", None),
            head_hidden_dim=getattr(opt, "head_hidden_dim", None),
            head_dropout=getattr(opt, "head_dropout", 0.1),
            use_lora=use_lora,
            lora_rank=getattr(opt, "lora_rank", 4),
            lora_alpha=getattr(opt, "lora_alpha", 8),
            lora_dropout=getattr(opt, "lora_dropout", 0.0),
            lora_bias=getattr(opt, "lora_bias", "none"),
            lora_target_modules=getattr(opt, "lora_target_modules", None),
            lora_target_last_n=getattr(opt, "lora_target_last_n", None),
            use_concept_head=getattr(opt, "dinov3_use_concept_head", False),
            concept_matrix_path=getattr(opt, "dinov3_concept_matrix_path", None),
            concept_mapping_trainable=getattr(opt, "dinov3_concept_mapping_trainable", False),
            concept_mapping_bias=getattr(opt, "dinov3_concept_mapping_bias", False),
            concept_head_hidden_dim=getattr(opt, "dinov3_concept_head_hidden_dim", None),
            concept_head_dropout=getattr(opt, "dinov3_concept_head_dropout", None),
            concept_sparsity_ratio=getattr(opt, "dinov3_concept_sparsity_ratio", 0.0),
            concept_loss_weight=getattr(opt, "dinov3_concept_loss_weight", 1.0),
            concept_eval_sparsity=getattr(opt, "dinov3_concept_eval_sparsity", None),
            disable_rope_train_jitter=getattr(opt, "dinov3_disable_rope_train_jitter", False),
            gradient_checkpointing=gradient_checkpointing,
        )

    def _ensure_valid_resolution(self, opt):
        if not hasattr(self.model, "supports_resolution"):
            return
        patch_size = getattr(self.model, "patch_size", 1)
        window_size = getattr(self.model, "window_size", None)
        if isinstance(window_size, (list, tuple)):
            window_size = window_size[0] if window_size else None
        if isinstance(window_size, int) and window_size > 0:
            step = patch_size * window_size
        else:
            step = patch_size
        for attr in ("loadSize", "cropSize"):
            size = getattr(opt, attr, None)
            if size is None:
                continue
            if not self.model.supports_resolution(size):
                adjusted = max(step, (size // step) * step)
                if adjusted != size:
                    print(
                        f"[Resolution] Adjusting {attr} from {size} to {adjusted} to align with step {step}."
                    )
                    setattr(opt, attr, adjusted)
        if getattr(opt, "loadSize", None) and getattr(opt, "cropSize", None):
            if opt.cropSize > opt.loadSize:
                print(f"[Resolution] Adjusting cropSize from {opt.cropSize} to match loadSize {opt.loadSize}.")
                opt.cropSize = opt.loadSize

    def _init_full_finetune_plugin(self):
        precision = str(getattr(self.opt, "full_finetune_precision", "fp32") or "fp32").lower()
        if precision not in {"fp32", "fp16", "bf16"}:
            precision = "fp32"
        self.full_finetune_precision = precision
        grad_accum_raw = getattr(self.opt, "full_finetune_grad_accum_steps", 1)
        try:
            self.grad_accum_steps = max(1, int(grad_accum_raw))
        except (TypeError, ValueError):
            self.grad_accum_steps = 1
        self.max_grad_norm = float(getattr(self.opt, "max_grad_norm", 0.0) or 0.0)
        self._grad_accum_counter = 0

        self.use_amp = (
            self.full_finetune_enable
            and precision in {"fp16", "bf16"}
            and self.device.type == "cuda"
        )
        if self.use_amp:
            self.amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        if self.is_main_process and self.full_finetune_enable:
            print(
                f"[FullFinetune] enabled precision={self.full_finetune_precision} "
                f"grad_accum_steps={self.grad_accum_steps} max_grad_norm={self.max_grad_norm}"
            )

    def _build_optimizer(self):
        if self.opt.optim == 'adam':
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999),
                weight_decay=float(getattr(self.opt, "weight_decay", 0.0) or 0.0),
            )
        if self.opt.optim == 'adamw':
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.opt.lr,
                betas=(self.opt.beta1, 0.999),
                weight_decay=float(getattr(self.opt, "weight_decay", 0.0) or 0.0),
            )
        if self.opt.optim == 'sgd':
            return torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.opt.lr,
                momentum=0.0,
                weight_decay=float(getattr(self.opt, "weight_decay", 0.0) or 0.0),
            )
        raise ValueError("optim should be [adam, adamw, sgd]")

    def reset_optimizer_for_wrapped_model(self):
        """Rebuild optimizer after external wrappers (e.g., official FSDP plugin) change model params."""
        if not self.isTrain:
            return
        self.optimizer = self._build_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        if self.is_main_process:
            print("[Optimizer] Rebuilt optimizer for wrapped model.")

    def _autocast_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _optimizer_step_impl(self):
        if self.max_grad_norm > 0.0:
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            trainable_params = [
                p for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def adjust_learning_rate(self):
        decay_factor = getattr(self.opt, "lr_decay_factor", 0.9)
        min_lr = getattr(self.opt, "lr_min", 1e-6)
        if decay_factor <= 0 or decay_factor >= 1:
            print(f"[LR] Invalid decay factor {decay_factor}, falling back to 0.9")
            decay_factor = 0.9

        reached_min = False
        prev_lr = None
        for param_group in self.optimizer.param_groups:
            prev_lr = param_group['lr']
            new_lr = prev_lr * decay_factor
            if new_lr <= min_lr:
                new_lr = min_lr
                reached_min = True
            param_group['lr'] = new_lr

        if prev_lr is None:
            return False

        self.lr = self.optimizer.param_groups[0]['lr']
        print('*' * 25)
        print(f'Changing lr from {prev_lr} to {self.lr}')
        if reached_min:
            print(f'[LR] Minimum lr {min_lr} reached; further decay will be skipped')
        print('*' * 25)
        return not reached_min

    def should_adjust_lr_on_plateau(self, loss_value: float) -> bool:
        patience_raw = getattr(self.opt, "lr_plateau_patience", 0)
        try:
            patience = int(patience_raw)
        except (TypeError, ValueError):
            patience = 0
        if patience <= 0:
            return False

        warmup_raw = getattr(self.opt, "lr_plateau_warmup", 0)
        delta_raw = getattr(self.opt, "lr_plateau_delta", 0.0)
        try:
            warmup = int(warmup_raw)
        except (TypeError, ValueError):
            warmup = 0
        try:
            delta = float(delta_raw)
        except (TypeError, ValueError):
            delta = 0.0
        if self.total_steps < warmup:
            return False

        if loss_value + delta < self._best_loss:
            self._best_loss = loss_value
            self._loss_plateau_counter = 0
            return False

        self._loss_plateau_counter += 1
        if self._loss_plateau_counter >= patience:
            self._loss_plateau_counter = 0
            self._best_loss = loss_value
            return True
        return False

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device)
        if self.num_classes == 1:
            self.label = self.label.float()


    def forward(self):
        self.output = self.model(self.input)


    def get_loss(self):
        def _compute_loss(logits: torch.Tensor) -> torch.Tensor:
            if self.num_classes == 1 and logits.dim() > 1:
                logits = logits.squeeze(1)
            return self.loss_fn(logits, self.label)

        output = self.output
        if isinstance(output, dict):
            main_logits = output.get("logits")
            concept_logits = output.get("concept_logits")
            cb_logits = output.get("cb_only")
            main_loss = _compute_loss(main_logits) if main_logits is not None else None
            concept_loss = _compute_loss(concept_logits) if concept_logits is not None else None
            cb_loss = _compute_loss(cb_logits) if cb_logits is not None else None
            if main_loss is None and concept_loss is None and cb_loss is None:
                raise RuntimeError("Model output dict has no logits to compute loss.")
            total_loss = main_loss
            if concept_loss is not None:
                weight = getattr(self.opt, "dinov3_concept_loss_weight", 1.0)
                total_loss = weight * concept_loss if total_loss is None else total_loss + weight * concept_loss
            if cb_loss is not None:
                cb_weight = float(getattr(self.opt, "clip_cb_loss_weight", 1.0))
                total_loss = cb_weight * cb_loss if total_loss is None else total_loss + cb_weight * cb_loss

            return total_loss
        if isinstance(output, tuple):
            return _compute_loss(output[0])
        return _compute_loss(output)

    def _init_augmix_js_branch(self):
        self.augmix_js_enable = bool(getattr(self.opt, "augmix_js_enable", False)) and bool(self.isTrain)
        self.augmix_js_weight = float(getattr(self.opt, "augmix_js_weight", 12.0) or 12.0)
        self.augmix_width = int(getattr(self.opt, "augmix_width", 3) or 3)
        self.augmix_depth = int(getattr(self.opt, "augmix_depth", -1) or -1)
        self.augmix_alpha = float(getattr(self.opt, "augmix_alpha", 1.0) or 1.0)
        self.augmix_num_views = int(getattr(self.opt, "augmix_num_views", 2) or 2)
        self.augmix_eps = float(getattr(self.opt, "augmix_js_eps", 1e-7) or 1e-7)
        self.augmix_max_distortions = int(getattr(self.opt, "aug_utils_max_distortions", 3) or 3)
        self.augmix_num_levels = int(getattr(self.opt, "aug_utils_num_levels", 5) or 5)
        self._augmix_distort_images = None

        trainmode = str(getattr(self.opt, "trainmode", "")).lower()
        modelname = str(getattr(self.opt, "modelname", ""))
        if trainmode == "lora" and modelname.startswith("CLIP:"):
            mean, std = CLIP_MEAN, CLIP_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self._augmix_mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._augmix_std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        if not self.augmix_js_enable:
            return
        try:
            from aug_utils_train.utils_data import distort_images as aug_utils_distort_images

            self._augmix_distort_images = aug_utils_distort_images
            print(
                f"[AugMix-JS] Enabled (w={self.augmix_js_weight}, views={self.augmix_num_views}, "
                f"width={self.augmix_width}, depth={self.augmix_depth}, alpha={self.augmix_alpha})"
            )
        except Exception as exc:
            self.augmix_js_enable = False
            self._augmix_distort_images = None
            print(f"[AugMix-JS] Disabled because aug_utils_train import failed: {exc}")

    def _extract_primary_logits(self, output):
        if isinstance(output, dict):
            for key in ("logits", "main", "token_logits", "concept_logits", "cb_only", "cls_only"):
                logits = output.get(key)
                if logits is not None:
                    return logits
            raise RuntimeError("Cannot find usable logits in model output dict for AugMix-JS.")
        if isinstance(output, tuple):
            return output[0]
        return output

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        if self.num_classes == 1 or logits.shape[1] == 1:
            p_fake = torch.sigmoid(logits).view(-1, 1)
            p_real = 1.0 - p_fake
            probs = torch.cat([p_real, p_fake], dim=1)
        else:
            probs = F.softmax(logits, dim=1)
        return probs.clamp_min(self.augmix_eps)

    def _js_divergence_from_probs(self, probs_list):
        if not probs_list:
            return None
        p_mixture = torch.stack(probs_list, dim=0).mean(dim=0).clamp_min(self.augmix_eps)
        p_mixture_log = torch.log(p_mixture)
        js = 0.0
        for p in probs_list:
            js = js + torch.sum(p_mixture * (p_mixture_log - torch.log(p.clamp_min(self.augmix_eps))), dim=1).mean()
        js = js / float(len(probs_list))
        return js

    def _denormalize_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._augmix_std + self._augmix_mean).clamp(0.0, 1.0)

    def _normalize_from_unit(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._augmix_mean) / self._augmix_std

    def _augmix_single_image(self, img01_cpu: torch.Tensor) -> torch.Tensor:
        # img01_cpu: [3,H,W] in [0,1], CPU tensor.
        alpha = max(self.augmix_alpha, 1e-6)
        ws = torch.distributions.Dirichlet(torch.tensor([alpha] * self.augmix_width)).sample().tolist()
        m = torch.distributions.Beta(alpha, alpha).sample().item()
        mix = torch.zeros_like(img01_cpu)
        for i in range(self.augmix_width):
            image_aug = img01_cpu.clone()
            depth = self.augmix_depth if self.augmix_depth > 0 else int(torch.randint(1, 4, (1,)).item())
            for _ in range(depth):
                image_aug, _, _ = self._augmix_distort_images(
                    image_aug,
                    max_distortions=self.augmix_max_distortions,
                    num_levels=self.augmix_num_levels,
                )
                image_aug = image_aug.to(torch.float32).clamp(0.0, 1.0)
            mix = mix + float(ws[i]) * image_aug
        mixed = (1.0 - float(m)) * img01_cpu + float(m) * mix
        return mixed.clamp(0.0, 1.0)

    def _build_augmix_views(self, clean_input: torch.Tensor):
        # Build on CPU for aug_utils compatibility, then move back to model device.
        clean01 = self._denormalize_to_unit(clean_input.detach()).cpu()
        views = []
        for _ in range(self.augmix_num_views):
            aug_batch = torch.empty_like(clean01)
            for idx in range(clean01.shape[0]):
                aug_batch[idx] = self._augmix_single_image(clean01[idx])
            aug_batch = aug_batch.to(self.device, non_blocking=True)
            aug_batch = self._normalize_from_unit(aug_batch)
            views.append(aug_batch)
        return views

    def _compute_augmix_js_loss(self, clean_output):
        if not self.augmix_js_enable or self._augmix_distort_images is None:
            return None
        clean_logits = self._extract_primary_logits(clean_output)
        clean_probs = self._logits_to_probs(clean_logits)
        aug_inputs = self._build_augmix_views(self.input)
        aug_probs = []
        for aug_input in aug_inputs:
            aug_output = self.model(aug_input)
            aug_logits = self._extract_primary_logits(aug_output)
            aug_probs.append(self._logits_to_probs(aug_logits))
        js_loss = self._js_divergence_from_probs([clean_probs] + aug_probs)
        return js_loss

    def optimize_parameters(self):
        if self._deepspeed_engine is not None:
            self.forward()
            base_loss = self.get_loss()
            js_loss = self._compute_augmix_js_loss(self.output)
            if js_loss is not None:
                self.augmix_js_loss = js_loss.detach()
                self.loss = base_loss + self.augmix_js_weight * js_loss
            else:
                self.augmix_js_loss = None
                self.loss = base_loss
            self._deepspeed_engine.backward(self.loss)
            self._deepspeed_engine.step()
            return

        with self._autocast_context():
            self.forward()
            base_loss = self.get_loss()
            js_loss = self._compute_augmix_js_loss(self.output)
            if js_loss is not None:
                self.augmix_js_loss = js_loss.detach()
                self.loss = base_loss + self.augmix_js_weight * js_loss
            else:
                self.augmix_js_loss = None
                self.loss = base_loss

        loss_for_backward = self.loss / float(self.grad_accum_steps)
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self._grad_accum_counter += 1
        if self._grad_accum_counter % self.grad_accum_steps == 0:
            self._optimizer_step_impl()

    def finalize_epoch(self):
        if self._deepspeed_engine is not None:
            return
        if not self.isTrain or self.grad_accum_steps <= 1:
            return
        if self._grad_accum_counter % self.grad_accum_steps == 0:
            return
        if self.is_main_process:
            remain = self._grad_accum_counter % self.grad_accum_steps
            print(f"[FullFinetune] finalize_epoch flush remaining accumulated steps: {remain}")
        self._optimizer_step_impl()
