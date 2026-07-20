# from pix2pix
import os
import torch
import torch.nn as nn
from models.clip import loratorch


def _parse_primary_gpu_id(gpu_ids):
    if gpu_ids is None:
        return -1
    if isinstance(gpu_ids, (list, tuple)):
        if not gpu_ids:
            return -1
        try:
            return int(gpu_ids[0])
        except (TypeError, ValueError):
            return -1
    if isinstance(gpu_ids, str):
        parts = [part.strip() for part in gpu_ids.split(',') if part.strip()]
        if not parts:
            return -1
        try:
            return int(parts[0])
        except ValueError:
            return -1
    try:
        return int(gpu_ids)
    except (TypeError, ValueError):
        return -1


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.resume_name = getattr(opt, "resume_name", None)
        self.load_dir = os.path.join(opt.checkpoints_dir, self.resume_name or opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.rank = int(getattr(opt, "rank", 0) or 0)
        self.is_main_process = self.rank == 0
        local_rank = int(getattr(opt, "local_rank", -1) or -1)
        if torch.cuda.is_available():
            if local_rank >= 0:
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                gpu_id = _parse_primary_gpu_id(getattr(opt, "gpu_ids", None))
                self.device = torch.device(f"cuda:{gpu_id}") if gpu_id >= 0 else torch.device("cpu")
        else:
            self.device = torch.device('cpu')

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def save_networks(self, epoch):
        if not self.is_main_process:
            return
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        model_ref = self._unwrap_model()

        if getattr(self.opt, "trainmode", None) == "lora":
            state_dict = {
                'backbone': getattr(model_ref, "model", None).state_dict() if getattr(self.opt, "save_backbone", False) else None,
                'lora': loratorch.lora_state_dict(model_ref),
                'optimizer': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
            }
            if hasattr(model_ref, "fc"):
                state_dict['fc'] = model_ref.fc.state_dict()
            if getattr(model_ref, "codebook_head", None) is not None:
                state_dict['codebook_head'] = model_ref.codebook_head.state_dict()
            torch.save(state_dict, save_path)
            print(f'Saving model {save_path}')
        elif getattr(self.opt, "trainmode", None) == "dinov3" and getattr(model_ref, "using_lora", False):
            state_dict = {
                'head': model_ref.head.state_dict(),
                'backbone_lora': model_ref.get_lora_state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
            }
            if getattr(model_ref, "concept_head", None) is not None:
                state_dict['concept_head'] = model_ref.concept_head.state_dict()
            if getattr(model_ref, "concept_mapping", None) is not None:
                state_dict['concept_mapping'] = model_ref.concept_mapping.state_dict()
            torch.save(state_dict, save_path)
            print(f'Saving DINOv3 LoRA model {save_path}')
        elif getattr(self.opt, "trainmode", None) == "dinov3" and getattr(model_ref, "freeze_backbone", False) and getattr(model_ref, "trainable_blocks", 0) == 0:
            # Save only the trainable classifier parameters to keep checkpoints small.
            model_state = model_ref.state_dict()
            trainable_names = [name for name, param in model_ref.named_parameters() if param.requires_grad]
            trainable_state = {name: model_state[name].detach().cpu().clone() for name in trainable_names}
            checkpoint = {
                'trainable_state': trainable_state,
                'trainable_parameter_names': trainable_names,
                'optimizer': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'total_steps': self.total_steps,
            }
            torch.save(checkpoint, save_path)
            print(f'Saving DINOv3 trainable parameters {save_path}')
        else:
            torch.save(model_ref.state_dict(), save_path)
            print(f'Saving model {save_path}')

    # load models from the disk
    def load_networks(self, epoch):
        """Load model/optimizer state.

        Accepts either an epoch tag (e.g., '0.80_0.83') or a full path to a
        checkpoint file. This makes explicit checkpoint paths work even when the
        directory name differs from `self.load_dir`.
        """
        if isinstance(epoch, (str, os.PathLike)) and os.path.isfile(str(epoch)):
            load_path = str(epoch)
        else:
            load_filename = 'model_epoch_%s.pth' % epoch
            load_path = os.path.join(self.load_dir, load_filename)

        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        model_ref = self._unwrap_model()
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        if isinstance(state_dict, dict) and 'model' in state_dict:
            model_ref.load_state_dict(state_dict['model'])
            self.total_steps = state_dict.get('total_steps', 0)
            if self.isTrain and not self.opt.new_optim and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
        elif isinstance(state_dict, dict) and 'trainable_state' in state_dict:
            current_state = model_ref.state_dict()
            for name, tensor in state_dict['trainable_state'].items():
                if name in current_state:
                    current_state[name] = tensor
                else:
                    print(f"[Checkpoint] Warning: skipped unknown parameter '{name}'")
            model_ref.load_state_dict(current_state)
            self.total_steps = state_dict.get('total_steps', 0)
            if self.isTrain and not self.opt.new_optim and state_dict.get('optimizer'):
                self.optimizer.load_state_dict(state_dict['optimizer'])
        elif getattr(self.opt, "trainmode", None) == "lora" and isinstance(state_dict, dict):
            if 'fc' in state_dict and hasattr(model_ref, "fc"):
                model_ref.fc.load_state_dict(state_dict['fc'])
            if 'codebook_head' in state_dict and getattr(model_ref, "codebook_head", None) is not None:
                if getattr(self.opt, "clip_override_codebook", False):
                    # Load head weights but keep the codebook buffer from config.
                    ckpt_state = state_dict['codebook_head']
                    current = model_ref.codebook_head.state_dict()
                    for k, v in ckpt_state.items():
                        if k != 'codebook':
                            current[k] = v
                    model_ref.codebook_head.load_state_dict(current, strict=False)
                else:
                    model_ref.codebook_head.load_state_dict(state_dict['codebook_head'])
            # LoRA state dict may be saved from the wrapper (keys start with 'model.')
            # or directly from the CLIP backbone (no prefix). Route to the matching object.
            lora_state = state_dict.get('lora', {})
            if any(k.startswith("model.") for k in lora_state):
                model_ref.load_state_dict(lora_state, strict=False)
            else:
                model_ref.model.load_state_dict(lora_state, strict=False)
            backbone_state = state_dict.get('backbone')
            if backbone_state:
                # LoRA backbone state may omit frozen params; allow non-strict load.
                model_ref.model.load_state_dict(backbone_state, strict=False)

            # 确保 LoRA 层的 merged 标记被重置，避免某些检查点加载后 merged=True
            # 导致推理/训练阶段跳过 LoRA 权重的合并。
            def _reset_lora_merge_flag(module):
                if getattr(module, "r", 0) > 0 and hasattr(module, "merged"):
                    module.merged = False
            model_ref.apply(_reset_lora_merge_flag)

            if not self.isTrain:
                def _merge_once(module):
                    if getattr(module, "r", 0) > 0 and hasattr(module, "lora_train"):
                        try:
                            module.merged = False  # force a fresh merge
                            module.lora_train(False)
                        except Exception:
                            pass
                model_ref.apply(_merge_once)

            self.total_steps = state_dict.get('total_steps', 0)
            if self.isTrain and not self.opt.new_optim and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
        elif getattr(self.opt, "trainmode", None) == "dinov3" and getattr(model_ref, "using_lora", False):
            # Preferred LoRA checkpoint format (head + backbone_lora).
            if isinstance(state_dict, dict) and 'head' in state_dict and 'backbone_lora' in state_dict:
                model_ref.head.load_state_dict(state_dict['head'])
                model_ref.load_lora_state_dict(state_dict['backbone_lora'])
                if 'concept_head' in state_dict and getattr(model_ref, "concept_head", None) is not None:
                    model_ref.concept_head.load_state_dict(state_dict['concept_head'])
                if 'concept_mapping' in state_dict and getattr(model_ref, "concept_mapping", None) is not None:
                    model_ref.concept_mapping.load_state_dict(state_dict['concept_mapping'])
                self.total_steps = state_dict.get('total_steps', 0)
                if self.isTrain and not self.opt.new_optim and 'optimizer' in state_dict:
                    self.optimizer.load_state_dict(state_dict['optimizer'])
            else:
                # Compatibility path: initialize LoRA run from a full-finetune/plain checkpoint.
                loaded_any = False
                if isinstance(state_dict, dict):
                    head_state = {
                        k[len("head."):]: v
                        for k, v in state_dict.items()
                        if isinstance(k, str) and k.startswith("head.")
                    }
                    if head_state:
                        missing, unexpected = model_ref.head.load_state_dict(head_state, strict=False)
                        print(
                            f"[Checkpoint] Loaded head from full checkpoint (missing={len(missing)}, "
                            f"unexpected={len(unexpected)})."
                        )
                        loaded_any = True

                    backbone_state = {
                        k[len("backbone."):]: v
                        for k, v in state_dict.items()
                        if isinstance(k, str) and k.startswith("backbone.")
                    }
                    if backbone_state:
                        target_backbone = model_ref.backbone
                        base_model = getattr(target_backbone, "base_model", None)
                        if base_model is not None and hasattr(base_model, "model"):
                            target_backbone = base_model.model
                        missing, unexpected = target_backbone.load_state_dict(backbone_state, strict=False)
                        print(
                            f"[Checkpoint] Loaded backbone base weights into LoRA model "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})."
                        )
                        loaded_any = True

                    if 'concept_head' in state_dict and getattr(model_ref, "concept_head", None) is not None:
                        model_ref.concept_head.load_state_dict(state_dict['concept_head'], strict=False)
                    if 'concept_mapping' in state_dict and getattr(model_ref, "concept_mapping", None) is not None:
                        model_ref.concept_mapping.load_state_dict(state_dict['concept_mapping'], strict=False)

                if not loaded_any:
                    # Last resort for any custom checkpoint format.
                    missing, unexpected = model_ref.load_state_dict(state_dict, strict=False)
                    print(
                        f"[Checkpoint] Fallback non-strict load for LoRA model "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})."
                    )
                self.total_steps = state_dict.get('total_steps', 0) if isinstance(state_dict, dict) else self.total_steps
                if self.isTrain and not self.opt.new_optim and isinstance(state_dict, dict) and 'optimizer' in state_dict:
                    self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            model_ref.load_state_dict(state_dict)
            self.total_steps = state_dict.get('total_steps', 0) if isinstance(state_dict, dict) else self.total_steps

        if self.isTrain and not self.opt.new_optim and hasattr(self, 'optimizer'):
            # move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def test(self):
        with torch.no_grad():
            self.forward()
