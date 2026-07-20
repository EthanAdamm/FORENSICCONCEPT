"""
Configuration loader for AIGC Detection project.
Loads configuration from YAML files and provides easy access to parameters.
"""

import yaml
import os
import argparse
from typing import Dict, Any, Optional


ROBUSTNESS_PRESETS = {
    'jpeg_q75': {
        'type': 'jpeg',
        'params': {'jpeg_quality': 75},
    },
    'jpeg_q70': {
        'type': 'jpeg',
        'params': {'jpeg_quality': 70},
    },
    'gaussian_blur_sigma1': {
        'type': 'gaussian_blur',
        'params': {'gaussian_blur_sigma': 1.0},
    },
    'gaussian_blur_sigma2': {
        'type': 'gaussian_blur',
        'params': {'gaussian_blur_sigma': 2.0},
    },
    'gaussian_noise_std01': {
        'type': 'gaussian_noise',
        'params': {'gaussian_noise_std': 0.1},
    },
    'gaussian_noise_std02': {
        'type': 'gaussian_noise',
        'params': {'gaussian_noise_std': 0.2},
    },
    'downsample_scale05': {
        'type': 'downsample',
        'params': {'downsample_scale': 0.5},
    },
    'color_jitter_brightness02': {
        'type': 'color_jitter',
        'params': {
            'cj_brightness': 0.2,
            'cj_contrast': 0.0,
            'cj_saturation': 0.0,
            'cj_hue': 0.0,
        },
    },
    'color_jitter_contrast02': {
        'type': 'color_jitter',
        'params': {
            'cj_brightness': 0.0,
            'cj_contrast': 0.2,
            'cj_saturation': 0.0,
            'cj_hue': 0.0,
        },
    },
}

ROBUSTNESS_PARAM_DEFAULTS = {
    'jpeg_quality': 75,
    'gaussian_blur_sigma': 1.0,
    'gaussian_noise_std': 0.1,
    'downsample_scale': 0.5,
    'cj_brightness': 0.0,
    'cj_contrast': 0.0,
    'cj_saturation': 0.0,
    'cj_hue': 0.0,
}


def _coerce_robustness_params(raw_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = dict(raw_params or {})
    return {
        'jpeg_quality': int(raw.get('jpeg_quality', ROBUSTNESS_PARAM_DEFAULTS['jpeg_quality']) or ROBUSTNESS_PARAM_DEFAULTS['jpeg_quality']),
        'gaussian_blur_sigma': float(raw.get('gaussian_blur_sigma', ROBUSTNESS_PARAM_DEFAULTS['gaussian_blur_sigma']) or ROBUSTNESS_PARAM_DEFAULTS['gaussian_blur_sigma']),
        'gaussian_noise_std': float(raw.get('gaussian_noise_std', ROBUSTNESS_PARAM_DEFAULTS['gaussian_noise_std']) or ROBUSTNESS_PARAM_DEFAULTS['gaussian_noise_std']),
        'downsample_scale': float(raw.get('downsample_scale', ROBUSTNESS_PARAM_DEFAULTS['downsample_scale']) or ROBUSTNESS_PARAM_DEFAULTS['downsample_scale']),
        'cj_brightness': float(raw.get('cj_brightness', ROBUSTNESS_PARAM_DEFAULTS['cj_brightness']) or ROBUSTNESS_PARAM_DEFAULTS['cj_brightness']),
        'cj_contrast': float(raw.get('cj_contrast', ROBUSTNESS_PARAM_DEFAULTS['cj_contrast']) or ROBUSTNESS_PARAM_DEFAULTS['cj_contrast']),
        'cj_saturation': float(raw.get('cj_saturation', ROBUSTNESS_PARAM_DEFAULTS['cj_saturation']) or ROBUSTNESS_PARAM_DEFAULTS['cj_saturation']),
        'cj_hue': float(raw.get('cj_hue', ROBUSTNESS_PARAM_DEFAULTS['cj_hue']) or ROBUSTNESS_PARAM_DEFAULTS['cj_hue']),
    }


def _resolve_robustness_entry(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    item = dict(entry or {})
    preset = str(item.get('preset', '') or '').strip().lower()
    robustness_type = str(item.get('type', '') or '').strip().lower()
    params = _coerce_robustness_params(item)

    if preset:
        preset_cfg = ROBUSTNESS_PRESETS.get(preset)
        if preset_cfg is None:
            valid = ', '.join(sorted(ROBUSTNESS_PRESETS.keys()))
            raise ValueError(f"Unknown testing.robustness.preset='{preset}'. Valid presets: {valid}")
        robustness_type = preset_cfg['type']
        params.update(preset_cfg['params'])

    return {
        'preset': preset,
        'type': robustness_type,
        'params': params,
    }


class ConfigLoader:
    """
    Configuration loader that handles YAML configuration files.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Key in dot notation (e.g., 'data.train_dataroot')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.

        Args:
            key: Key in dot notation (e.g., 'data.train_dataroot')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update_from_args(self, args: argparse.Namespace):
        """
        Update configuration with command line arguments.

        Args:
            args: Parsed command line arguments
        """
        # Update paths if provided
        if hasattr(args, 'train_dataroot') and args.train_dataroot:
            self.set('data.train_dataroot', args.train_dataroot)
        if hasattr(args, 'test_dataroot') and args.test_dataroot:
            self.set('data.test_dataroot', args.test_dataroot)
        if hasattr(args, 'checkpoints_dir') and args.checkpoints_dir:
            self.set('models.checkpoints_dir', args.checkpoints_dir)

        # Update training parameters if provided
        if hasattr(args, 'lr') and args.lr:
            self.set('training.lr', args.lr)
        if hasattr(args, 'batch_size') and args.batch_size:
            self.set('training.batch_size', args.batch_size)
        if hasattr(args, 'niter') and args.niter:
            self.set('training.niter', args.niter)
        if hasattr(args, 'gpu_ids') and args.gpu_ids:
            self.set('training.gpu_ids', args.gpu_ids)
        if hasattr(args, 'name') and args.name:
            self.set('training.name', args.name)

    def get_testing_robustness(self) -> Dict[str, Any]:
        """
        Resolve test-time robustness preprocessing from config.

        Supports either:
        1) `testing.robustness.preset` for one-click selection, or
        2) `testing.robustness.type` + explicit params, or
        3) `testing.robustness.chain` for sequential perturbations.
        """
        robustness_cfg = self.get('testing.robustness', {}) or {}
        if not isinstance(robustness_cfg, dict):
            robustness_cfg = {}
        chain_cfg = robustness_cfg.get('chain', self.get('testing.robustness_chain', [])) or []
        if not isinstance(chain_cfg, list):
            raise ValueError("testing.robustness.chain must be a list of robustness entries.")

        resolved_chain = []
        for idx, entry in enumerate(chain_cfg):
            if not isinstance(entry, dict):
                raise ValueError(f"testing.robustness.chain[{idx}] must be a mapping.")
            resolved = _resolve_robustness_entry(entry)
            if not resolved['type']:
                raise ValueError(f"testing.robustness.chain[{idx}] must define preset or type.")
            resolved_chain.append(resolved)

        single_entry = _resolve_robustness_entry({
            'preset': robustness_cfg.get('preset', self.get('testing.robustness_preset', '')),
            'type': robustness_cfg.get('type', self.get('testing.robustness_type', '')),
            'jpeg_quality': robustness_cfg.get('jpeg_quality', self.get('testing.robustness_jpeg_quality', ROBUSTNESS_PARAM_DEFAULTS['jpeg_quality'])),
            'gaussian_blur_sigma': robustness_cfg.get('gaussian_blur_sigma', self.get('testing.robustness_gaussian_blur_sigma', ROBUSTNESS_PARAM_DEFAULTS['gaussian_blur_sigma'])),
            'gaussian_noise_std': robustness_cfg.get('gaussian_noise_std', self.get('testing.robustness_gaussian_noise_std', ROBUSTNESS_PARAM_DEFAULTS['gaussian_noise_std'])),
            'downsample_scale': robustness_cfg.get('downsample_scale', self.get('testing.robustness_downsample_scale', ROBUSTNESS_PARAM_DEFAULTS['downsample_scale'])),
            'cj_brightness': robustness_cfg.get('cj_brightness', self.get('testing.robustness_cj_brightness', ROBUSTNESS_PARAM_DEFAULTS['cj_brightness'])),
            'cj_contrast': robustness_cfg.get('cj_contrast', self.get('testing.robustness_cj_contrast', ROBUSTNESS_PARAM_DEFAULTS['cj_contrast'])),
            'cj_saturation': robustness_cfg.get('cj_saturation', self.get('testing.robustness_cj_saturation', ROBUSTNESS_PARAM_DEFAULTS['cj_saturation'])),
            'cj_hue': robustness_cfg.get('cj_hue', self.get('testing.robustness_cj_hue', ROBUSTNESS_PARAM_DEFAULTS['cj_hue'])),
        })

        if resolved_chain:
            primary = resolved_chain[0]
        else:
            primary = single_entry

        return {
            'preset': primary['preset'],
            'type': primary['type'],
            'params': primary['params'],
            'chain': resolved_chain,
        }

    def create_directories(self):
        """
        Create necessary directories based on configuration.
        """
        # Create output directories
        dirs_to_create = [self.get('models.checkpoints_dir')]

        for dir_path in dirs_to_create:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

    def to_namespace(self, is_train: bool = True) -> argparse.Namespace:
        """
        Convert configuration to argparse Namespace for compatibility.

        Returns:
            Namespace containing all configuration parameters
        """
        namespace = argparse.Namespace()

        # Set common parameters for compatibility with existing code
        namespace.dataroot = self.get('data.train_dataroot')
        extra_train_roots = self.get('data.extra_train_roots', [])
        if isinstance(extra_train_roots, str):
            extra_train_roots = [p.strip() for p in extra_train_roots.split(',') if p.strip()]
        if extra_train_roots is None:
            extra_train_roots = []
        namespace.extra_train_roots = list(extra_train_roots)
        namespace.test_dataroot = self.get('data.test_dataroot')
        namespace.train_split = self.get('data.train_split', 'train')
        namespace.val_split = self.get('data.val_split', 'val')
        namespace.checkpoints_dir = self.get('models.checkpoints_dir')
        namespace.arch = self.get('training.arch', 'res50')
        namespace.trainmode = self.get('training.trainmode', 'lora')
        namespace.modelname = self.get('training.modelname', 'CLIP:ViT-L/14@336px')
        namespace.lr = self.get('training.lr', 0.0001)
        namespace.lr_decay_factor = self.get('training.lr_decay_factor', 0.9)
        namespace.lr_min = self.get('training.lr_min', 1e-6)
        namespace.weight_decay = self.get('training.weight_decay', 0.0)
        namespace.max_grad_norm = self.get('training.max_grad_norm', 0.0)
        namespace.lr_plateau_patience = self.get('training.lr_plateau_patience', 0)
        namespace.lr_plateau_delta = self.get('training.lr_plateau_delta', 0.0)
        namespace.lr_plateau_warmup = self.get('training.lr_plateau_warmup', 0)
        namespace.batch_size = self.get('training.batch_size', 16)
        namespace.niter = self.get('training.niter', 1000)
        namespace.delr_freq = self.get('training.delr_freq', 20)
        namespace.loss_freq = self.get('training.loss_freq', 400)
        namespace.data_aug = self.get('training.data_aug', True)
        namespace.use_testshift_like_aug = self.get('training.use_testshift_like_aug', False)
        namespace.use_testshift_like_aug_eval = self.get('testing.use_testshift_like_aug_eval', False)
        namespace.testshift_like_apply_prob = self.get('training.testshift_like_apply_prob', 1.0)
        namespace.testshift_like_max_groups = self.get('training.testshift_like_max_groups', 3)
        namespace.use_aug_utils_train = self.get('training.use_aug_utils_train', False)
        namespace.aug_utils_prob = self.get('training.aug_utils_prob', 1.0)
        namespace.aug_utils_max_distortions = self.get('training.aug_utils_max_distortions', 3)
        namespace.aug_utils_num_levels = self.get('training.aug_utils_num_levels', 5)
        namespace.augmix_js_enable = self.get('training.augmix_js_enable', False)
        namespace.augmix_js_weight = self.get('training.augmix_js_weight', 12.0)
        namespace.augmix_width = self.get('training.augmix_width', 3)
        namespace.augmix_depth = self.get('training.augmix_depth', -1)
        namespace.augmix_alpha = self.get('training.augmix_alpha', 1.0)
        namespace.augmix_num_views = self.get('training.augmix_num_views', 2)
        namespace.augmix_js_eps = self.get('training.augmix_js_eps', 1.0e-7)
        namespace.loadSize = self.get('training.loadSize', 224)
        namespace.cropSize = self.get('training.cropSize', 224)
        namespace.gpu_ids = self.get('training.gpu_ids', '0')
        namespace.name = self.get('training.name', 'AIGC_Detection')
        namespace.resume_name = self.get('training.resume_name', None)

        namespace.freeze_backbone = self.get('training.freeze_backbone', True)
        namespace.trainable_blocks = self.get('training.trainable_blocks', 0)
        namespace.head_hidden_dim = self.get('training.head_hidden_dim', None)
        namespace.head_dropout = self.get('training.head_dropout', 0.1)
        namespace.use_lora = self.get('training.use_lora', False)
        namespace.lora_rank = self.get('training.lora_rank', 4)
        namespace.lora_alpha = self.get('training.lora_alpha', 8)
        namespace.lora_dropout = self.get('training.lora_dropout', 0.0)
        namespace.lora_bias = self.get('training.lora_bias', 'none')
        target_modules = self.get('training.lora_target_modules', None)
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(',') if m.strip()]
        namespace.lora_target_modules = target_modules
        namespace.lora_target_last_n = self.get('training.lora_target_last_n', None)

        namespace.dinov3_variant = self.get('training.dinov3_variant', 'vitl16')
        namespace.dinov3_backend = self.get('training.dinov3_backend', 'official')
        namespace.dinov3_repo_path = self.get('training.dinov3_repo_path', None)
        namespace.dinov3_max_blocks = self.get('training.dinov3_max_blocks', None)
        namespace.dinov3_weights = self.get('training.dinov3_weights', None)
        namespace.dinov3_disable_rope_train_jitter = self.get(
            'training.dinov3_disable_rope_train_jitter',
            False,
        )
        namespace.dinov3_accel_backend = self.get('training.dinov3_accel_backend', 'none')
        namespace.dinov3_accel_compile = self.get('training.dinov3_accel_compile', False)
        namespace.dinov3_accel_checkpointing = self.get('training.dinov3_accel_checkpointing', True)
        namespace.dinov3_accel_checkpointing_full = self.get('training.dinov3_accel_checkpointing_full', False)
        namespace.dinov3_accel_cudagraphs = self.get('training.dinov3_accel_cudagraphs', False)
        namespace.dinov3_accel_preserve_loaded_weights = self.get(
            'training.dinov3_accel_preserve_loaded_weights',
            True,
        )
        namespace.dinov3_accel_param_dtype = self.get('training.dinov3_accel_param_dtype', None)
        namespace.dinov3_accel_reduce_dtype = self.get('training.dinov3_accel_reduce_dtype', None)
        namespace.deepspeed_config_path = self.get('training.deepspeed_config_path', None)
        namespace.deepspeed_zero_stage = self.get('training.deepspeed_zero_stage', 2)
        namespace.deepspeed_grad_accum_steps = self.get('training.deepspeed_grad_accum_steps', None)
        namespace.deepspeed_offload_optimizer_device = self.get(
            'training.deepspeed_offload_optimizer_device', 'none'
        )
        namespace.deepspeed_offload_param_device = self.get('training.deepspeed_offload_param_device', 'none')
        namespace.full_finetune_enable = self.get('training.full_finetune_enable', False)
        namespace.full_finetune_precision = self.get('training.full_finetune_precision', 'fp32')
        namespace.full_finetune_grad_accum_steps = self.get('training.full_finetune_grad_accum_steps', 1)
        namespace.full_finetune_gradient_checkpointing = self.get(
            'training.full_finetune_gradient_checkpointing',
            False,
        )
        namespace.dinov3_use_concept_head = self.get('training.dinov3_use_concept_head', False)
        namespace.dinov3_concept_matrix_path = self.get('training.dinov3_concept_matrix_path', None)
        namespace.dinov3_concept_mapping_trainable = self.get('training.dinov3_concept_mapping_trainable', False)
        namespace.dinov3_concept_mapping_bias = self.get('training.dinov3_concept_mapping_bias', False)
        namespace.dinov3_concept_head_hidden_dim = self.get('training.dinov3_concept_head_hidden_dim', None)
        namespace.dinov3_concept_head_dropout = self.get('training.dinov3_concept_head_dropout', None)
        namespace.dinov3_concept_sparsity_ratio = self.get('training.dinov3_concept_sparsity_ratio', 0.0)
        concept_eval = self.get('training.dinov3_concept_eval_sparsity', None)
        if isinstance(concept_eval, str):
            items = [item.strip() for item in concept_eval.split(',') if item.strip()]
            concept_eval = [float(item) for item in items]
        namespace.dinov3_concept_eval_sparsity = concept_eval
        namespace.dinov3_concept_eval_log_path = self.get('training.dinov3_concept_eval_log_path', None)
        # CLIP options (CLS + codebook top-k head)
        namespace.clip_use_main_head = self.get('training.clip_use_main_head', True)
        namespace.clip_codebook_path = self.get('training.clip_codebook_path', None)
        namespace.clip_use_codebook_inject = self.get('training.clip_use_codebook_inject', True)
        namespace.clip_codebook_dim = self.get('training.clip_codebook_dim', 1280)
        namespace.clip_num_concepts = self.get('training.clip_num_concepts', 200)
        namespace.clip_tau = self.get('training.clip_tau', 0.1)
        namespace.clip_codebook_l2 = self.get('training.clip_codebook_l2', True)
        namespace.clip_freeze_codebook = self.get('training.clip_freeze_codebook', True)
        namespace.clip_override_codebook = self.get('training.clip_override_codebook', False)
        namespace.clip_num_patches = self.get('training.clip_num_patches', None)
        namespace.clip_cb_topk = self.get('training.clip_cb_topk', 20)
        namespace.clip_cb_topr = self.get('training.clip_cb_topr', 8)
        namespace.clip_cb_tau_w = self.get('training.clip_cb_tau_w', 0.1)
        namespace.clip_cb_weight_mode = self.get('training.clip_cb_weight_mode', 'score')
        namespace.clip_cb_mlp_hidden = self.get('training.clip_cb_mlp_hidden', None)
        namespace.clip_cb_detach_tokens = self.get('training.clip_cb_detach_tokens', False)
        namespace.clip_train_lora = self.get('training.clip_train_lora', True)
        namespace.clip_train_fc = self.get('training.clip_train_fc', True)
        namespace.clip_load_init_weights = self.get('training.clip_load_init_weights', True)
        namespace.clip_cb_loss_weight = self.get('training.clip_cb_loss_weight', 1.0)
        concept_loss = self.get('training.dinov3_concept_loss_weight', 1.0)
        namespace.dinov3_concept_loss_weight = concept_loss
        namespace.initial_eval = self.get('training.initial_eval', False)
        namespace.clip_lora_rank = self.get('training.clip_lora_rank', 4)
        namespace.clip_lora_alpha = self.get('training.clip_lora_alpha', 8)
        namespace.clip_freeze_backbone = self.get('training.clip_freeze_backbone', True)
        namespace.clip_model_path = self.get('models.pretrained.clip_model', None)
        namespace.num_threads = self.get('system.num_threads', 8)
        namespace.classes = ""
        namespace.isTrain = bool(is_train)
        namespace.pos_weight = self.get('training.pos_weight', None)
        namespace.num_classes = self.get('training.num_classes', 1)
        namespace.reverse = self.get('training.reverse', False)
        namespace.save_backbone = self.get('training.save_backbone', False)

        # Add missing parameters from base_options.py
        namespace.mode = self.get('training.mode', 'binary')
        namespace.class_bal = self.get('training.class_bal', False)
        namespace.class_bal_strategy = self.get('training.class_bal_strategy', 'auto')
        namespace.class_bal_strict = self.get('training.class_bal_strict', False)
        namespace.serial_batches = self.get('training.serial_batches', False)
        namespace.resize_or_crop = self.get('training.resize_or_crop', 'scale_and_crop')
        namespace.no_flip = self.get('training.no_flip', False)
        namespace.init_type = self.get('training.init_type', 'normal')
        namespace.init_gain = self.get('training.init_gain', 0.02)
        namespace.suffix = self.get('training.suffix', '')
        namespace.epoch = self.get('training.epoch', 'latest')

        # Data augmentation parameters
        namespace.rz_interp = self.get('training.rz_interp', 'bilinear')
        namespace.blur_prob = self.get('training.blur_prob', 0.0)
        namespace.blur_sig = self.get('training.blur_sig', '0.5')
        namespace.jpg_prob = self.get('training.jpg_prob', 0.0)
        namespace.jpg_method = self.get('training.jpg_method', 'cv2')
        namespace.jpg_qual = self.get('training.jpg_qual', '75')
        # 兼容 legacy 预处理（对齐 NPR：resize -> crop -> ImageNet 归一化）
        namespace.legacy_preprocess = self.get('training.legacy_preprocess', False)

        # Testing parameters from test_options.py
        namespace.model_path = self.get('testing.model_path', '')
        namespace.no_resize = self.get('training.no_resize', False)
        namespace.no_crop = self.get('testing.no_crop', False)
        namespace.eval = self.get('testing.eval', False)
        namespace.earlystop_epoch = self.get('training.earlystop_epoch', 15)
        # Optional: dump codebook top-k patch coords during testing
        namespace.codebook_coords_path = self.get('testing.codebook_coords_path', None)
        namespace.codebook_coords_limit = self.get('testing.codebook_coords_limit', None)
        robustness_cfg = self.get_testing_robustness()
        namespace.robustness_preset = robustness_cfg['preset']
        namespace.robustness_type = robustness_cfg['type']
        namespace.robustness_chain = robustness_cfg['chain']
        namespace.robustness_jpeg_quality = robustness_cfg['params']['jpeg_quality']
        namespace.robustness_gaussian_blur_sigma = robustness_cfg['params']['gaussian_blur_sigma']
        namespace.robustness_gaussian_noise_std = robustness_cfg['params']['gaussian_noise_std']
        namespace.robustness_downsample_scale = robustness_cfg['params']['downsample_scale']
        namespace.robustness_cj_brightness = robustness_cfg['params']['cj_brightness']
        namespace.robustness_cj_contrast = robustness_cfg['params']['cj_contrast']
        namespace.robustness_cj_saturation = robustness_cfg['params']['cj_saturation']
        namespace.robustness_cj_hue = robustness_cfg['params']['cj_hue']

        # Training parameters from train_options.py
        namespace.optim = self.get('training.optim', 'adam')
        namespace.new_optim = self.get('training.new_optim', False)
        namespace.save_latest_freq = self.get('training.save_latest_freq', 2000)
        namespace.save_epoch_freq = self.get('training.save_epoch_freq', 20)
        namespace.continue_train = self.get('training.continue_train', False)
        namespace.epoch_count = self.get('training.epoch_count', 1)
        namespace.last_epoch = self.get('training.last_epoch', -1)
        namespace.beta1 = self.get('training.beta1', 0.9)

        def flatten_config(config, parent_key=''):
            items = []
            for k, v in config.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten_config(self.config)

        # Add all flattened config items
        for k, v in flat_config.items():
            setattr(namespace, k, v)

        return namespace

    def save_config(self, output_path: str):
        """
        Save current configuration to a YAML file.

        Args:
            output_path: Path to save the configuration file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def print_config(self):
        """
        Print the current configuration in a formatted way.
        """
        print("=" * 80)
        print("Configuration:")
        print("=" * 80)

        def print_dict(d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print("  " * indent + f"{k}:")
                    print_dict(v, indent + 1)
                else:
                    print("  " * indent + f"{k}: {v}")

        print_dict(self.config)
        print("=" * 80)


def create_argument_parser():
    """
    Create argument parser for command line interface.

    Returns:
        ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="AIGC Detection Training with Configuration File Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    # Override options
    parser.add_argument(
        '--train_dataroot',
        type=str,
        help='Training data directory (overrides config)'
    )
    parser.add_argument(
        '--test_dataroot',
        type=str,
        help='Test data directory (overrides config)'
    )
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        help='Checkpoints directory (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--niter',
        type=int,
        help='Number of iterations (overrides config)'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        help='GPU IDs (overrides config)'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name (overrides config)'
    )

    return parser


def load_config_from_args(args: argparse.Namespace) -> ConfigLoader:
    """
    Load configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        ConfigLoader instance
    """
    config_loader = ConfigLoader(args.config)
    config_loader.update_from_args(args)
    config_loader.create_directories()

    return config_loader


if __name__ == "__main__":
    # Test the configuration loader
    parser = create_argument_parser()
    args = parser.parse_args()

    config_loader = load_config_from_args(args)
    config_loader.print_config()
