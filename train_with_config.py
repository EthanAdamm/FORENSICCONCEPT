"""Configuration-driven training entry point for AIGC detection."""
import os
import time
import re
import contextlib
from datetime import timedelta
import torch
import torch.nn
import torch.distributed as dist
import copy
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None
import numpy as np
from typing import Dict, List
from validate import validate
from data import create_dataloader
import data as data_module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from networks.trainer import Trainer
from options.train_options import TrainOptions
from util import Logger
from config_loader import ConfigLoader, create_argument_parser, load_config_from_args
from plugins.dinov3_official_accel import apply_dinov3_acceleration_plugin
import random


def seed_torch(seed=1029):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class UnifiedTrainer:
    """
    Unified trainer that handles the complete training pipeline including:
    - Visual pretraining
    - Model conversion to HuggingFace format
    - Vision model replacement in LLaVA
    """

    def __init__(self, config_loader: ConfigLoader, *, is_train: bool = True):
        """
        Initialize the unified trainer.

        Args:
            config_loader: ConfigLoader instance containing all configuration
        """
        self.config = config_loader
        self.is_train = bool(is_train)
        self._init_distributed()
        self.setup_environment()
        self.setup_data_and_model()

    def _init_distributed(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA devices.")
            torch.cuda.set_device(self.local_rank)
            timeout_minutes_raw = self.config.get("system.ddp_timeout_minutes", 180)
            try:
                timeout_minutes = max(10, int(timeout_minutes_raw))
            except (TypeError, ValueError):
                timeout_minutes = 180
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=timeout_minutes),
                )
            if self.is_main_process:
                print(f"[DDP] Process group timeout set to {timeout_minutes} minutes.")

    def cleanup(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

    def _build_train_dataloader(self):
        return self._build_dataloader_for_opt(self.opt, distributed=self.distributed)

    def _build_dataloader_for_opt(self, opt_obj, *, distributed: bool, log_prefix: str = "[Data]"):
        dataset = data_module.get_dataset(opt_obj)
        shuffle = not opt_obj.serial_batches if (opt_obj.isTrain and not opt_obj.class_bal) else False
        sampler = None
        if distributed and opt_obj.class_bal:
            sampler = data_module.get_bal_sampler(
                dataset,
                distributed=True,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=int(getattr(opt_obj, "seed", 0)),
            )
            shuffle = False
            if self.is_main_process:
                print(f"{log_prefix} Using class-balanced weighted sampler in DDP mode.")
        elif distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False
        elif opt_obj.class_bal:
            sampler = data_module.get_bal_sampler(dataset)

        return DataLoader(
            dataset,
            batch_size=opt_obj.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(opt_obj.num_threads),
        )

    def _unwrap_model_for_eval(self):
        model = self.model.model
        if hasattr(model, "module"):
            return model.module
        return model

    def setup_environment(self):
        """Setup training environment."""
        seed_torch(self.config.get('training.seed', 100) + self.rank)

        # Create necessary directories
        self.config.create_directories()
        self._barrier()

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging and tensorboard."""
        # Evaluation模式下不改写训练日志/TensorBoard，保持目录干净
        if not self.is_train or not self.is_main_process:
            self.train_writer = None
            self.val_writer = None
            return

        log_dir = os.path.join(
            self.config.get('models.checkpoints_dir'),
            self.config.get('training.name')
        )
        os.makedirs(log_dir, exist_ok=True)

        # Setup file logger
        Logger(os.path.join(log_dir, self.config.get('logging.log_file', 'training.log')))

        # Setup tensorboard writers
        if self.config.get('logging.tensorboard', True) and SummaryWriter is not None:
            try:
                self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
                self.val_writer = SummaryWriter(os.path.join(log_dir, "val"))
            except (PermissionError, OSError) as exc:
                # tensorboardX fails in restricted environments; fall back to no-op writers
                print(f"[Logging] TensorBoard disabled: {exc}")
                self.train_writer = None
                self.val_writer = None
        else:
            if self.config.get('logging.tensorboard', True) and SummaryWriter is None:
                print("[Logging] TensorBoard disabled: no compatible writer is installed.")
            self.train_writer = None
            self.val_writer = None

    def setup_data_and_model(self):
        """Setup data loaders and model."""
        # Convert config to namespace for compatibility
        self.opt = self.config.to_namespace(is_train=self.is_train)
        self.opt.distributed = self.distributed
        self.opt.rank = self.rank
        self.opt.world_size = self.world_size
        self.opt.local_rank = self.local_rank if self.distributed else -1
        self.opt.ddp_find_unused_parameters = bool(
            self.config.get("training.ddp_find_unused_parameters", False)
        )
        if self.distributed:
            # Trainer reads opt.gpu_ids[0] to select device; force per-process local rank.
            self.opt.gpu_ids = str(self.local_rank)
        # Update specific paths
        if self.is_train:
            train_root = self.config.get('data.train_dataroot')
            train_split = self.config.get('data.train_split')
            if train_split in (None, '', '.', './'):
                self.opt.dataroot = train_root
            elif os.path.isabs(train_split):
                self.opt.dataroot = train_split
            else:
                self.opt.dataroot = os.path.join(train_root, train_split)

            extra_train_roots = self.config.get('data.extra_train_roots', []) or []
            if isinstance(extra_train_roots, str):
                extra_train_roots = [p.strip() for p in extra_train_roots.split(',') if p.strip()]
            self.opt.extra_train_roots = list(extra_train_roots)
            if self.is_main_process and self.opt.extra_train_roots:
                print(f"[Data] extra_train_roots: {self.opt.extra_train_roots}")

        self.test_dataroot = self.config.get('data.test_dataroot')

        # Create data loader
        self.data_loader = self._build_train_dataloader() if self.is_train else None

        # Create model
        self.model = Trainer(self.opt)
        self.accel_backend = "none"
        if self.is_train:
            self.accel_backend = apply_dinov3_acceleration_plugin(
                self.model,
                self.opt,
                is_main_process=self.is_main_process,
            )

        if self.distributed and self.accel_backend == "none":
            self.model.model = torch.nn.parallel.DistributedDataParallel(
                self.model.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=bool(getattr(self.opt, "ddp_find_unused_parameters", False)),
            )
        elif self.distributed and self.accel_backend != "none" and self.is_main_process:
            print(f"[AccelPlugin] DDP wrapper skipped (backend={self.accel_backend}).")
        if self.is_train:
            # Save an immutable snapshot of the effective config alongside checkpoints for traceability.
            if self.is_main_process:
                try:
                    cfg_out = os.path.join(self.model.save_dir, "config.yaml")
                    self.config.save_config(cfg_out)
                except Exception as exc:
                    print(f"[Config] Failed to save config snapshot: {exc}")
        self._barrier()

    # def create_test_options(self):
    #     """Reuse training options to produce a test configuration without reparsing CLI."""
    #     test_opt = copy.deepcopy(self.opt)
    #     test_opt.isTrain = False
    #     test_opt.serial_batches = True
    #     test_opt.batch_size = self.config.get('testing.batch_size', 64)
    #     test_opt.no_resize = self.config.get('testing.no_resize', False)
    #     test_opt.no_crop = self.config.get('testing.no_crop', False)
    #     test_opt.noise_type = self.config.get('testing.noise_type')
    #     test_opt.noise_ratio = self.config.get('testing.noise_ratio')
    #     test_opt.legacy_preprocess = self.config.get('training.legacy_preprocess', False)
    #     test_opt.data_aug = self.config.get('testing.data_aug', False)
    #     test_opt.use_aug_utils_eval = self.config.get('testing.use_aug_utils_eval', False)
    #     test_opt.use_testshift_like_aug_eval = self.config.get('testing.use_testshift_like_aug_eval', False)
    #     test_opt.testshift_like_apply_prob = self.config.get(
    #         'testing.testshift_like_apply_prob',
    #         self.config.get('training.testshift_like_apply_prob', 1.0),
    #     )
    #     test_opt.testshift_like_max_groups = self.config.get(
    #         'testing.testshift_like_max_groups',
    #         self.config.get('training.testshift_like_max_groups', 3),
    #     )
    #     test_opt.max_eval_samples = self.config.get('testing.max_eval_samples', 0)
    #     test_opt.loadSize = self.config.get('training.loadSize', test_opt.loadSize)
    #     test_opt.cropSize = self.config.get('training.cropSize', test_opt.cropSize)
    #     test_opt.trainmode = self.config.get('training.trainmode', test_opt.trainmode)
    #     test_opt.modelname = self.config.get('training.modelname', getattr(test_opt, 'modelname', ''))
    #     # 为了对齐 NPR 的验证预处理：默认执行 resize -> crop -> flip -> ImageNet 归一化
    #     # 当 legacy_preprocess 开启时强制开启 resize（NPR 测试阶段不使用 no_resize）
    #     if test_opt.legacy_preprocess:
    #         test_opt.no_resize = False
    #     test_opt.classes = []
    #     # Never leak training extra roots into evaluation datasets.
    #     test_opt.extra_train_roots = []
    #     test_opt.distributed = False
    #     test_opt.rank = 0
    #     test_opt.world_size = 1
    #     test_opt.local_rank = -1
    #     return test_opt
    def create_test_options(self):
        """Reuse training options to produce a test configuration without reparsing CLI."""
        test_opt = copy.deepcopy(self.opt)
        robustness_cfg = self.config.get_testing_robustness()

        test_opt.isTrain = False
        test_opt.serial_batches = True
        test_opt.batch_size = self.config.get('testing.batch_size', 64)
        test_opt.no_resize = self.config.get('testing.no_resize', False)
        test_opt.no_crop = self.config.get('testing.no_crop', False)
        test_opt.noise_type = self.config.get('testing.noise_type')
        test_opt.noise_ratio = self.config.get('testing.noise_ratio')
        test_opt.legacy_preprocess = self.config.get('training.legacy_preprocess', False)
        test_opt.data_aug = self.config.get('testing.data_aug', False)
        test_opt.use_aug_utils_eval = self.config.get('testing.use_aug_utils_eval', False)
        test_opt.use_testshift_like_aug_eval = self.config.get('testing.use_testshift_like_aug_eval', False)
        test_opt.testshift_like_apply_prob = self.config.get(
            'testing.testshift_like_apply_prob',
            self.config.get('training.testshift_like_apply_prob', 1.0),
        )
        test_opt.testshift_like_max_groups = self.config.get(
            'testing.testshift_like_max_groups',
            self.config.get('training.testshift_like_max_groups', 3),
        )

        test_opt.robustness_preset = robustness_cfg['preset']
        test_opt.robustness_type = robustness_cfg['type']
        test_opt.robustness_chain = robustness_cfg['chain']
        test_opt.robustness_jpeg_quality = robustness_cfg['params']['jpeg_quality']
        test_opt.robustness_gaussian_blur_sigma = robustness_cfg['params']['gaussian_blur_sigma']
        test_opt.robustness_gaussian_noise_std = robustness_cfg['params']['gaussian_noise_std']
        test_opt.robustness_downsample_scale = robustness_cfg['params']['downsample_scale']
        test_opt.robustness_cj_brightness = robustness_cfg['params']['cj_brightness']
        test_opt.robustness_cj_contrast = robustness_cfg['params']['cj_contrast']
        test_opt.robustness_cj_saturation = robustness_cfg['params']['cj_saturation']
        test_opt.robustness_cj_hue = robustness_cfg['params']['cj_hue']

        test_opt.max_eval_samples = self.config.get('testing.max_eval_samples', 0)
        test_opt.loadSize = self.config.get('training.loadSize', test_opt.loadSize)
        test_opt.cropSize = self.config.get('training.cropSize', test_opt.cropSize)
        test_opt.trainmode = self.config.get('training.trainmode', test_opt.trainmode)
        test_opt.modelname = self.config.get('training.modelname', getattr(test_opt, 'modelname', ''))

        # 为了对齐 NPR 的验证预处理：默认执行 resize -> crop -> flip -> ImageNet 归一化
        # 当 legacy_preprocess 开启时强制开启 resize（NPR 测试阶段不使用 no_resize）
        if test_opt.legacy_preprocess:
            test_opt.no_resize = False

        test_opt.classes = []
        test_opt.extra_train_roots = []
        test_opt.distributed = False
        test_opt.rank = 0
        test_opt.world_size = 1
        test_opt.local_rank = -1
        return test_opt

    def test_model(self):
        """Test the model on all configured test datasets."""
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        eval_groups = self.config.get('testing.groups', [])

        def _pick_primary_head(metrics: Dict[str, dict]) -> str:
            if "main" in metrics:
                return "main"
            if "token" in metrics:
                return "token"
            if "concept" in metrics:
                return "concept"
            return ""

        def _format_head_label(head_name: str) -> str:
            if head_name == "cls_only":
                return "cls_only"
            if head_name == "cb_only":
                return "cb_only"
            if head_name == "main":
                return "Main"
            if head_name == "token":
                return "Token"
            return head_name.replace("concept", "Concept")

        if not eval_groups:
            # Fallback to legacy behaviour when groups are not configured
            test_opt = self.create_test_options()
            test_vals = self.config.get('testing.test_vals', [])
            multiclass = self.config.get('testing.multiclass', [])
            accs, aucs, aps = [], [], []
            cb_accs, cb_aucs, cb_aps = [], [], []
            for v_id, val in enumerate(test_vals):
                test_opt.dataroot = f'{self.test_dataroot}/{val}'
                test_opt.dataset_name = val
                if v_id < len(multiclass) and multiclass[v_id]:
                    test_opt.classes = os.listdir(test_opt.dataroot)
                else:
                    test_opt.classes = []
                head_metrics, _ = validate(self._unwrap_model_for_eval(), test_opt)
                primary_head = _pick_primary_head(head_metrics)
                primary_metrics = head_metrics.get(primary_head, {}) if primary_head else {}
                if primary_head:
                    accs.append(primary_metrics.get("acc", 0.0))
                    aucs.append(primary_metrics.get("auc", 0.0))
                    aps.append(primary_metrics.get("ap", 0.0))
                cb_metrics = head_metrics.get("cb_only")
                if cb_metrics is not None:
                    cb_accs.append(cb_metrics.get("acc", 0.0))
                    cb_aucs.append(cb_metrics.get("auc", 0.0))
                    cb_aps.append(cb_metrics.get("ap", 0.0))
                for head_name, metrics in head_metrics.items():
                    label = _format_head_label(head_name)
                    print(
                        f"({v_id} {val:>16}) [{label}] acc: {metrics.get('acc', 0.0)*100:.1f}; "
                        f"auc: {metrics.get('auc', 0.0)*100:.1f}; ap: {metrics.get('ap', 0.0)*100:.1f}, "
                        f"racc: {metrics.get('r_acc', 0.0)*100:.1f}, "
                        f"facc: {metrics.get('f_acc', 0.0)*100:.1f};"
                    )
            mean_acc = float(np.mean(accs)) if accs else 0.0
            mean_auc = float(np.mean(aucs)) if aucs else 0.0
            mean_ap = float(np.mean(aps)) if aps else 0.0
            if not accs:
                raise RuntimeError(
                    "Evaluation produced no results. Check testing.test_vals and data.test_dataroot."
                )
            print(
                f"({len(test_vals)} {'Mean':>16}) "
                f"acc: {mean_acc*100:.1f}; auc: {mean_auc*100:.1f}; ap: {mean_ap*100:.1f}"
            )
            if cb_accs:
                cb_mean_acc = float(np.mean(cb_accs))
                cb_mean_auc = float(np.mean(cb_aucs))
                cb_mean_ap = float(np.mean(cb_aps))
                print(
                    f"({len(test_vals)} {'cb_only Mean':>16}) "
                    f"acc: {cb_mean_acc*100:.1f}; auc: {cb_mean_auc*100:.1f}; ap: {cb_mean_ap*100:.1f}"
                )
            print('*' * 25)
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            return mean_acc, mean_auc, mean_ap

        overall_accs = []
        overall_aucs = []
        overall_aps = []
        overall_head_stats: Dict[str, Dict[str, List[float]]] = {}
        overall_concept_stats: Dict[str, Dict[str, List[float]]] = {}

        for group_cfg in eval_groups:
            group_name = group_cfg.get('name') or os.path.basename(group_cfg.get('path', 'unknown'))
            base_path = group_cfg.get('path')
            if not base_path or not os.path.exists(base_path):
                print(f"[Eval] Skipping group '{group_name}' because path does not exist: {base_path}")
                continue

            mode = group_cfg.get('mode', 'subdirs')
            include = set(group_cfg.get('include', []) or [])
            exclude = set(group_cfg.get('exclude', []) or [])
            datasets_to_eval = []

            if mode == 'single':
                datasets_to_eval.append((group_name, base_path))
            else:
                if mode == 'list':
                    candidate_subdirs = list(include)
                else:  # default to enumerating direct sub directories
                    candidate_subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if include:
                    candidate_subdirs = [d for d in candidate_subdirs if d in include]
                if exclude:
                    candidate_subdirs = [d for d in candidate_subdirs if d not in exclude]
                candidate_subdirs.sort()
                for sub in candidate_subdirs:
                    datasets_to_eval.append((sub, os.path.join(base_path, sub)))

            test_opt = self.create_test_options()
            override_keys = [
                "data_aug",
                "no_resize",
                "no_crop",
                "noise_type",
                "noise_ratio",
                "use_aug_utils_train",
                "use_aug_utils_eval",
                "use_testshift_like_aug_eval",
                "testshift_like_apply_prob",
                "testshift_like_max_groups",
                "aug_utils_prob",
                "aug_utils_max_distortions",
                "aug_utils_num_levels",
                "max_eval_samples",
            ]
            for key in override_keys:
                if key in group_cfg:
                    setattr(test_opt, key, group_cfg.get(key))
            group_results = []
            for dataset_name, dataset_path in datasets_to_eval:
                if not os.path.exists(dataset_path):
                    print(f"[Eval] Skip '{dataset_name}' in group '{group_name}' (missing path: {dataset_path})")
                    continue
                test_opt.dataroot = dataset_path
                test_opt.dataset_name = dataset_name
                test_opt.classes = []  # allow recursive discovery of 0/1 structure

                head_metrics, _ = validate(self._unwrap_model_for_eval(), test_opt)
                primary_head = _pick_primary_head(head_metrics)
                primary_metrics = head_metrics.get(primary_head, {}) if primary_head else {}

                main_acc = primary_metrics.get("acc", 0.0)
                main_auc = primary_metrics.get("auc", 0.0)
                main_ap = primary_metrics.get("ap", 0.0)
                if primary_head:
                    overall_accs.append(main_acc)
                    overall_aucs.append(main_auc)
                    overall_aps.append(main_ap)

                result_entry = {
                    'dataset': dataset_name,
                    'acc': main_acc,
                    'auc': main_auc,
                    'ap': main_ap,
                    'r_acc': primary_metrics.get('r_acc', 0.0),
                    'f_acc': primary_metrics.get('f_acc', 0.0),
                }

                for head_name, metrics in head_metrics.items():
                    if head_name.startswith("concept"):
                        stats = overall_concept_stats.setdefault(head_name, {'accs': [], 'aucs': [], 'aps': []})
                        stats['accs'].append(metrics.get('acc', 0.0))
                        stats['aucs'].append(metrics.get('auc', 0.0))
                        stats['aps'].append(metrics.get('ap', 0.0))
                        result_entry.setdefault('concept_variants', {})[head_name] = metrics
                    if head_name == "cb_only":
                        stats = overall_head_stats.setdefault("cb_only", {'accs': [], 'aucs': [], 'aps': []})
                        stats['accs'].append(metrics.get('acc', 0.0))
                        stats['aucs'].append(metrics.get('auc', 0.0))
                        stats['aps'].append(metrics.get('ap', 0.0))

                    label = _format_head_label(head_name)
                    print(
                        f"[Eval] {group_name:>12} | {dataset_name:>12} :: [{label}] acc {metrics.get('acc', 0.0)*100:.2f} "
                        f"auc {metrics.get('auc', 0.0)*100:.2f} ap {metrics.get('ap', 0.0)*100:.2f} "
                        f"racc {metrics.get('r_acc', 0.0)*100:.2f} "
                        f"facc {metrics.get('f_acc', 0.0)*100:.2f}"
                    )

                group_results.append(result_entry)

            if group_results:
                group_acc = float(np.mean([r['acc'] for r in group_results]))
                group_auc = float(np.mean([r['auc'] for r in group_results]))
                group_ap = float(np.mean([r['ap'] for r in group_results]))
                summary_entry = {
                    'results': group_results,
                    'mean_acc': group_acc,
                    'mean_auc': group_auc,
                    'mean_ap': group_ap,
                }

                concept_means = {}
                for result in group_results:
                    concepts = result.get('concept_variants') or {}
                    for head_name, metrics in concepts.items():
                        concept_means.setdefault(head_name, {'accs': [], 'aucs': [], 'aps': []})
                        concept_means[head_name]['accs'].append(metrics.get('acc', 0.0))
                        concept_means[head_name]['aucs'].append(metrics.get('auc', 0.0))
                        concept_means[head_name]['aps'].append(metrics.get('ap', 0.0))

                if concept_means:
                    summary_entry['concept_means'] = {
                        name: {
                            'acc': float(np.mean(values['accs'])) if values['accs'] else 0.0,
                            'auc': float(np.mean(values['aucs'])) if values['aucs'] else 0.0,
                            'ap': float(np.mean(values['aps'])) if values['aps'] else 0.0,
                        }
                        for name, values in concept_means.items()
                    }

                print(
                    f"[Eval] {group_name:>12} | {'Mean':>12} :: "
                    f"acc {group_acc*100:.2f} auc {group_auc*100:.2f} ap {group_ap*100:.2f}"
                )
                if concept_means:
                    for name, values in summary_entry['concept_means'].items():
                        label = name.replace('concept', 'Concept')
                        print(
                            f"{'':>33}[{label} Mean] "
                            f"acc {values['acc']*100:.2f} auc {values['auc']*100:.2f} ap {values['ap']*100:.2f}"
                        )

        if not overall_accs:
            raise RuntimeError(
                "Evaluation produced no results. Every testing group was empty, missing, "
                "or failed to provide model outputs; check testing.groups paths."
            )
        mean_acc = float(np.mean(overall_accs))
        mean_auc = float(np.mean(overall_aucs))
        mean_ap = float(np.mean(overall_aps))
        print(
            f"[Eval] {'Overall':>12} | {'Mean':>12} :: "
            f"acc {mean_acc*100:.2f} auc {mean_auc*100:.2f} ap {mean_ap*100:.2f}"
        )
        cb_stats = overall_head_stats.get("cb_only")
        if cb_stats and cb_stats.get('accs'):
            cb_overall_acc = float(np.mean(cb_stats['accs']))
            cb_overall_auc = float(np.mean(cb_stats['aucs']))
            cb_overall_ap = float(np.mean(cb_stats['aps']))
            print(
                f"{'':>33}[cb_only Overall] "
                f"acc {cb_overall_acc*100:.2f} auc {cb_overall_auc*100:.2f} ap {cb_overall_ap*100:.2f}"
            )
        if overall_concept_stats:
            for head_name, values in overall_concept_stats.items():
                label = head_name.replace('concept', 'Concept')
                concept_overall_acc = float(np.mean(values['accs'])) if values['accs'] else 0.0
                concept_overall_auc = float(np.mean(values['aucs'])) if values['aucs'] else 0.0
                concept_overall_ap = float(np.mean(values['aps'])) if values['aps'] else 0.0
                print(
                    f"{'':>33}[{label} Overall] acc {concept_overall_acc*100:.2f} "
                    f"auc {concept_overall_auc*100:.2f} "
                    f"ap {concept_overall_ap*100:.2f}"
                )
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        return mean_acc, mean_auc, mean_ap


    def train(self):
        """Execute the training loop."""
        if self.is_main_process:
            print(f"Starting training with config: {self.config.config_path}")
        if self.is_main_process and self.config.get('logging.print_options', True):
            self.config.print_config()

        # Initial testing
        initial_eval = self.config.get('training.initial_eval', False)
        if initial_eval:
            if self.is_main_process:
                print("Initial model testing...")
            self.model.train()
            self.model.eval()
            self._barrier()
            if self.is_main_process:
                self.test_model()
            self._barrier()
        else:
            if self.is_main_process:
                print("Initial model testing skipped (set training.initial_eval=true to enable).")
        self.model.train()

        if self.is_main_process:
            print(f"Current working directory: {os.getcwd()}")

        plateau_patience_raw = self.config.get('training.lr_plateau_patience', 0)
        try:
            plateau_patience = int(plateau_patience_raw)
        except (TypeError, ValueError):
            plateau_patience = 0
        plateau_enabled = plateau_patience > 0
        if self.distributed and plateau_enabled:
            if self.is_main_process:
                print("[LR] Disabling loss-plateau LR adjustment in DDP to keep optimizer states synchronized.")
            plateau_enabled = False
        lr_adjustment_available = True
        delr_freq_raw = self.config.get('training.delr_freq', 20)
        try:
            delr_freq = max(1, int(delr_freq_raw))
        except (TypeError, ValueError):
            delr_freq = 20
        total_epochs_raw = self.config.get('training.niter', 1000)
        try:
            total_epochs = max(1, int(total_epochs_raw))
        except (TypeError, ValueError):
            total_epochs = 1000
        skip_eval_during_train = bool(self.config.get('training.skip_eval_during_train', False))
        skip_final_eval = bool(self.config.get('training.skip_final_eval', skip_eval_during_train))
        save_by_epoch = bool(self.config.get('training.save_by_epoch', skip_eval_during_train))
        save_before_eval = bool(self.config.get('training.save_before_eval', False))
        save_best_by_auc = bool(self.config.get('training.save_best_by_auc', False))
        keep_only_best_auc_checkpoints = bool(self.config.get('training.keep_only_best_auc_checkpoints', False))
        save_last_checkpoint = bool(self.config.get('training.save_last_checkpoint', True))
        best_auc_keep_topk_raw = self.config.get('training.keep_topk_best_auc', 5)
        try:
            best_auc_keep_topk = max(0, int(best_auc_keep_topk_raw))
        except (TypeError, ValueError):
            best_auc_keep_topk = 5
        if keep_only_best_auc_checkpoints and save_best_by_auc:
            save_by_epoch = False
            save_before_eval = False
        best_auc_delta = float(self.config.get('training.best_auc_delta', 0.0) or 0.0)
        best_auc = float("-inf")
        best_auc_epoch = -1
        if self.is_main_process and skip_eval_during_train:
            print("[Eval] Per-epoch validation disabled (training.skip_eval_during_train=true).")
            if save_by_epoch:
                print("[Checkpoint] Saving checkpoints by epoch index.")
        if self.is_main_process and save_best_by_auc:
            print(f"[Checkpoint] Best-AUC selection enabled (delta={best_auc_delta}).")
            if best_auc_keep_topk > 0:
                print(f"[Checkpoint] Keep top-{best_auc_keep_topk} best-AUC checkpoints.")
        if self.is_main_process and save_before_eval and not skip_eval_during_train:
            print("[Checkpoint] save_before_eval enabled: save checkpoint before each validation.")
        if self.is_main_process and keep_only_best_auc_checkpoints and save_best_by_auc:
            print("[Checkpoint] keep_only_best_auc_checkpoints enabled: disable epoch/pre-eval checkpoint saves.")
        batch_size_raw = self.config.get('training.batch_size', 16)
        try:
            train_batch_size = max(1, int(batch_size_raw))
        except (TypeError, ValueError):
            train_batch_size = 16
        loss_freq_raw = self.config.get('training.loss_freq', 400)
        try:
            loss_print_freq = max(1, int(loss_freq_raw))
        except (TypeError, ValueError):
            loss_print_freq = 400

        eval_every_steps_raw = self.config.get('training.eval_every_steps', 0)
        try:
            eval_every_steps = max(0, int(eval_every_steps_raw))
        except (TypeError, ValueError):
            eval_every_steps = 0
        eval_by_epoch = bool(self.config.get('training.eval_by_epoch', True))
        if skip_eval_during_train:
            eval_every_steps = 0
            eval_by_epoch = False
        if self.is_main_process and eval_every_steps > 0:
            print(f"[Eval] Step-based validation enabled every {eval_every_steps} steps.")
        if self.is_main_process and not eval_by_epoch and not skip_eval_during_train:
            print("[Eval] Epoch-end validation disabled (training.eval_by_epoch=false).")

        val_ft_cfg_raw = self.config.get('training.val_finetune_after_eval', {})
        if isinstance(val_ft_cfg_raw, bool):
            val_ft_cfg = {"enable": bool(val_ft_cfg_raw)}
        elif isinstance(val_ft_cfg_raw, dict):
            val_ft_cfg = dict(val_ft_cfg_raw)
        else:
            val_ft_cfg = {}
        val_ft_enable = bool(val_ft_cfg.get("enable", False))
        val_ft_run_on_step_eval = bool(val_ft_cfg.get("run_on_step_eval", False))
        try:
            val_ft_epochs_per_round = max(1, int(val_ft_cfg.get("epochs_per_round", 1)))
        except (TypeError, ValueError):
            val_ft_epochs_per_round = 1
        try:
            val_ft_max_steps = max(0, int(val_ft_cfg.get("max_steps", 0)))
        except (TypeError, ValueError):
            val_ft_max_steps = 0
        try:
            val_ft_batch_size = max(1, int(val_ft_cfg.get("batch_size", train_batch_size)))
        except (TypeError, ValueError):
            val_ft_batch_size = train_batch_size
        val_ft_class_bal = bool(val_ft_cfg.get("class_bal", False))
        val_ft_use_train_aug = bool(val_ft_cfg.get("use_train_augmentations", False))

        def _normalize_roots_local(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [p.strip() for p in value.split(',') if p.strip()]
            if isinstance(value, (list, tuple)):
                return [str(p).strip() for p in value if str(p).strip()]
            return [str(value).strip()]

        val_ft_roots = _normalize_roots_local(val_ft_cfg.get("roots"))
        if not val_ft_roots and bool(val_ft_cfg.get("use_testing_groups", True)):
            for group_cfg in self.config.get('testing.groups', []) or []:
                if not isinstance(group_cfg, dict):
                    continue
                path = str(group_cfg.get("path", "")).strip()
                if path:
                    val_ft_roots.append(path)
        if not val_ft_roots and self.test_dataroot:
            val_ft_roots.append(str(self.test_dataroot))
        val_ft_roots = _normalize_roots_local(val_ft_roots)
        val_ft_roots = [p for p in val_ft_roots if os.path.exists(p)]
        if val_ft_enable and not val_ft_roots:
            if self.is_main_process:
                print("[ValFT] Disabled: no valid roots were found.")
            val_ft_enable = False
        if val_ft_enable and self.is_main_process:
            print(
                f"[ValFT] enabled roots={val_ft_roots} bs={val_ft_batch_size} "
                f"epochs_per_round={val_ft_epochs_per_round} max_steps={val_ft_max_steps or 'all'} "
                f"class_bal={val_ft_class_bal} use_train_aug={val_ft_use_train_aug} "
                f"run_on_step_eval={val_ft_run_on_step_eval}"
            )

        curriculum_phases = self.config.get('training.curriculum_phases', []) or []
        if not isinstance(curriculum_phases, list):
            curriculum_phases = []
        current_phase_idx = -1
        last_eval_step = -1
        best_auc_marker = ""

        def _prune_best_auc_checkpoints():
            if not self.is_main_process:
                return
            if not save_best_by_auc:
                return
            if best_auc_keep_topk <= 0:
                return
            save_dir = getattr(self.model, "save_dir", None)
            if not save_dir or not os.path.isdir(save_dir):
                return

            pattern = re.compile(r"^model_epoch_best_auc_([0-9]*\.?[0-9]+)_(e|s)(\d+)\.pth$")
            scored = []
            for filename in os.listdir(save_dir):
                match = pattern.match(filename)
                if not match:
                    continue
                auc = float(match.group(1))
                marker = int(match.group(3))
                scored.append((auc, marker, filename))

            if len(scored) <= best_auc_keep_topk:
                return

            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            to_remove = scored[best_auc_keep_topk:]
            removed = 0
            for _, _, filename in to_remove:
                path = os.path.join(save_dir, filename)
                try:
                    os.remove(path)
                    removed += 1
                except FileNotFoundError:
                    continue
            if removed > 0:
                print(
                    f"[Checkpoint] Pruned {removed} best-AUC checkpoints in {save_dir}, "
                    f"keeping top {best_auc_keep_topk}."
                )

        def _normalize_roots(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [p.strip() for p in value.split(',') if p.strip()]
            if isinstance(value, (list, tuple)):
                return [str(p).strip() for p in value if str(p).strip()]
            return [str(value).strip()]

        def _active_phase_index(epoch_one_based):
            if not curriculum_phases:
                return -1
            for idx, phase in enumerate(curriculum_phases):
                if not isinstance(phase, dict):
                    continue
                start_raw = phase.get("start_epoch", 1)
                end_raw = phase.get("end_epoch", total_epochs)
                try:
                    start_epoch = max(1, int(start_raw))
                except (TypeError, ValueError):
                    start_epoch = 1
                try:
                    end_epoch = max(start_epoch, int(end_raw))
                except (TypeError, ValueError):
                    end_epoch = total_epochs
                if start_epoch <= epoch_one_based <= end_epoch:
                    return idx
            return -1

        def _apply_curriculum_if_needed(epoch_one_based):
            nonlocal current_phase_idx
            nonlocal train_batch_size
            target_idx = _active_phase_index(epoch_one_based)
            if target_idx < 0 or target_idx == current_phase_idx:
                return

            phase = curriculum_phases[target_idx]
            phase_name = str(phase.get("name", f"phase_{target_idx}"))

            if "train_dataroot" in phase and phase.get("train_dataroot"):
                self.opt.dataroot = str(phase.get("train_dataroot"))
            if "extra_train_roots" in phase:
                self.opt.extra_train_roots = _normalize_roots(phase.get("extra_train_roots"))

            override_keys = [
                "data_aug",
                "use_testshift_like_aug",
                "testshift_like_apply_prob",
                "testshift_like_max_groups",
                "use_aug_utils_train",
                "aug_utils_prob",
                "aug_utils_max_distortions",
                "aug_utils_num_levels",
                "no_resize",
                "loadSize",
                "cropSize",
            ]
            for key in override_keys:
                if key in phase:
                    setattr(self.opt, key, phase.get(key))

            if "batch_size" in phase:
                try:
                    self.opt.batch_size = max(1, int(phase.get("batch_size")))
                    train_batch_size = self.opt.batch_size
                except (TypeError, ValueError):
                    pass

            self.data_loader = self._build_train_dataloader()
            current_phase_idx = target_idx
            if self.is_main_process:
                print(
                    f"[Curriculum] Switched to '{phase_name}' at epoch {epoch_one_based}: "
                    f"dataroot={self.opt.dataroot}, extra_roots={getattr(self.opt, 'extra_train_roots', [])}, "
                    f"bs={self.opt.batch_size}, testshift_prob={getattr(self.opt, 'testshift_like_apply_prob', 1.0)}"
                )

        def _run_validation(eval_label, *, epoch_for_tag=None, step_for_tag=None):
            nonlocal best_auc
            nonlocal best_auc_epoch
            nonlocal best_auc_marker
            nonlocal last_eval_step

            did_save_epoch_before_eval = False
            if save_before_eval:
                self._barrier()
                if self.is_main_process:
                    if epoch_for_tag is not None and save_by_epoch:
                        pre_eval_tag = str(epoch_for_tag)
                        did_save_epoch_before_eval = True
                    elif step_for_tag is not None:
                        pre_eval_tag = f"pre_eval_s{int(step_for_tag)}"
                    elif epoch_for_tag is not None:
                        pre_eval_tag = f"pre_eval_e{int(epoch_for_tag)}"
                    else:
                        pre_eval_tag = f"pre_eval_s{int(self.model.total_steps)}"
                    self.model.save_networks(pre_eval_tag)
                    print(f"[Checkpoint] Saved pre-eval checkpoint: {pre_eval_tag}")
                self._barrier()

            self.model.eval()
            self._barrier()
            run_eval_on_all_ranks = self.distributed and self.accel_backend in {"official_fsdp", "deepspeed"}
            if self.is_main_process or run_eval_on_all_ranks:
                # FSDP/DeepSpeed evaluation requires all ranks to participate in forward pass.
                if (not self.is_main_process) and run_eval_on_all_ranks:
                    with open(os.devnull, "w") as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            acc, auc, ap = self.test_model()
                else:
                    acc, auc, ap = self.test_model()

            if self.is_main_process:
                if epoch_for_tag is not None:
                    if save_by_epoch and not did_save_epoch_before_eval:
                        self.model.save_networks(str(int(epoch_for_tag)))
                    elif not save_best_by_auc:
                        self.model.save_networks(f'{acc:.3f}_{auc:.3f}')
                else:
                    if save_by_epoch:
                        self.model.save_networks(f"step_{int(step_for_tag)}")
                    elif not save_best_by_auc:
                        self.model.save_networks(f'{acc:.3f}_{auc:.3f}')

                if save_best_by_auc and (auc > best_auc + best_auc_delta):
                    best_auc = float(auc)
                    if epoch_for_tag is not None:
                        best_auc_epoch = int(epoch_for_tag)
                        best_auc_marker = f"epoch={best_auc_epoch}"
                        save_tag = f"best_auc_{best_auc:.4f}_e{best_auc_epoch}"
                    else:
                        best_auc_marker = f"step={int(step_for_tag)}"
                        save_tag = f"best_auc_{best_auc:.4f}_s{int(step_for_tag)}"
                    self.model.save_networks(save_tag)
                    print(
                        f"[Checkpoint] New best AUC={best_auc:.6f} at {best_auc_marker}; "
                        "saved best checkpoint."
                    )
                _prune_best_auc_checkpoints()
            else:
                acc, auc, ap = 0.0, 0.0, 0.0

            if self.distributed:
                metrics = torch.tensor([acc, auc, ap], dtype=torch.float32, device=self.model.device)
                dist.broadcast(metrics, src=0)
                acc, auc, ap = float(metrics[0].item()), float(metrics[1].item()), float(metrics[2].item())
            self._barrier()

            if self.val_writer:
                self.val_writer.add_scalar('accuracy', acc, self.model.total_steps)
                self.val_writer.add_scalar('auc', auc, self.model.total_steps)
                self.val_writer.add_scalar('ap', ap, self.model.total_steps)

            if self.is_main_process:
                print(f"(Val @ {eval_label}) acc: {acc}; auc: {auc}; ap: {ap}")
            self.model.train()
            last_eval_step = int(self.model.total_steps)

        def _run_val_finetune_after_eval(eval_label, *, epoch_for_sampler):
            if not val_ft_enable:
                return

            ft_opt = copy.deepcopy(self.opt)
            ft_opt.isTrain = True
            ft_opt.serial_batches = False
            ft_opt.class_bal = bool(val_ft_class_bal)
            ft_opt.batch_size = int(val_ft_batch_size)
            ft_opt.dataroot = val_ft_roots[0]
            ft_opt.extra_train_roots = list(val_ft_roots[1:])

            if not val_ft_use_train_aug:
                # Default to "distribution alignment": no synthetic train-time corruption.
                ft_opt.data_aug = False
                ft_opt.no_flip = True
                ft_opt.use_aug_utils_train = False
                ft_opt.use_testshift_like_aug = False

            # Optional fine control for this stage.
            for key in [
                "data_aug",
                "no_flip",
                "no_resize",
                "loadSize",
                "cropSize",
                "use_aug_utils_train",
                "aug_utils_prob",
                "aug_utils_max_distortions",
                "aug_utils_num_levels",
                "use_testshift_like_aug",
                "testshift_like_apply_prob",
                "testshift_like_max_groups",
            ]:
                if key in val_ft_cfg:
                    setattr(ft_opt, key, val_ft_cfg.get(key))

            self._barrier()
            ft_loader = self._build_dataloader_for_opt(
                ft_opt,
                distributed=self.distributed,
                log_prefix="[ValFT]",
            )
            self._barrier()

            start_steps = int(self.model.total_steps)
            ft_steps = 0
            t0 = time.time()
            self.model.train()

            for local_ep in range(val_ft_epochs_per_round):
                if hasattr(ft_loader, "dataset") and hasattr(ft_loader.dataset, "set_epoch"):
                    ft_loader.dataset.set_epoch(int(epoch_for_sampler) * 1000 + int(local_ep))
                if hasattr(ft_loader, "sampler") and hasattr(ft_loader.sampler, "set_epoch"):
                    ft_loader.sampler.set_epoch(int(epoch_for_sampler) * 1000 + int(local_ep))

                for data in ft_loader:
                    if val_ft_max_steps > 0 and ft_steps >= val_ft_max_steps:
                        break
                    self.model.total_steps += 1
                    self.model.set_input(data)
                    self.model.optimize_parameters()
                    ft_steps += 1

                    if self.is_main_process and (self.model.total_steps % loss_print_freq == 0):
                        loss_scalar = float(self.model.loss.detach().item())
                        print(
                            f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())} "
                            f"[ValFT] loss: {loss_scalar} at step: {self.model.total_steps} "
                            f"(trigger={eval_label})"
                        )

                if val_ft_max_steps > 0 and ft_steps >= val_ft_max_steps:
                    break

            self._barrier()
            if self.is_main_process:
                dt = time.time() - t0
                print(
                    f"[ValFT] done trigger={eval_label}, steps={ft_steps}, "
                    f"global_step {start_steps}->{self.model.total_steps}, time={dt:.1f}s"
                )

        # Training loop
        for epoch in range(total_epochs):
            _apply_curriculum_if_needed(epoch + 1)
            epoch_start_time = time.time()
            epoch_iter = 0
            if hasattr(self.data_loader, "dataset") and hasattr(self.data_loader.dataset, "set_epoch"):
                self.data_loader.dataset.set_epoch(epoch)
            if hasattr(self.data_loader, "sampler") and hasattr(self.data_loader.sampler, "set_epoch"):
                self.data_loader.sampler.set_epoch(epoch)

            if self.is_main_process:
                print(f"Epoch {epoch + 1}/{total_epochs}")

            for i, data in enumerate(self.data_loader):
                self.model.total_steps += 1
                epoch_iter += train_batch_size
                self.model.set_input(data)
                self.model.optimize_parameters()
                # Log training loss
                if self.model.total_steps % loss_print_freq == 0:
                    loss_scalar = float(self.model.loss.detach().item())
                    js_scalar = None
                    if getattr(self.model, "augmix_js_loss", None) is not None:
                        js_scalar = float(self.model.augmix_js_loss.item())
                    if self.is_main_process:
                        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                        msg = (
                            f"{timestamp} Train loss: {loss_scalar} at step: {self.model.total_steps} lr {self.model.lr}"
                        )
                        if js_scalar is not None:
                            msg += f" augmix_js: {js_scalar}"
                        print(msg)

                    if plateau_enabled and lr_adjustment_available and self.model.should_adjust_lr_on_plateau(loss_scalar):
                        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                        if self.is_main_process:
                            print(f"{timestamp} Loss plateau detected (patience {plateau_patience}); adjusting lr")
                        lr_adjustment_available = self.model.adjust_learning_rate()
                        if not lr_adjustment_available:
                            plateau_enabled = False

                    if self.train_writer:
                        self.train_writer.add_scalar('loss', loss_scalar, self.model.total_steps)
                        if js_scalar is not None:
                            self.train_writer.add_scalar('augmix_js_loss', js_scalar, self.model.total_steps)

                if (
                    eval_every_steps > 0
                    and self.model.total_steps % eval_every_steps == 0
                ):
                    _run_validation(
                        f"step {self.model.total_steps}",
                        step_for_tag=int(self.model.total_steps),
                    )
                    if val_ft_enable and val_ft_run_on_step_eval:
                        _run_val_finetune_after_eval(
                            f"step {self.model.total_steps}",
                            epoch_for_sampler=int(epoch + 1),
                        )

            if hasattr(self.model, "finalize_epoch"):
                self.model.finalize_epoch()

            # Learning rate decay
            if (
                lr_adjustment_available
                and not plateau_enabled
                and epoch % delr_freq == 0
                and epoch != 0
            ):
                if self.is_main_process:
                    print(f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())} "
                          f"Changing lr at the end of epoch {epoch}, iters {self.model.total_steps}")
                lr_adjustment_available = self.model.adjust_learning_rate()

            if skip_eval_during_train:
                self._barrier()
                if self.is_main_process:
                    if save_by_epoch:
                        self.model.save_networks(str(epoch + 1))
                    else:
                        if save_last_checkpoint:
                            self.model.save_networks('last')
                self._barrier()
                self.model.train()
                continue

            if eval_by_epoch:
                if last_eval_step == int(self.model.total_steps):
                    if self.is_main_process:
                        print(
                            f"[Eval] Skip epoch-end validation at epoch {epoch + 1}: "
                            f"already validated at step {self.model.total_steps}."
                        )
                else:
                    _run_validation(
                        f"epoch {epoch + 1}",
                        epoch_for_tag=int(epoch + 1),
                    )
                    _run_val_finetune_after_eval(
                        f"epoch {epoch + 1}",
                        epoch_for_sampler=int(epoch + 1),
                    )

        # Final testing and saving
        self.model.eval()
        # self.validate_holdout_split()
        self._barrier()
        if self.is_main_process:
            if not skip_final_eval:
                self.test_model()
            else:
                print("[Eval] Final validation disabled (training.skip_final_eval=true).")
            if save_best_by_auc and (best_auc_epoch > 0 or best_auc_marker):
                marker = best_auc_marker or f"epoch={best_auc_epoch}"
                print(f"[Checkpoint] Best AUC summary: {marker}, auc={best_auc:.6f}")
            if save_last_checkpoint:
                self.model.save_networks('last')
            print("Training completed!")
        self._barrier()

def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load configuration
    config_loader = load_config_from_args(args)

    # Create and run trainer
    trainer = UnifiedTrainer(config_loader)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()
