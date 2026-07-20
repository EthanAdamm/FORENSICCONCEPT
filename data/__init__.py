import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Sampler
import math

from .datasets import dataset_folder, infer_single_label_from_root, is_dfdc_train_faces_root

'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

def _build_dataset_from_root(opt, root):
    import os
    if not root or not os.path.exists(root):
        raise FileNotFoundError(f"Training dataroot not found: {root}")

    if infer_single_label_from_root(root) is not None:
        return dataset_folder(opt, root)
    if is_dfdc_train_faces_root(root, opt):
        return dataset_folder(opt, root)

    if len(opt.classes) == 0:
        classes = [
            cls for cls in os.listdir(root)
            if not cls.startswith('.') and os.path.isdir(os.path.join(root, cls))
        ]
    else:
        classes = opt.classes

    class_set = set(classes)
    has_standard_binary = {'0_real', '1_fake'}.issubset(class_set)
    has_reversed_binary = {'0_fake', '1_real'}.issubset(class_set)
    has_raw_synthesis = {'raw', 'synthesis'}.issubset(class_set)
    has_real_fake = {'real', 'fake'}.issubset(class_set)

    if not (has_standard_binary or has_reversed_binary or has_raw_synthesis or has_real_fake):
        dset_lst = []
        for cls in classes:
            sub_root = os.path.join(root, cls)
            if not os.path.isdir(sub_root):
                continue
            dset = dataset_folder(opt, sub_root)
            dset_lst.append(dset)
        if not dset_lst:
            # Fallback for flat-labeled folders or alternative binary aliases
            # handled inside data/datasets.py binary_dataset.
            return dataset_folder(opt, root)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, root)


def _coerce_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _get_first_opt(opt, names, default=None):
    for name in names:
        if hasattr(opt, name):
            value = getattr(opt, name)
            if value is not None:
                return value
    return default


class DynamicExtraSamplingDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for: main dataset + per-epoch sampled subset from external pools.

    Typical use:
      - main: training_data_final
      - extra: FFPP/CelebDF/DFDC pools
      - each epoch: sample len(main) items from extra (optionally 1:1 real/fake)
    """

    def __init__(
        self,
        main_dataset,
        extra_dataset,
        *,
        match_main_count=True,
        balance_binary=True,
        seed=0,
        log_first_n_epochs=0,
    ):
        self.main_dataset = main_dataset
        self.extra_dataset = extra_dataset
        self.main_len = int(len(main_dataset))
        self.extra_len = int(len(extra_dataset))
        self.match_main_count = bool(match_main_count)
        self.balance_binary = bool(balance_binary)
        self.seed = int(seed)
        self.log_first_n_epochs = max(0, int(log_first_n_epochs))
        self.current_epoch = 0

        self.sample_extra_count = self.main_len if self.match_main_count else self.extra_len
        self._all_extra_indices = torch.arange(self.extra_len, dtype=torch.long)
        self._sampled_extra_indices = torch.empty(0, dtype=torch.long)
        self._sampled_extra_targets = []

        try:
            self._main_targets = [int(t) for t in _collect_targets(self.main_dataset)]
        except Exception:
            self._main_targets = None
        try:
            self._extra_targets = [int(t) for t in _collect_targets(self.extra_dataset)]
        except Exception:
            self._extra_targets = None

        self._extra_has_binary_targets = False
        self._extra_real_pool = torch.empty(0, dtype=torch.long)
        self._extra_fake_pool = torch.empty(0, dtype=torch.long)
        if self._extra_targets is not None and len(self._extra_targets) == self.extra_len:
            real_idx = [i for i, t in enumerate(self._extra_targets) if int(t) == 0]
            fake_idx = [i for i, t in enumerate(self._extra_targets) if int(t) == 1]
            if real_idx and fake_idx:
                self._extra_has_binary_targets = True
                self._extra_real_pool = torch.as_tensor(real_idx, dtype=torch.long)
                self._extra_fake_pool = torch.as_tensor(fake_idx, dtype=torch.long)

        self.set_epoch(0)

    @staticmethod
    def _sample_pool(pool, n, generator):
        if n <= 0:
            return torch.empty(0, dtype=torch.long)
        m = int(pool.numel())
        if m <= 0:
            return torch.empty(0, dtype=torch.long)
        if m >= n:
            perm = torch.randperm(m, generator=generator)[:n]
            return pool[perm]
        choice = torch.randint(low=0, high=m, size=(n,), generator=generator)
        return pool[choice]

    def _sample_extra_indices_for_epoch(self, epoch):
        if self.extra_len <= 0 or self.sample_extra_count <= 0:
            return torch.empty(0, dtype=torch.long)

        g = torch.Generator()
        g.manual_seed(self.seed + int(epoch))
        n = int(self.sample_extra_count)

        if self.balance_binary and self._extra_has_binary_targets:
            n_real = n // 2
            n_fake = n - n_real
            real_pick = self._sample_pool(self._extra_real_pool, n_real, g)
            fake_pick = self._sample_pool(self._extra_fake_pool, n_fake, g)
            sampled = torch.cat([real_pick, fake_pick], dim=0)
            if sampled.numel() > 1:
                order = torch.randperm(sampled.numel(), generator=g)
                sampled = sampled[order]
            return sampled

        if self.extra_len >= n:
            perm = torch.randperm(self.extra_len, generator=g)[:n]
            return self._all_extra_indices[perm]
        choice = torch.randint(low=0, high=self.extra_len, size=(n,), generator=g)
        return self._all_extra_indices[choice]

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self._sampled_extra_indices = self._sample_extra_indices_for_epoch(self.current_epoch)

        if self._extra_targets is not None and len(self._extra_targets) == self.extra_len:
            self._sampled_extra_targets = [
                int(self._extra_targets[int(i)]) for i in self._sampled_extra_indices.tolist()
            ]
        else:
            self._sampled_extra_targets = []

        if self.log_first_n_epochs > 0 and self.current_epoch < self.log_first_n_epochs:
            n_main = self.main_len
            n_extra = int(self._sampled_extra_indices.numel())
            if self._sampled_extra_targets:
                extra_real = sum(1 for t in self._sampled_extra_targets if t == 0)
                extra_fake = n_extra - extra_real
                print(
                    f"[DataMix] epoch={self.current_epoch} main={n_main}, "
                    f"sampled_extra={n_extra} (real={extra_real}, fake={extra_fake})"
                )
            else:
                print(
                    f"[DataMix] epoch={self.current_epoch} main={n_main}, "
                    f"sampled_extra={n_extra}"
                )

    @property
    def targets(self):
        if self._main_targets is None or self._extra_targets is None:
            raise AttributeError("DynamicExtraSamplingDataset has no stable targets metadata")
        return list(self._main_targets) + list(self._sampled_extra_targets)

    def __len__(self):
        return self.main_len + int(self._sampled_extra_indices.numel())

    def __getitem__(self, index):
        idx = int(index)
        if idx < self.main_len:
            return self.main_dataset[idx]
        extra_idx = int(self._sampled_extra_indices[idx - self.main_len].item())
        return self.extra_dataset[extra_idx]


def get_dataset(opt):
    import os

    extra_roots = getattr(opt, "extra_train_roots", None) or []
    if isinstance(extra_roots, str):
        extra_roots = [p.strip() for p in extra_roots.split(",") if p.strip()]

    dynamic_sampling_enabled = _coerce_bool(
        _get_first_opt(
            opt,
            [
                "dynamic_extra_sampling_enable",
                "data_dynamic_extra_sampling_enable",
            ],
            False,
        ),
        default=False,
    )
    dynamic_match_main_count = _coerce_bool(
        _get_first_opt(
            opt,
            [
                "dynamic_extra_sampling_match_main_count",
                "data_dynamic_extra_sampling_match_main_count",
            ],
            True,
        ),
        default=True,
    )
    dynamic_balance_binary = _coerce_bool(
        _get_first_opt(
            opt,
            [
                "dynamic_extra_sampling_balance_binary",
                "data_dynamic_extra_sampling_balance_binary",
            ],
            True,
        ),
        default=True,
    )
    dynamic_seed = int(
        _get_first_opt(
            opt,
            [
                "dynamic_extra_sampling_seed",
                "data_dynamic_extra_sampling_seed",
                "seed",
            ],
            0,
        )
    )
    dynamic_log_first_n_epochs = int(
        _get_first_opt(
            opt,
            [
                "dynamic_extra_sampling_log_first_n_epochs",
                "data_dynamic_extra_sampling_log_first_n_epochs",
            ],
            0,
        )
    )

    if bool(getattr(opt, "isTrain", False)) and dynamic_sampling_enabled and extra_roots:
        main_root = os.path.abspath(opt.dataroot)
        main_dataset = _build_dataset_from_root(opt, main_root)

        unique_extra_roots = []
        seen_extra = set()
        for r in extra_roots:
            if not r:
                continue
            r_abs = os.path.abspath(r)
            if r_abs == main_root:
                continue
            if r_abs in seen_extra:
                continue
            seen_extra.add(r_abs)
            unique_extra_roots.append(r_abs)

        extra_datasets = [_build_dataset_from_root(opt, root) for root in unique_extra_roots]
        if not extra_datasets:
            return main_dataset

        extra_dataset = extra_datasets[0]
        if len(extra_datasets) > 1:
            extra_dataset = torch.utils.data.ConcatDataset(extra_datasets)

        mixed_dataset = DynamicExtraSamplingDataset(
            main_dataset,
            extra_dataset,
            match_main_count=dynamic_match_main_count,
            balance_binary=dynamic_balance_binary,
            seed=dynamic_seed,
            log_first_n_epochs=dynamic_log_first_n_epochs,
        )
        return mixed_dataset

    roots = [opt.dataroot]
    roots.extend(extra_roots)

    unique_roots = []
    seen = set()
    for r in roots:
        if not r:
            continue
        r_abs = os.path.abspath(r)
        if r_abs in seen:
            continue
        seen.add(r_abs)
        unique_roots.append(r_abs)

    datasets_all = []
    for root in unique_roots:
        dset = _build_dataset_from_root(opt, root)
        datasets_all.append(dset)

    if len(datasets_all) == 1:
        return datasets_all[0]
    return torch.utils.data.ConcatDataset(datasets_all)

def _collect_targets(dataset):
    targets = getattr(dataset, 'targets', None)
    if targets is not None:
        return [int(t) for t in targets]
    if hasattr(dataset, 'datasets'):
        merged = []
        for d in dataset.datasets:
            merged.extend(_collect_targets(d))
        return merged
    raise AttributeError('Cannot build balanced sampler: dataset has no targets attribute')


class DistributedWeightedSampler(Sampler):
    """DDP-friendly class-balanced sampler using weighted multinomial sampling."""
    def __init__(self, sample_weights, num_samples, num_replicas, rank, seed=0):
        self.weights = torch.as_tensor(sample_weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=g,
        )
        rank_samples = sampled[self.rank:self.total_size:self.num_replicas]
        return iter(rank_samples.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class ClassUniformSampler(Sampler):
    """Class-uniform sampler: each epoch enforces near-equal samples per class."""
    def __init__(self, class_to_indices, num_samples, seed=0):
        if not class_to_indices:
            raise ValueError("class_to_indices must not be empty")
        self.class_to_indices = {
            int(k): torch.as_tensor(v, dtype=torch.long)
            for k, v in class_to_indices.items()
            if len(v) > 0
        }
        if not self.class_to_indices:
            raise ValueError("class_to_indices has no non-empty classes")
        self.classes = sorted(self.class_to_indices.keys())
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    @staticmethod
    def _draw(pool, n, generator):
        if n <= 0:
            return torch.empty(0, dtype=torch.long)
        m = int(pool.numel())
        if m <= 0:
            raise ValueError("Cannot draw from an empty class pool")
        if m >= n:
            perm = torch.randperm(m, generator=generator)[:n]
            return pool[perm]
        idx = torch.randint(low=0, high=m, size=(n,), generator=generator)
        return pool[idx]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        num_classes = len(self.classes)
        base = self.num_samples // num_classes
        rem = self.num_samples % num_classes

        chunks = []
        for i, cls in enumerate(self.classes):
            need = base + (1 if i < rem else 0)
            chunks.append(self._draw(self.class_to_indices[cls], need, g))
        sampled = torch.cat(chunks, dim=0)
        order = torch.randperm(sampled.numel(), generator=g)
        sampled = sampled[order]
        return iter(sampled.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class DistributedClassUniformSampler(Sampler):
    """DDP-friendly class-uniform sampler with rank-wise sharding."""
    def __init__(self, class_to_indices, num_samples, num_replicas, rank, seed=0):
        if not class_to_indices:
            raise ValueError("class_to_indices must not be empty")
        self.class_to_indices = {
            int(k): torch.as_tensor(v, dtype=torch.long)
            for k, v in class_to_indices.items()
            if len(v) > 0
        }
        if not self.class_to_indices:
            raise ValueError("class_to_indices has no non-empty classes")
        self.classes = sorted(self.class_to_indices.keys())
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _draw(pool, n, generator):
        if n <= 0:
            return torch.empty(0, dtype=torch.long)
        m = int(pool.numel())
        if m <= 0:
            raise ValueError("Cannot draw from an empty class pool")
        if m >= n:
            perm = torch.randperm(m, generator=generator)[:n]
            return pool[perm]
        idx = torch.randint(low=0, high=m, size=(n,), generator=generator)
        return pool[idx]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        num_classes = len(self.classes)
        base = self.total_size // num_classes
        rem = self.total_size % num_classes

        chunks = []
        for i, cls in enumerate(self.classes):
            need = base + (1 if i < rem else 0)
            chunks.append(self._draw(self.class_to_indices[cls], need, g))
        sampled = torch.cat(chunks, dim=0)
        order = torch.randperm(sampled.numel(), generator=g)
        sampled = sampled[order]

        rank_samples = sampled[self.rank:self.total_size:self.num_replicas]
        return iter(rank_samples.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _collect_class_to_indices(targets):
    class_to_indices = {}
    for idx, t in enumerate(targets):
        cls = int(t)
        class_to_indices.setdefault(cls, []).append(idx)
    return class_to_indices


def get_bal_sampler(dataset, *, distributed=False, num_replicas=1, rank=0, seed=0, strategy="auto"):
    targets = _collect_targets(dataset)
    if len(targets) == 0:
        raise ValueError('Cannot build balanced sampler: empty targets')

    strategy = str(strategy or "auto").strip().lower()
    class_to_indices = _collect_class_to_indices(targets)

    if strategy == "auto":
        # For binary deepfake labels (0=real, 1=fake), enforce strict near-1:1
        # class balance per epoch by default.
        if len(class_to_indices) == 2 and set(class_to_indices.keys()) == {0, 1}:
            strategy = "uniform"
        else:
            strategy = "weighted"

    if strategy in {"uniform", "strict", "strict_binary", "exact"}:
        if distributed:
            num_samples = int(math.ceil(len(targets) / float(max(1, num_replicas))))
            return DistributedClassUniformSampler(
                class_to_indices=class_to_indices,
                num_samples=num_samples,
                num_replicas=max(1, num_replicas),
                rank=rank,
                seed=seed,
            )
        return ClassUniformSampler(
            class_to_indices=class_to_indices,
            num_samples=len(targets),
            seed=seed,
        )

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    if distributed:
        num_samples = int(math.ceil(len(sample_weights) / float(max(1, num_replicas))))
        return DistributedWeightedSampler(
            sample_weights=sample_weights,
            num_samples=num_samples,
            num_replicas=max(1, num_replicas),
            rank=rank,
            seed=seed,
        )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    distributed = bool(getattr(opt, 'distributed', False))
    if opt.class_bal:
        strategy = str(getattr(opt, 'class_bal_strategy', 'auto') or 'auto')
        if bool(getattr(opt, 'class_bal_strict', False)) and strategy.lower() in {'weighted', 'auto'}:
            strategy = 'uniform'
        sampler = get_bal_sampler(
            dataset,
            distributed=distributed,
            num_replicas=int(getattr(opt, 'world_size', 1)),
            rank=int(getattr(opt, 'rank', 0)),
            seed=int(getattr(opt, 'seed', 0)),
            strategy=strategy,
        )
        shuffle = False
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
