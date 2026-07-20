"""Run evaluation on a trained checkpoint using the same configuration pipeline as training."""
from pathlib import Path
from typing import Optional, Tuple

import torch

from config_loader import create_argument_parser, load_config_from_args
from train_with_config import UnifiedTrainer


def _load_checkpoint_from_path(trainer: UnifiedTrainer, checkpoint: Path) -> str:
    """Load a checkpoint file directly by temporarily redirecting the save directory.

    Args:
        trainer: UnifiedTrainer instance with an initialized model.
        checkpoint: Path to the checkpoint file to load.

    Returns:
        The epoch tag inferred from the checkpoint name.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    epoch_tag = checkpoint.stem.replace("model_epoch_", "")
    base_model = trainer.model
    original_save_dir = base_model.save_dir
    original_load_dir = base_model.load_dir
    try:
        base_model.save_dir = str(checkpoint.parent)
        base_model.load_dir = str(checkpoint.parent)
        # Allow direct file loading even if directory names have been renamed.
        base_model.load_networks(str(checkpoint))
    finally:
        base_model.save_dir = original_save_dir
        base_model.load_dir = original_load_dir
    return epoch_tag


def _resolve_checkpoint_argument(checkpoint: Optional[str]) -> Optional[Path]:
    """Normalize checkpoint argument to a concrete file path.

    Supports passing a directory (picks the newest model_epoch_*.pth inside).
    """
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("model_epoch_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No model_epoch_*.pth found under {checkpoint_path}")
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[Checkpoint] No file specified, using latest in directory: {latest.name}")
        return latest
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def _resolve_config_path(args, default_config: str) -> Path:
    """Pick the config file to load, preferring the one next to the checkpoint."""
    requested_path = Path(args.config).expanduser() if args.config else Path(default_config)
    checkpoint_hint = getattr(args, "checkpoint_path", None)

    if checkpoint_hint and args.config == default_config:
        ckpt = Path(checkpoint_hint).expanduser()
        ckpt_dir = ckpt.parent if ckpt.is_file() else ckpt
        candidate_cfg = ckpt_dir / "config.yaml"
        if candidate_cfg.exists():
            print(f"[Config] Auto-using config next to checkpoint: {candidate_cfg}")
            return candidate_cfg

    if requested_path.exists():
        return requested_path

    if checkpoint_hint:
        ckpt = Path(checkpoint_hint).expanduser()
        ckpt_dir = ckpt.parent if ckpt.is_file() else ckpt
        fallback_cfg = ckpt_dir / "config.yaml"
        if fallback_cfg.exists():
            print(f"[Config] Requested config {requested_path} missing; falling back to {fallback_cfg}")
            return fallback_cfg

    raise FileNotFoundError(f"Configuration file not found: {requested_path}")


def _prepare_trainer(args) -> Tuple[UnifiedTrainer, Optional[str]]:
    """Create a UnifiedTrainer and ensure the desired checkpoint is loaded."""
    config_loader = load_config_from_args(args)

    checkpoint_path = _resolve_checkpoint_argument(getattr(args, "checkpoint_path", None))
    config_checkpoint = config_loader.get('testing.model_path', None)

    epoch_tag = getattr(args, "epoch", None)

    if not checkpoint_path and not epoch_tag and config_checkpoint:
        checkpoint_path = _resolve_checkpoint_argument(config_checkpoint)

    if checkpoint_path:
        # Directly load the specified checkpoint during initialization.
        config_loader.set('training.continue_train', True)
        config_loader.set('training.epoch', str(checkpoint_path))
        trainer = UnifiedTrainer(config_loader, is_train=False)
        epoch_tag = checkpoint_path.stem.replace("model_epoch_", "")
    else:
        if not epoch_tag:
            epoch_tag = config_loader.get('training.epoch', 'latest')
        # Ensure the trainer restores the requested checkpoint during initialization.
        config_loader.set('training.continue_train', True)
        config_loader.set('training.epoch', epoch_tag)
        trainer = UnifiedTrainer(config_loader, is_train=False)

    return trainer, epoch_tag


def main():
    parser = create_argument_parser()
    parser.add_argument(
        '--epoch',
        type=str,
        help='Checkpoint tag to load (matches the suffix in model_epoch_<tag>.pth).'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='Optional explicit path to a checkpoint file; overrides --epoch.'
    )
    default_config = parser.get_default('config')
    args = parser.parse_args()
    args.config = str(_resolve_config_path(args, default_config))
    trainer, epoch_tag = _prepare_trainer(args)
    trainer.model.eval()

    with torch.no_grad():
        mean_acc, mean_auc, mean_ap = trainer.test_model()

    if epoch_tag:
        print(f"Evaluated checkpoint: {epoch_tag}")
    print(
        f"Mean Accuracy: {mean_acc*100:.2f}%  |  "
        f"Mean AUC: {mean_auc*100:.2f}%  |  "
        f"Mean AP: {mean_ap*100:.2f}%"
    )


if __name__ == '__main__':
    main()
