import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from data import create_dataloader
from util import ConceptSparsityRecorder


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, reverse: bool) -> dict:
    if reverse:
        real_mask = y_true == 1
        fake_mask = y_true == 0
    else:
        real_mask = y_true == 0
        fake_mask = y_true == 1

    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob.size > 0 else 0.5
    except ValueError:
        # roc_auc_score is undefined when only one class is present in y_true.
        auc = 0.5

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "auc": float(auc),
        "ap": average_precision_score(y_true, y_prob) if y_prob.size > 0 else 0.0,
        "r_acc": accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.any() else 0.0,
        "f_acc": accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.any() else 0.0,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def _format_ratio_key(ratio: float) -> str:
    ratio_str = f"{ratio:.6f}".rstrip("0").rstrip(".")
    return ratio_str if ratio_str else "0"


def _extract_dataset_paths(dataset):
    """Return a flat list of file paths for ImageFolder-style datasets, or None."""
    if isinstance(dataset, Subset):
        base_paths = _extract_dataset_paths(dataset.dataset)
        if base_paths is None:
            return None
        return [base_paths[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        paths = []
        for sub in dataset.datasets:
            sub_paths = _extract_dataset_paths(sub)
            if sub_paths is None:
                return None
            paths.extend(sub_paths)
        return paths
    if hasattr(dataset, "samples"):
        return [path for path, _ in dataset.samples]
    if hasattr(dataset, "imgs"):
        return [path for path, _ in dataset.imgs]
    return None


def _get_patch_size_from_model(model) -> int:
    """Best-effort extraction of patch size from CLIP/ViT style models."""
    visual = getattr(model, "visual", None)
    if visual is None and hasattr(model, "model"):
        visual = getattr(model.model, "visual", None)
    if visual is None:
        return 0
    patch_size = getattr(visual, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    if patch_size:
        return int(patch_size)
    conv1 = getattr(visual, "conv1", None)
    if conv1 is not None and hasattr(conv1, "kernel_size"):
        ks = conv1.kernel_size
        return int(ks[0]) if isinstance(ks, (tuple, list)) else int(ks)
    return 0


def _infer_grid_size(num_patches: int, height: int, width: int, patch_size: int):
    """Infer (grid_h, grid_w) for patch tokens."""
    if patch_size and height % patch_size == 0 and width % patch_size == 0:
        grid_h = height // patch_size
        grid_w = width // patch_size
        if grid_h * grid_w == num_patches:
            return grid_h, grid_w
    if num_patches:
        side = int(math.sqrt(num_patches))
        if side * side == num_patches:
            return side, side
    if patch_size and num_patches and width:
        grid_w = max(1, int(round(width / patch_size)))
        grid_h = max(1, int(math.ceil(num_patches / grid_w)))
        return grid_h, grid_w
    return 0, 0


def validate(model, opt):
    print(
        f"[validate] dataroot={getattr(opt, 'dataroot', None)} "
        f"data_aug={getattr(opt, 'data_aug', None)} "
        f"isTrain={getattr(opt, 'isTrain', None)}"
    )
    data_loader = create_dataloader(opt)

    # Keep evaluation on the same device as the model to avoid cross-GPU tensor mixups.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    num_classes = getattr(opt, "num_classes", 1)
    reverse = getattr(opt, "reverse", False)
    try:
        max_eval_samples = int(getattr(opt, "max_eval_samples", 0) or 0)
    except (TypeError, ValueError):
        max_eval_samples = 0


    eval_ratios = (
        getattr(opt, "clip_concept_eval_sparsity", None)
        or getattr(opt, "dinov3_concept_eval_sparsity", None)
        or []
    )
    sparsity_recorder = ConceptSparsityRecorder(
        getattr(opt, "clip_concept_eval_log_path", None)
        or getattr(opt, "dinov3_concept_eval_log_path", None),
        dataset=getattr(opt, "dataroot", None),
        extra_meta={
            "mode": "validate",
            "experiment": getattr(opt, "name", None),
        },
    )

    coords_path_raw = getattr(opt, "codebook_coords_path", None)
    dataset_name = getattr(opt, "dataset_name", None) or Path(getattr(opt, "dataroot", "dataset")).name or "dataset"
    if coords_path_raw and "{dataset}" in str(coords_path_raw):
        coords_path_raw = str(coords_path_raw).replace("{dataset}", dataset_name)
    coords_limit = getattr(opt, "codebook_coords_limit", None)
    try:
        coords_limit = int(coords_limit) if coords_limit is not None else None
    except (TypeError, ValueError):
        coords_limit = None

    coords_path = Path(coords_path_raw).expanduser() if coords_path_raw else None
    coords_mode = None
    coords_base_dir = None
    if coords_path:
        suffix = coords_path.suffix.lower()
        if suffix in {".dat", ".jsonl"}:
            coords_mode = "dat" if suffix == ".dat" else "jsonl"
        else:
            coords_base_dir = coords_path
            coords_mode = "dat"
    coords_fp = None
    coords_bin = None
    indices_bin = None
    scores_bin = None
    concepts_bin = None
    paths_fp = None
    sample_paths = None
    dumped_labels = []
    dumped_samples = 0
    meta_template = {}
    if coords_path:
        if coords_base_dir is not None:
            coords_path = coords_base_dir / dataset_name / (
                "top_patch_coords.dat" if coords_mode == "dat" else "coords.jsonl"
            )
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        sample_paths = _extract_dataset_paths(data_loader.dataset)
        if coords_mode == "jsonl":
            coords_fp = coords_path.open("w", encoding="utf-8")
            print(f"[validate] dumping codebook coords to {coords_path}")
        else:
            coords_bin = coords_path.open("wb")
            indices_bin = coords_path.with_name("top_patch_indices.dat").open("wb")
            scores_bin = coords_path.with_name("top_patch_scores.dat").open("wb")
            concepts_bin = coords_path.with_name("top_patch_concept_ids.dat").open("wb")
            paths_fp = coords_path.with_name("paths.txt").open("w", encoding="utf-8")
            print(f"[validate] dumping codebook coords to {coords_path} (dat format)")

    sample_offset = 0
    with torch.no_grad():
        y_true: list = []
        head_probs: defaultdict = defaultdict(list)
        head_preds: defaultdict = defaultdict(list)
        for img, label in tqdm(data_loader, desc="Testing", leave=False):
            if max_eval_samples > 0 and sample_offset >= max_eval_samples:
                break
            if max_eval_samples > 0:
                remain = max_eval_samples - sample_offset
                if remain <= 0:
                    break
                if img.shape[0] > remain:
                    img = img[:remain]
                    label = label[:remain]
            in_tens = img.to(model_device, non_blocking=True)
            if coords_path is not None:
                try:
                    raw_output = model(in_tens, return_debug=True)
                except TypeError:
                    raw_output = model(in_tens)
            else:
                raw_output = model(in_tens)

            head_outputs = {}
            concept_raw = None
            if isinstance(raw_output, dict):
                if raw_output.get("logits") is not None:
                    head_outputs["main"] = raw_output["logits"]
                if raw_output.get("cb_only") is not None:
                    head_outputs["cb_only"] = raw_output["cb_only"]
                if raw_output.get("concept_logits") is not None:
                    head_outputs["concept"] = raw_output["concept_logits"]
                if raw_output.get("token_logits") is not None:
                    head_outputs["token"] = raw_output["token_logits"]
                concept_raw = raw_output.get("concept_features_raw")
            elif isinstance(raw_output, tuple):
                head_outputs["main"] = raw_output[0]
            else:
                head_outputs["main"] = raw_output

            if concept_raw is not None and eval_ratios:
                for ratio in eval_ratios:
                    try:
                        ratio_val = float(ratio)
                    except (TypeError, ValueError):
                        continue
                    key = f"concept_s_{_format_ratio_key(ratio_val)}"
                    if sparsity_recorder.enabled():
                        logits_variant, keep_mask = model.compute_concept_logits_with_ratio(
                            concept_raw, ratio_val, return_mask=True
                        )
                        sparsity_recorder.update(key, ratio_val, keep_mask)
                    else:
                        logits_variant = model.compute_concept_logits_with_ratio(concept_raw, ratio_val)
                    head_outputs[key] = logits_variant

            labels_cpu = label.flatten().cpu().tolist()
            y_true.extend(labels_cpu)

            for head_name, logits in head_outputs.items():
                logits_cpu = logits.detach().cpu()
                if logits_cpu.ndim == 1:
                    logits_cpu = logits_cpu.unsqueeze(1)
                if num_classes == 1 or logits_cpu.shape[1] == 1:
                    probs_fake = torch.sigmoid(logits_cpu).flatten()
                    preds = (probs_fake >= 0.5).long()
                    head_probs[head_name].extend(probs_fake.tolist())
                    head_preds[head_name].extend(preds.tolist())
                else:
                    probs = torch.softmax(logits_cpu, dim=1)
                    probs_fake = probs[:, 1]
                    preds = torch.argmax(probs, dim=1)
                    head_probs[head_name].extend(probs_fake.tolist())
                    head_preds[head_name].extend(preds.tolist())

            if coords_path is not None and isinstance(raw_output, dict):
                topk_idx = raw_output.get("codebook_topk_idx")
                if topk_idx is not None:
                    num_patches = None
                    tokens = raw_output.get("tokens")
                    if tokens is not None and hasattr(tokens, "shape"):
                        num_patches = int(tokens.shape[1])
                        if "embedding_dim" not in meta_template:
                            meta_template["embedding_dim"] = int(tokens.shape[-1])
                    if num_patches is None:
                        num_patches = int(getattr(model, "num_patches", 0) or 0)
                    if not num_patches:
                        num_patches = int(getattr(opt, "clip_num_patches", 0) or 0)
                    if not num_patches:
                        num_patches = int(topk_idx.max().item() + 1)

                    height, width = int(in_tens.shape[-2]), int(in_tens.shape[-1])
                    patch_size = _get_patch_size_from_model(model)
                    grid_h, grid_w = _infer_grid_size(num_patches, height, width, patch_size)

                    if coords_mode == "jsonl":
                        topk_idx_cpu = topk_idx.detach().cpu().tolist()
                        for b, idx_list in enumerate(topk_idx_cpu):
                            global_idx = sample_offset + b
                            if coords_limit is not None and global_idx >= coords_limit:
                                continue
                            path = None
                            if sample_paths is not None and global_idx < len(sample_paths):
                                path = sample_paths[global_idx]
                            coords_rc = None
                            coords_xyxy = None
                            if grid_w and grid_h:
                                coords_rc = []
                                coords_xyxy = []
                                for idx in idx_list:
                                    row = int(idx) // grid_w
                                    col = int(idx) % grid_w
                                    coords_rc.append([row, col])
                                    if patch_size:
                                        x0 = col * patch_size
                                        y0 = row * patch_size
                                        coords_xyxy.append([x0, y0, x0 + patch_size, y0 + patch_size])
                            entry = {
                                "index": int(global_idx),
                                "path": path,
                                "label": int(labels_cpu[b]) if b < len(labels_cpu) else None,
                                "topk_idx": [int(i) for i in idx_list],
                                "coords_rc": coords_rc,
                                "coords_xyxy": coords_xyxy if coords_xyxy else None,
                                "grid_size": [int(grid_h), int(grid_w)] if grid_h and grid_w else None,
                                "patch_size": int(patch_size) if patch_size else None,
                                "image_size": [height, width],
                            }
                            coords_fp.write(json.dumps(entry, ensure_ascii=True) + "\n")
                    else:
                        idx_np = topk_idx.detach().cpu().to(torch.int32).numpy()
                        if idx_np.ndim == 1:
                            idx_np = idx_np.reshape(-1, 1)
                        top_k = idx_np.shape[1]
                        scores = raw_output.get("codebook_score")
                        topk_concept = raw_output.get("codebook_topk_concept")
                        if scores is not None:
                            score_np = scores.detach().cpu().numpy()
                            score_sel = np.take_along_axis(score_np, idx_np.astype(np.int64), axis=1).astype(np.float32)
                        else:
                            score_sel = np.zeros_like(idx_np, dtype=np.float32)
                        if topk_concept is not None:
                            concept_sel = topk_concept.detach().cpu().to(torch.int32).numpy()
                        else:
                            concept_sel = np.zeros_like(idx_np, dtype=np.int32)

                        if grid_w and grid_h:
                            row = idx_np // grid_w
                            col = idx_np % grid_w
                            coords_packed = (row.astype(np.int32) << 16) | col.astype(np.int32)
                        else:
                            coords_packed = idx_np.astype(np.int32)

                        for b in range(idx_np.shape[0]):
                            global_idx = sample_offset + b
                            if coords_limit is not None and global_idx >= coords_limit:
                                continue
                            coords_packed[b].tofile(coords_bin)
                            idx_np[b].tofile(indices_bin)
                            score_sel[b].tofile(scores_bin)
                            if concepts_bin is not None:
                                concept_sel[b].reshape(-1).tofile(concepts_bin)
                            if paths_fp is not None and sample_paths is not None and global_idx < len(sample_paths):
                                paths_fp.write(f"{sample_paths[global_idx]}\n")
                            dumped_labels.append(int(labels_cpu[b]) if b < len(labels_cpu) else 0)
                            dumped_samples += 1
                        if "num_patches" not in meta_template:
                            meta_template["num_patches"] = int(num_patches)
                        if "top_k" not in meta_template:
                            meta_template["top_k"] = int(top_k)
                        if "top_r" not in meta_template and concept_sel is not None and concept_sel.ndim == 3:
                            meta_template["top_r"] = int(concept_sel.shape[-1])
                        if "grid_h" not in meta_template and grid_h and grid_w:
                            meta_template["grid_h"] = int(grid_h)
                            meta_template["grid_w"] = int(grid_w)
                        if "patch_size" not in meta_template and patch_size:
                            meta_template["patch_size"] = int(patch_size)

            sample_offset += len(labels_cpu)
    y_true_arr = np.array(y_true)
    metrics = {}
    for head_name, preds in head_preds.items():
        y_pred_arr = np.array(preds)
        y_prob_arr = np.array(head_probs[head_name])
        metrics[head_name] = _compute_binary_metrics(y_true_arr, y_pred_arr, y_prob_arr, reverse)

    sparsity_recorder.finalize(total_samples=len(y_true))

    if coords_fp is not None:
        coords_fp.close()
    if coords_mode == "dat":
        if coords_bin is not None:
            coords_bin.close()
        if indices_bin is not None:
            indices_bin.close()
        if scores_bin is not None:
            scores_bin.close()
        if concepts_bin is not None:
            concepts_bin.close()
        paths_path = coords_path.with_name("paths.txt")
        labels_path = coords_path.with_name("labels.npy")
        meta_path = coords_path.with_name("meta.json")
        if paths_fp is not None:
            paths_fp.close()
        if dumped_labels:
            np.save(labels_path, np.array(dumped_labels, dtype=np.int64))
        patch_grid = None
        if meta_template.get("grid_h") and meta_template.get("grid_w"):
            patch_grid = [meta_template["grid_h"], meta_template["grid_w"]]
        patch_size = meta_template.get("patch_size")
        meta = {
            "top_patch_tokens.dat": None,
            "top_patch_scores.dat": "top_patch_scores.dat",
            "cls_tokens.dat": None,
            "top_patch_indices.dat": "top_patch_indices.dat",
            "top_patch_coords.dat": coords_path.name,
            "top_patch_concept_ids.dat": "top_patch_concept_ids.dat",
            "num_samples": dumped_samples,
            "num_patches": meta_template.get("num_patches"),
            "top_k": meta_template.get("top_k"),
            "top_r": meta_template.get("top_r"),
            "embedding_dim": meta_template.get("embedding_dim"),
            "projection": "post",
            "dtype": "float32",
            "labels.npy": labels_path.name,
            "paths.txt": paths_path.name,
            "patch_grid": patch_grid,
            "patch_size": [patch_size, patch_size] if patch_size else None,
        }
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)
    return metrics, y_true_arr
