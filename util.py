import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)


class ConceptSparsityRecorder:
    """Accumulates kept-cluster statistics for concept sparsification runs."""

    def __init__(
        self,
        output_path: Optional[str],
        *,
        dataset: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        accumulate_runs: bool = True,
    ) -> None:
        self._path = Path(output_path).expanduser().resolve() if output_path else None
        self._records: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._accumulate = accumulate_runs
        if dataset:
            self._meta["dataroot"] = dataset
        if extra_meta:
            self._meta.update(extra_meta)

    def enabled(self) -> bool:
        return self._path is not None

    def update(self, key: str, ratio: float, keep_mask: Optional[torch.Tensor]) -> None:
        if not self.enabled() or keep_mask is None:
            return
        entry = self._records.setdefault(key, {"ratio": ratio, "counts": None, "samples": 0})
        counts = keep_mask.to(torch.int64).sum(dim=0).detach().cpu()
        if entry["counts"] is None:
            entry["counts"] = counts.clone()
        else:
            entry["counts"] += counts
        entry["samples"] += keep_mask.shape[0]

    def finalize(self, *, total_samples: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not self.enabled() or not self._records:
            return None

        run_entry: Dict[str, Any] = {
            "metadata": dict(self._meta),
            "ratios": [],
        }
        if total_samples is not None:
            run_entry["metadata"]["total_samples"] = int(total_samples)

        def _sort_key(item: tuple[str, Dict[str, Any]]) -> float:
            return item[1]["ratio"]

        for key, entry in sorted(self._records.items(), key=_sort_key):
            counts_tensor: Optional[torch.Tensor] = entry.get("counts")
            if counts_tensor is None:
                continue
            counts_list = counts_tensor.tolist()
            order = [idx for idx, val in sorted(
                enumerate(counts_list), key=lambda pair: pair[1], reverse=True
            ) if val > 0]
            kept_total = int(sum(counts_list))
            ratio_payload = {
                "ratio": float(entry["ratio"]),
                "samples": int(entry["samples"]),
                "kept_total": kept_total,
                "avg_kept_per_sample": float(kept_total / entry["samples"])
                if entry["samples"]
                else 0.0,
                "clusters": [
                    {"cluster_id": int(idx), "kept": int(counts_list[idx])}
                    for idx in order
                ],
            }
            run_entry["ratios"].append(ratio_payload)

        payload: Any
        if self._accumulate:
            runs = []
            if self._path.is_file():
                try:
                    existing = json.loads(self._path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and "runs" in existing and isinstance(existing["runs"], list):
                        runs = existing["runs"]
                    elif isinstance(existing, list):
                        runs = existing
                    elif isinstance(existing, dict) and existing:
                        runs = [existing]
                except json.JSONDecodeError:
                    runs = []
            runs.append(run_entry)
            payload = {"runs": runs}
        else:
            payload = run_entry

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return run_entry
