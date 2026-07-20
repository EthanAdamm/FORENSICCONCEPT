import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from config_loader import ConfigLoader  # noqa: E402


class ReleaseSmokeTests(unittest.TestCase):
    def test_project_license_is_present(self):
        license_text = (RELEASE_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("ForensicConcept Authors", license_text)

    def test_public_configs_load(self):
        configs = sorted((RELEASE_ROOT / "configs").glob("*.yaml"))
        self.assertEqual(len(configs), 4)
        for config_path in configs:
            with self.subTest(config=config_path.name):
                opt = ConfigLoader(str(config_path)).to_namespace(is_train=True)
                self.assertIn(opt.trainmode, {"dinov3", "lora"})
                self.assertEqual(opt.cropSize, 224)
                self.assertEqual(opt.loadSize, 224)
                self.assertEqual(opt.num_classes, 1)

    def test_released_codebook(self):
        codebook = np.load(
            RELEASE_ROOT / "assets" / "codebooks" / "cleandift_codebook.npy",
            allow_pickle=False,
        )
        self.assertEqual(codebook.shape, (200, 1280))
        self.assertEqual(codebook.dtype, np.float32)
        self.assertTrue(np.isfinite(codebook).all())

    def test_albumentations_downscale_compatibility(self):
        from data.datasets import _build_downscale_aug

        transform = _build_downscale_aug()
        self.assertIsNotNone(transform)

    def test_concept_mapping_bias_is_honored(self):
        import torch.nn as nn

        from networks.dinov3_detector import Dinov3Config, Dinov3Detector

        with tempfile.TemporaryDirectory() as tmp_dir:
            matrix_path = Path(tmp_dir) / "concepts.npy"
            np.save(matrix_path, np.ones((3, 4), dtype=np.float32))
            detector = Dinov3Detector.__new__(Dinov3Detector)
            nn.Module.__init__(detector)
            detector.config = Dinov3Config(
                use_concept_head=True,
                concept_matrix_path=str(matrix_path),
                concept_mapping_bias=True,
                concept_mapping_trainable=True,
            )
            detector._init_concept_modules(embed_dim=4, num_classes=1)
            self.assertIsNotNone(detector.concept_mapping.bias)


if __name__ == "__main__":
    unittest.main()
