import cv2
import numpy as np
import inspect
import re
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations import Downscale, Compose, ImageCompression, GaussNoise, MotionBlur, GaussianBlur
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
from PIL import ImageOps
from PIL import UnidentifiedImageError
from scipy.ndimage import gaussian_filter
from torchvision.transforms import InterpolationMode
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _supports_albu_arg(cls, arg_name):
    try:
        return arg_name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


def _build_downscale_aug():
    """Build Downscale across the Albumentations 1.x and 2.x APIs."""
    if _supports_albu_arg(Downscale, "scale_range"):
        return Downscale(
            scale_range=(0.25, 0.75),
            interpolation_pair={
                "downscale": cv2.INTER_LINEAR,
                "upscale": cv2.INTER_LINEAR,
            },
            p=0.5,
        )
    return Downscale(
        scale_min=0.25,
        scale_max=0.75,
        interpolation=cv2.INTER_LINEAR,
        p=0.5,
    )


def _build_testshift_like_aug(p_apply=1.0, max_groups=3):
    # group 1) blur
    blur_group = A.OneOf([
        A.GaussianBlur(sigma_limit=(0.1, 5.0), p=1.0),
        A.Defocus(radius=(1, 8), alias_blur=(0.1, 0.5), p=1.0),
    ], p=1.0)

    # group 2) color distortion
    color_group = A.OneOf([
        A.ChromaticAberration(
            primary_distortion_limit=(0.01, 0.08),
            secondary_distortion_limit=(0.01, 0.12),
            p=1.0,
        ),
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=1.0, hue=0.0, p=1.0),
    ], p=1.0)

    # group 3) jpeg
    if _supports_albu_arg(A.ImageCompression, "quality_range"):
        jpeg_group = A.ImageCompression(compression_type="jpeg", quality_range=(4, 43), p=1.0)
    else:
        jpeg_group = A.ImageCompression(quality_lower=4, quality_upper=43, p=1.0)

    # group 4) noise
    if _supports_albu_arg(A.GaussNoise, "std_range"):
        gauss_noise = A.GaussNoise(std_range=(0.032, 0.100), mean_range=(0.0, 0.0), per_channel=True, p=1.0)
    else:
        gauss_noise = A.GaussNoise(var_limit=(0.032 ** 2, 0.100 ** 2), mean=0.0, per_channel=True, p=1.0)
    if hasattr(A, "SaltAndPepper"):
        impulse_noise = A.SaltAndPepper(amount=(0.001, 0.03), salt_vs_pepper=(0.45, 0.55), p=1.0)
    else:
        impulse_noise = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
    noise_group = A.OneOf([
        gauss_noise,
        impulse_noise,
    ], p=1.0)

    # group 5) brightness
    brightness_group = A.RandomBrightnessContrast(
        brightness_limit=(-0.8, 0.8),
        contrast_limit=(0.0, 0.0),
        brightness_by_max=True,
        p=1.0,
    )

    # group 6) spatial distortion
    spatial_group = A.OneOf([
        A.GridDistortion(
            num_steps=5,
            distort_limit=(0.05, 0.5),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.Sequential([
            A.ToGray(num_output_channels=3, p=1.0),
            A.Posterize(num_bits=(3, 5), p=1.0),
        ], p=1.0),
    ], p=1.0)

    # group 7) sharpness / contrast
    sharp_contrast_group = A.RandomToneCurve(scale=0.2, p=1.0)

    groups = [
        blur_group,
        color_group,
        jpeg_group,
        noise_group,
        brightness_group,
        spatial_group,
        sharp_contrast_group,
    ]

    max_groups = max(1, min(int(max_groups), len(groups)))
    choices = [A.SomeOf(groups, n=n, replace=False, p=1.0) for n in range(1, max_groups + 1)]

    return A.Compose([
        A.OneOf(choices, p=1.0),
    ], p=float(p_apply))

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def _apply_single_robustness(img, robustness_type, params=None):
    params = dict(params or {})
    robustness_type = str(robustness_type or "").strip().lower()
    if not robustness_type:
        return img

    if robustness_type == "jpeg":
        quality = int(params.get("jpeg_quality", 75) or 75)
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality)
        out.seek(0)
        with Image.open(out) as aug_img:
            return aug_img.convert("RGB")

    if robustness_type == "gaussian_blur":
        sigma = float(params.get("gaussian_blur_sigma", 1.0) or 1.0)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    if robustness_type == "gaussian_noise":
        std = float(params.get("gaussian_noise_std", 0.1) or 0.1)
        img_np = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(loc=0.0, scale=std, size=img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0.0, 1.0)
        img_np = (img_np * 255.0).round().astype(np.uint8)
        return Image.fromarray(img_np)

    if robustness_type == "downsample":
        scale = float(params.get("downsample_scale", 0.5) or 0.5)
        width, height = img.size
        down_w = max(1, int(round(width * scale)))
        down_h = max(1, int(round(height * scale)))
        down = img.resize((down_w, down_h), resample=Image.BILINEAR)
        return down.resize((width, height), resample=Image.BILINEAR)

    if robustness_type == "color_jitter":
        brightness = float(params.get("cj_brightness", 0.0) or 0.0)
        contrast = float(params.get("cj_contrast", 0.0) or 0.0)
        saturation = float(params.get("cj_saturation", 0.0) or 0.0)
        hue = float(params.get("cj_hue", 0.0) or 0.0)

        out = img
        if brightness != 0.0:
            out = TF.adjust_brightness(out, 1.0 + brightness)
        if contrast != 0.0:
            out = TF.adjust_contrast(out, 1.0 + contrast)
        if saturation != 0.0:
            out = TF.adjust_saturation(out, 1.0 + saturation)
        if hue != 0.0:
            hue = max(-0.5, min(0.5, hue))
            out = TF.adjust_hue(out, hue)
        return out

    raise ValueError(f"Unsupported robustness_type: {robustness_type}")


def _apply_explicit_robustness(img, opt):
    robustness_chain = getattr(opt, "robustness_chain", None) or []
    if robustness_chain:
        out = img
        for idx, entry in enumerate(robustness_chain):
            if not isinstance(entry, dict):
                raise ValueError(f"robustness_chain[{idx}] must be a dict, got {type(entry).__name__}")
            out = _apply_single_robustness(
                out,
                entry.get("type", ""),
                entry.get("params", {}),
            )
        return out

    return _apply_single_robustness(
        img,
        getattr(opt, "robustness_type", ""),
        {
            "jpeg_quality": getattr(opt, "robustness_jpeg_quality", 75),
            "gaussian_blur_sigma": getattr(opt, "robustness_gaussian_blur_sigma", 1.0),
            "gaussian_noise_std": getattr(opt, "robustness_gaussian_noise_std", 0.1),
            "downsample_scale": getattr(opt, "robustness_downsample_scale", 0.5),
            "cj_brightness": getattr(opt, "robustness_cj_brightness", 0.0),
            "cj_contrast": getattr(opt, "robustness_cj_contrast", 0.0),
            "cj_saturation": getattr(opt, "robustness_cj_saturation", 0.0),
            "cj_hue": getattr(opt, "robustness_cj_hue", 0.0),
        },
    )


import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torchvision.transforms
from scipy.ndimage import gaussian_filter


class ImageProcessor:
    def __init__(self, opt):
        self.opt = opt

    def data_augment(self, img):
        noise_type = getattr(self.opt, "noise_type", None)
        noise_ratio = getattr(self.opt, "noise_ratio", None)

        if noise_type == 'jpg':
            img_processed = self.pil_jpg_new(img, noise_ratio)
        elif noise_type == 'resize':
            width, height = img.size
            img_processed = torchvision.transforms.Resize((int(height / noise_ratio), int(width / noise_ratio)))(img)
        elif noise_type == 'blur':
            img = np.array(img)
            self.gaussian_blur(img, noise_ratio)
            img_processed = Image.fromarray(img)
        elif noise_type is None:
            img_processed = img

        return img_processed

    def pil_jpg_new(self, img, compress_val):
        out = BytesIO()
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        img = Image.fromarray(img)
        out.close()
        return img

    def gaussian_blur(self, img, sigma):
        # Check if the image has a third dimension
        if img.ndim == 3:
            # Check the number of channels in the image
            channels = img.shape[2]
            # if channels == 0:
            #     # If there are no channels, return the image as is
            #     return img
            if channels == 1:
                # If there is one channel, apply the filter to the single channel
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
            elif channels == 2:
                # If there are two channels, apply the filter to both channels
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
                gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
            elif channels == 3:
                # If there are three channels, apply the filter to all three channels
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
                gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
                gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
        else:
            # If the image does not have a third dimension, apply the filter to the 2D image
            gaussian_filter(img, output=img, sigma=sigma)

        # return img


def encode_image(image_path, processor=None):
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)

        # Apply data augmentation if processor is provided
        if processor is not None:
            # print(processor.opt)
            img = processor.data_augment(img)

        # Convert image to Base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img, img_base64

class Options:
    def __init__(self, noise_type=None, noise_ratio=None):
        """
        初始化数据增强的配置参数。

        :param noise_type: 噪声类型，可选值为 'jpg', 'resize', 'blur' 或 None
        :param noise_ratio: 噪声强度或比例，具体含义取决于 noise_type
        """
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
    def __str__(self):
        """返回对象的字符串表示，便于调试"""
        return (f"Options(noise_type={self.noise_type}, noise_ratio={self.noise_ratio}")


# Robust ImageFolder: skip unreadable/corrupted files instead of crashing workers.
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, skip_broken=True, skip_log_limit=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_broken = bool(skip_broken)
        self.skip_log_limit = max(0, int(skip_log_limit))
        self._skip_log_count = 0

    @staticmethod
    def _is_image_decode_error(exc):
        return isinstance(exc, (UnidentifiedImageError, OSError, ValueError))

    def __getitem__(self, index):
        if not self.skip_broken:
            return super().__getitem__(index)

        total = len(self.samples)
        if total == 0:
            raise IndexError("SafeImageFolder is empty")

        idx = int(index) % total
        for _ in range(total):
            try:
                return super().__getitem__(idx)
            except Exception as exc:
                if not self._is_image_decode_error(exc):
                    raise

                bad_path = self.samples[idx][0]
                if self._skip_log_count < self.skip_log_limit:
                    print(
                        f"[Data] Skipping unreadable image: {bad_path} "
                        f"({type(exc).__name__}: {exc})"
                    )
                    self._skip_log_count += 1
                idx = (idx + 1) % total

        raise RuntimeError(
            "Unable to fetch a readable image from dataset. "
            "All candidates appear unreadable."
        )


class FlatLabeledImageDataset(torch.utils.data.Dataset):
    """Fallback dataset for flat folders: labels parsed from filename suffix (_real/_fake)."""

    def __init__(self, samples, transform=None, skip_broken=True, skip_log_limit=0):
        self.samples = list(samples)
        self.transform = transform
        self.skip_broken = bool(skip_broken)
        self.skip_log_limit = max(0, int(skip_log_limit))
        self._skip_log_count = 0
        self.targets = [int(t) for _, t in self.samples]
        self.class_to_idx = {'0_real': 0, '1_fake': 1}
        self.classes = ['0_real', '1_fake']
        self.imgs = self.samples

    @staticmethod
    def _is_image_decode_error(exc):
        return isinstance(exc, (UnidentifiedImageError, OSError, ValueError))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        total = len(self.samples)
        if total == 0:
            raise IndexError("FlatLabeledImageDataset is empty")

        idx = int(index) % total
        for _ in range(total):
            path, target = self.samples[idx]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform is not None:
                        img = self.transform(img)
                return img, int(target)
            except Exception as exc:
                if (not self.skip_broken) or (not self._is_image_decode_error(exc)):
                    raise
                if self._skip_log_count < self.skip_log_limit:
                    print(
                        f"[Data] Skipping unreadable image: {path} "
                        f"({type(exc).__name__}: {exc})"
                    )
                    self._skip_log_count += 1
                idx = (idx + 1) % total

        raise RuntimeError("Unable to fetch a readable image from flat dataset.")


# Example usage:
# opt = SomeOptionsClass()  # Replace with your actual options class
# print(opt_class)


def _find_binary_roots(root):
    """Yield directories whose immediate children contain 0_real and 1_fake."""
    binary_roots = []
    valid_pairs = (
        {'0_real', '1_fake'},
        {'0_fake', '1_real'},
        {'raw', 'synthesis'},     # deepfake datasets
        {'real', 'fake'},
    )
    for current_root, dirs, _ in os.walk(root):
        present = set(dirs)
        if any(pair.issubset(present) for pair in valid_pairs):
            binary_roots.append(current_root)
    # ensure deterministic ordering so label mapping is stable
    binary_roots.sort()
    return binary_roots


def infer_single_label_from_root(root):
    """Infer fixed binary label from directory name for one-class dataset roots."""
    # Canonical deepfake aliases used by Celeb-DF/FF++ style trees.
    alias_to_label = {
        # real
        "0_real": 0,
        "1_real": 0,
        "real": 0,
        "raw": 0,
        "original_sequences": 0,
        "celeb-real-mtcnn": 0,
        "youtube-real-mtcnn": 0,
        # fake
        "1_fake": 1,
        "0_fake": 1,
        "fake": 1,
        "synthesis": 1,
        "manipulated_sequences": 1,
        "celeb-synthesis-mtcnn": 1,
    }
    base = os.path.basename(os.path.abspath(root)).strip().lower()
    return alias_to_label.get(base)


def _collect_recursive_image_samples(root, label):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    samples = []
    for current_root, _, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                samples.append((os.path.join(current_root, fname), int(label)))
    samples.sort(key=lambda x: x[0])
    return samples


def _resolve_dfdc_labels_path(root, opt):
    root_abs = os.path.abspath(root)
    explicit_candidates = [
        getattr(opt, "dfdc_labels_npy", None),
        getattr(opt, "data_dfdc_labels_npy", None),
    ]
    candidates = [c for c in explicit_candidates if c]
    candidates.extend(
        [
            os.path.join(os.path.dirname(root_abs), "labels.npy"),
            os.path.join(os.path.dirname(os.path.dirname(root_abs)), "labels.npy"),
            os.path.join(root_abs, "labels.npy"),
        ]
    )
    seen = set()
    for candidate in candidates:
        p = os.path.abspath(os.path.expanduser(str(candidate)))
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return None


def is_dfdc_train_faces_root(root, opt=None):
    root_abs = os.path.abspath(root)
    if os.path.basename(root_abs).strip().lower() != "train_faces":
        return False
    opt_obj = opt if opt is not None else object()
    return _resolve_dfdc_labels_path(root_abs, opt_obj) is not None


def _build_dfdc_train_faces_dataset(opt, root, transform):
    root_abs = os.path.abspath(root)
    if not is_dfdc_train_faces_root(root_abs, opt):
        return None

    labels_path = _resolve_dfdc_labels_path(root_abs, opt)
    if labels_path is None:
        return None

    try:
        labels_raw = np.load(labels_path, allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f"Failed to load DFDC labels file: {labels_path} ({exc})") from exc

    if not isinstance(labels_raw, dict):
        raise RuntimeError(f"DFDC labels file is not a dict: {labels_path}")

    video_to_label = {}
    for k, v in labels_raw.items():
        key = str(k)
        if key.endswith(".mp4"):
            key = key[:-4]
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv in (0, 1):
            video_to_label[key] = iv

    if not video_to_label:
        raise RuntimeError(f"No valid binary labels parsed from: {labels_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    samples = []
    total_video_dirs = 0
    missing_label_video_dirs = 0
    missing_label_examples = []

    for part_name in sorted(os.listdir(root_abs)):
        part_dir = os.path.join(root_abs, part_name)
        if not os.path.isdir(part_dir):
            continue
        for video_name in sorted(os.listdir(part_dir)):
            video_dir = os.path.join(part_dir, video_name)
            if not os.path.isdir(video_dir):
                continue
            total_video_dirs += 1
            label = video_to_label.get(video_name)
            if label is None:
                missing_label_video_dirs += 1
                if len(missing_label_examples) < 5:
                    missing_label_examples.append(video_dir)
                continue
            for fname in sorted(os.listdir(video_dir)):
                if os.path.splitext(fname)[1].lower() in exts:
                    samples.append((os.path.join(video_dir, fname), int(label)))

    if not samples:
        raise RuntimeError(
            f"Detected DFDC train_faces root but no labeled image files were collected under {root_abs}"
        )

    if missing_label_video_dirs > 0:
        print(
            f"[DFDC] Warning: {missing_label_video_dirs}/{total_video_dirs} video folders in {root_abs} "
            f"missing labels from {labels_path}. Examples: {missing_label_examples}"
        )

    real_count = sum(1 for _, t in samples if int(t) == 0)
    fake_count = len(samples) - real_count
    print(
        f"[DFDC] Loaded train_faces with labels from {labels_path}: "
        f"samples={len(samples)} real={real_count} fake={fake_count}"
    )

    skip_broken_images = bool(getattr(opt, "skip_broken_images", True))
    skip_log_limit = int(getattr(opt, "skip_broken_images_log_limit", 0))
    return FlatLabeledImageDataset(
        samples=samples,
        transform=transform,
        skip_broken=skip_broken_images,
        skip_log_limit=skip_log_limit,
    )


def _build_binary_datasets(opt, root, transform):
    """Create ImageFolder datasets for every binary directory discovered under root."""
    binary_roots = _find_binary_roots(root)
    if not binary_roots:
        dfdc_dataset = _build_dfdc_train_faces_dataset(opt, root, transform)
        if dfdc_dataset is not None:
            return dfdc_dataset

        # Fallback: flat folder with filename labels like xxx_real.png / xxx_fake.png
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        suffix_re = re.compile(r'(?:^|[_-])(real|fake)$', flags=re.IGNORECASE)
        flat_samples = []
        root_path = os.path.abspath(root)
        for p in sorted(os.listdir(root_path)):
            abs_p = os.path.join(root_path, p)
            if not os.path.isfile(abs_p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in exts:
                continue
            stem = os.path.splitext(p)[0]
            m = suffix_re.search(stem)
            if not m:
                continue
            label = 0 if m.group(1).lower() == 'real' else 1
            flat_samples.append((abs_p, label))

        skip_broken_images = bool(getattr(opt, "skip_broken_images", True))
        skip_log_limit = int(getattr(opt, "skip_broken_images_log_limit", 0))
        if flat_samples:
            return FlatLabeledImageDataset(
                samples=flat_samples,
                transform=transform,
                skip_broken=skip_broken_images,
                skip_log_limit=skip_log_limit,
            )

        forced_label = infer_single_label_from_root(root)
        if forced_label is not None:
            recursive_samples = _collect_recursive_image_samples(root, forced_label)
            if not recursive_samples:
                raise RuntimeError(
                    f"Single-label root inferred for {root}, but no image files were found recursively."
                )
            return FlatLabeledImageDataset(
                samples=recursive_samples,
                transform=transform,
                skip_broken=skip_broken_images,
                skip_log_limit=skip_log_limit,
            )

        raise RuntimeError(
            f"Could not locate binary folders (0_real/1_fake, raw/synthesis, real/fake), "
            f"flat *_real/*_fake files, or a known single-label root alias under {root}"
        )

    datasets_list = []
    skip_broken_images = bool(getattr(opt, "skip_broken_images", True))
    skip_log_limit = int(getattr(opt, "skip_broken_images_log_limit", 0))
    for binary_root in binary_roots:
        dset = SafeImageFolder(
            binary_root,
            transform=transform,
            skip_broken=skip_broken_images,
            skip_log_limit=skip_log_limit,
        )
        idx_to_class = {idx: cls_name for cls_name, idx in dset.class_to_idx.items()}
        class_names = set(idx_to_class.values())

        if class_names == {'0_real', '1_fake'}:
            desired_order = {'0_real': 0, '1_fake': 1}
        elif class_names == {'0_fake', '1_real'}:
            # reverse-labelled datasets: map back to canonical ordering
            desired_order = {'1_real': 0, '0_fake': 1}
        elif class_names == {'raw', 'synthesis'}:
            desired_order = {'raw': 0, 'synthesis': 1}
        elif class_names == {'real', 'fake'}:
            desired_order = {'real': 0, 'fake': 1}
        else:
            raise RuntimeError(
                f"Unsupported class names {class_names} under {binary_root}; "
                "expected 0_real/1_fake, 0_fake/1_real, raw/synthesis, or real/fake"
            )

        remapped = []
        for path, target in dset.samples:
            cls_name = idx_to_class[target]
            if cls_name not in desired_order:
                raise RuntimeError(
                    f"Encountered unexpected class '{cls_name}' in {binary_root}; expected one of {list(desired_order.keys())}"
                )
            remapped.append((path, desired_order[cls_name]))

        if remapped != dset.samples:
            dset.samples = remapped
            dset.imgs = remapped
            dset.targets = [t for _, t in remapped]

        # Normalise metadata to canonical labels for downstream consumers
        dset.class_to_idx = {'0_real': 0, '1_fake': 1}
        dset.classes = ['0_real', '1_fake']
        datasets_list.append(dset)

    if len(datasets_list) == 1:
        return datasets_list[0]
    return torch.utils.data.ConcatDataset(datasets_list)


def _pad_to_min_size(img, target_h, target_w):
    """Pad image with zeros so that height/width are at least target sizes."""
    width, height = img.size
    pad_w = max(target_w - width, 0)
    pad_h = max(target_h - height, 0)
    if pad_w == 0 and pad_h == 0:
        return img

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=0)


def binary_dataset(opt, root):
    # Allow switching to a simplified (legacy) preprocessing pipeline to match the NPR script.
    legacy_preprocess = getattr(opt, "legacy_preprocess", False)

    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        # rz_func = transforms.Resize(opt.loadSize)
    if legacy_preprocess:
        # Match the older flow: resize -> basic augment -> crop -> flip -> IMAGENET normalize.
        normalize_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # normalize_transform = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: _apply_explicit_robustness(img, opt)),
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            normalize_transform,
        ])
        return _build_binary_datasets(opt, root, transform)

    NOISE_TYPE = getattr(opt, "noise_type", None)
    NOISE_RATIO = getattr(opt, "noise_ratio", None)

    opt_class = Options(noise_type=NOISE_TYPE, noise_ratio=NOISE_RATIO)
    processor = ImageProcessor(opt_class)

    # 定义Albumentations增强（兼容不同版本的 ImageCompression 参数）
    def _build_image_compression():
        try:
            return ImageCompression(quality_range=(60, 100), p=0.5)
        except TypeError:
            return ImageCompression(quality_lower=60, quality_upper=100, p=0.5)

    aug = Compose([
        _build_image_compression(),
        GaussNoise(p=0.2),
        MotionBlur(p=0.2),
        GaussianBlur(blur_limit=3, p=0.5),
        _build_downscale_aug(),
    ])

    # 自定义转换函数：将PIL图像转换为NumPy数组，应用Albumentations增强，再转回PIL
    def albumentations_aug(img):
        # 将PIL图像转换为NumPy数组 (H, W, C) RGB格式
        img_np = np.array(img)
        # 应用Albumentations增强
        augmented = aug(image=img_np)
        # 从字典中提取增强后的图像
        img_aug = augmented['image']
        # 将NumPy数组转回PIL图像
        return Image.fromarray(img_aug)

    DATA_AUG = getattr(opt, "data_aug", False)
    USE_TESTSHIFT_LIKE_AUG_TRAIN = bool(getattr(opt, "use_testshift_like_aug", False)) and bool(opt.isTrain)
    USE_TESTSHIFT_LIKE_AUG_EVAL = bool(getattr(opt, "use_testshift_like_aug_eval", False)) and (not bool(opt.isTrain))
    USE_TESTSHIFT_LIKE_AUG = USE_TESTSHIFT_LIKE_AUG_TRAIN or USE_TESTSHIFT_LIKE_AUG_EVAL
    TESTSHIFT_LIKE_APPLY_PROB = float(getattr(opt, "testshift_like_apply_prob", 1.0))
    TESTSHIFT_LIKE_MAX_GROUPS = int(getattr(opt, "testshift_like_max_groups", 3))
    USE_AUG_UTILS_EVAL = bool(getattr(opt, "use_aug_utils_eval", False)) and (not bool(opt.isTrain))
    USE_AUG_UTILS_TRAIN = bool(getattr(opt, "use_aug_utils_train", False)) and (
        bool(opt.isTrain) or USE_AUG_UTILS_EVAL
    )
    AUG_UTILS_PROB = float(getattr(opt, "aug_utils_prob", 1.0))
    AUG_UTILS_MAX_DISTORTIONS = int(getattr(opt, "aug_utils_max_distortions", 3))
    AUG_UTILS_NUM_LEVELS = int(getattr(opt, "aug_utils_num_levels", 5))
    # 创建数据增强变换（deepfake专用 TestShift-like 与原有增强互斥，避免叠加偏移）
    if USE_TESTSHIFT_LIKE_AUG:
        testshift_aug = _build_testshift_like_aug(
            p_apply=TESTSHIFT_LIKE_APPLY_PROB,
            max_groups=TESTSHIFT_LIKE_MAX_GROUPS,
        )

        def _testshift_like_aug(img):
            img_np = np.array(img)
            out = testshift_aug(image=img_np)["image"]
            return Image.fromarray(out)

        data_aug = transforms.Lambda(_testshift_like_aug)
        print(
            f"[Aug] Using testshift-like aug (prob={TESTSHIFT_LIKE_APPLY_PROB}, "
            f"max_groups={TESTSHIFT_LIKE_MAX_GROUPS})"
        )
    elif DATA_AUG:
        data_aug = transforms.Lambda(albumentations_aug)
    else:
        data_aug = transforms.Lambda(lambda img: img)

    if isinstance(opt.cropSize, (tuple, list)):
        target_h = opt.cropSize[0]
        target_w = opt.cropSize[1] if len(opt.cropSize) > 1 else opt.cropSize[0]
    else:
        target_h = target_w = opt.cropSize

    apply_padding = opt.no_resize and (opt.isTrain or not opt.no_crop)
    pad_func = transforms.Lambda(lambda img: _pad_to_min_size(img, target_h, target_w)) if apply_padding else transforms.Lambda(lambda img: img)

    trainmode = str(getattr(opt, 'trainmode', '')).lower()
    modelname = str(getattr(opt, 'modelname', ''))
    if trainmode == 'lora' and modelname.startswith('CLIP:'):
        normalize_transform = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    else:
        normalize_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Optional: apply augmentations from aug_utils_train on tensor in [0, 1].
    aug_utils_transform = transforms.Lambda(lambda tens: tens)
    if USE_AUG_UTILS_TRAIN:
        try:
            from aug_utils_train.utils_data import distort_images as aug_utils_distort_images

            def _apply_aug_utils_train(tens):
                if random() > AUG_UTILS_PROB:
                    return tens
                distorted, _, _ = aug_utils_distort_images(
                    tens,
                    max_distortions=AUG_UTILS_MAX_DISTORTIONS,
                    num_levels=AUG_UTILS_NUM_LEVELS,
                )
                return distorted

            aug_utils_transform = transforms.Lambda(_apply_aug_utils_train)
            stage = "train" if bool(opt.isTrain) else "eval"
            print(
                f"[Aug] Using aug_utils_train@{stage} (prob={AUG_UTILS_PROB}, "
                f"max_distortions={AUG_UTILS_MAX_DISTORTIONS}, num_levels={AUG_UTILS_NUM_LEVELS})"
            )
        except Exception as exc:
            print(f"[Aug] Failed to enable aug_utils_train, fallback to default aug only: {exc}")

    transform = transforms.Compose([
        transforms.Lambda(lambda img: processor.data_augment(img)),
        transforms.Lambda(lambda img: _apply_explicit_robustness(img, opt)),
        # 使用自定义函数替换原processor.data_augment
        data_aug,
        rz_func,         # 后续PyTorch变换 (如调整大小)
        pad_func,        # 当禁用resize且图像不足裁剪尺寸时补零
        crop_func,       # 裁剪
        flip_func,       # 翻转
        transforms.ToTensor(),
        aug_utils_transform,
        normalize_transform,
    ])

    return _build_binary_datasets(opt, root, transform)


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
