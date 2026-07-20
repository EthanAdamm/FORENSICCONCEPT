from .clip_models import CLIPModel


VALID_NAMES = [
    "CLIP:ViT-L/14",
]





def get_model(name, opt=None):
    if name not in VALID_NAMES:
        # Allow custom CLIP identifiers when explicit checkpoint path is provided.
        custom_clip_ok = (
            name.startswith("CLIP:")
            and opt is not None
            and bool(getattr(opt, "clip_model_path", None))
        )
        assert custom_clip_ok, f"Unsupported model name: {name}"
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:], 1, opt)
    raise ValueError(f"Unsupported model name: {name}")
