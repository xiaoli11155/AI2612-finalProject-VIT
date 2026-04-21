from .hybrid_vit import HybridViT
from .resnet_baseline import ResNetBaseline
from .vit import VisionTransformer


def build_model(model_cfg, num_classes: int):
    model_type = model_cfg["type"].lower()
    if model_type == "vit":
        return VisionTransformer(
            image_size=int(model_cfg["image_size"]),
            patch_size=int(model_cfg["patch_size"]),
            num_classes=num_classes,
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            attn_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        )
    if model_type == "resnet":
        return ResNetBaseline(
            variant=model_cfg.get("variant", "resnet18"),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_type == "hybrid":
        return HybridViT(
            image_size=int(model_cfg["image_size"]),
            num_classes=num_classes,
            stem_channels=list(model_cfg.get("stem_channels", [64, 128])),
            embed_dim=int(model_cfg["embed_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            attn_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
