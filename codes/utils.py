import torch
import timm
from backbone.vit import VisionTransformer


def get_model(
    num_classes: int = 1000,
    use_lora: bool = False,
    pretrained: bool = False,
    lora_rank: int = 0,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0
) -> VisionTransformer:
    """
    Factory for VisionTransformer models.
    Supports optional LoRA adapters and custom dropout/stochastic-depth rates.

    Args:
        num_classes: number of classes for the classification head.
        use_lora: whether to insert LoRA adapters into MLPs.
        pretrained: if True, load pretrained weights from timm (except final head).
        lora_rank: rank for LoRA adapters (ignored if use_lora=False).
        img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, qkv_bias:
            core ViT architecture hyperparameters.
        drop_rate: dropout probability after projections.
        attn_drop_rate: dropout probability inside attention.
        drop_path_rate: stochastic depth rate across blocks.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_lora=use_lora,
        lora_rank=lora_rank
    )

    if pretrained:
        # load base weights and skip the classification head
        base_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        state_dict = base_model.state_dict()
        # remove head params to avoid size mismatch
        state_dict.pop('head.weight', None)
        state_dict.pop('head.bias', None)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pretrained weights from timm (skipped head): deit_tiny_patch16_224")

    return model


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    """
    Load a checkpoint into model.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"⚠️ Skipping incompatible or missing keys: {missing + unexpected}")
