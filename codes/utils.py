from backbone.vit import VisionTransformer
import timm
import torch
def load_timm_pretrained_weights(custom_model, model_name='deit_tiny_patch16_224'):
    """Loads pretrained weights from timm into your custom model."""
    timm_model = timm.create_model(model_name, pretrained=True)
    pretrained_state_dict = timm_model.state_dict()

    # Remove the classifier head weights if num_classes differ
    pretrained_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if not k.startswith("head.")
    }

    missing, unexpected = custom_model.load_state_dict(pretrained_state_dict, strict=False)

    print(f"Loaded pretrained weights from timm: {model_name}")
 

def load_model_weights(model, checkpoint_path, strict=False):
    print(f"🔄 Loading model weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = model.state_dict()
    compatible_state = {k: v for k, v in state_dict.items()
                        if k in model_state and model_state[k].shape == v.shape}
    missing_keys = [k for k in model_state if k not in compatible_state]
    print(f"⚠️ Skipping incompatible or missing keys: {missing_keys[:5]}... (+{len(missing_keys) - 5} more)" if len(missing_keys) > 5 else f"⚠️ Skipping keys: {missing_keys}")
    model.load_state_dict(compatible_state, strict=strict)


def get_model(num_classes=100, use_lora=False, lora_rank=8, pretrained=True , drop_path_rate=0, drop_rate=0, attn_drop_rate=0):
    if not use_lora:
        lora_rank = 0
    model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=num_classes,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=drop_rate,        
    attn_drop_rate=attn_drop_rate,   
    drop_path_rate=drop_path_rate,  
    use_lora=use_lora,
    lora_rank=lora_rank
)


    if pretrained:
        load_timm_pretrained_weights(model, model_name='deit_tiny_patch16_224')
        print("✅ Loaded pretrained weights from timm")

    return model

def print_parameter_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    percent = 100 * trainable_params / total_params

    print(f"\n📊 Parameter Summary:")
    print(f"  🔢 Total Parameters     : {total_m:.2f}M")
    print(f"  ✅ Trainable Parameters : {trainable_m:.2f}M")
    print(f"  📉 % Trainable          : {percent:.2f}%")
