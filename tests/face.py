from backbone.vits_face import ViTs_face

import torch
import torch.nn as nn

def test_vits_face():
    """Test ViTs_face model with dummy inputs"""
    
    # Model parameters
    config = {
        'loss_type': 'ArcFace',  # or 'CosFace', 'SFace', 'Softmax'
        'GPU_ID': None,  # Use None for CPU, [0] for single GPU
        'num_class': 1000,  # Number of identities
        'image_size': 112,
        'patch_size': 8,
        'ac_patch_size': 12,
        'pad': 4,
        'dim': 512,
        'depth': 12,
        'heads': 8,
        'mlp_dim': 2048,
        'pool': 'cls',
        'channels': 3,
        'dim_head': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'lora_rank': 8
    }
    
    # Create model
    model = ViTs_face(**config)
    model.eval()
    
    # Create dummy inputs
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 112, 112)
    dummy_labels = torch.randint(0, config['num_class'], (batch_size,))
    
    print("=== Model Architecture ===")
    print(f"Model: {model.__class__.__name__}")
    print(f"Loss type: {config['loss_type']}")
    print(f"Number of classes: {config['num_class']}")
    print(f"Input shape: {dummy_images.shape}")
    print(f"Labels shape: {dummy_labels.shape}")
    
    # Test inference mode (no labels)
    print("\n=== Inference Mode (no labels) ===")
    with torch.no_grad():
        embeddings = model(dummy_images)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embeddings sample: {embeddings[0, :5]}")  # First 5 values
    
    # Test training mode (with labels)
    print("\n=== Training Mode (with labels) ===")
    model.train()
    loss_output, embeddings = model(dummy_images, dummy_labels)
    
    print(f"Loss output shape: {loss_output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Loss sample values: {loss_output[0, :5]}")
    
    # Test different loss types
    print("\n=== Testing Different Loss Functions ===")
    loss_types = ['Softmax', 'ArcFace', 'CosFace', 'SFace']
    
    for loss_type in loss_types:
        print(f"\n--- {loss_type} Loss ---")
        config_test = config.copy()
        config_test['loss_type'] = loss_type
        
        model_test = ViTs_face(**config_test)
        model_test.eval()
        
        with torch.no_grad():
            if loss_type == 'SFace':
                # SFace returns multiple outputs
                outputs = model_test(dummy_images, dummy_labels)
                print(f"SFace outputs (6 values): {[x.shape if hasattr(x, 'shape') else x for x in outputs]}")
            else:
                loss_out, emb_out = model_test(dummy_images, dummy_labels)
                print(f"Loss shape: {loss_out.shape}, Embedding shape: {emb_out.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test gradient flow
    print(f"\n=== Gradient Test ===")
    model.train()
    loss_out, emb_out = model(dummy_images, dummy_labels)
    
    # Compute a simple loss for backprop test
    if config['loss_type'] == 'SFace':
        total_loss = loss_out[1]  # SFace returns loss as second element
    else:
        total_loss = torch.nn.functional.cross_entropy(loss_out, dummy_labels)
    
    total_loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'weight' in name and len(grad_norms) <= 3:  # Show first 3 weight gradients
                print(f"{name}: grad_norm = {grad_norm:.6f}")
    
    print(f"Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")

if __name__ == "__main__":
    test_vits_face()