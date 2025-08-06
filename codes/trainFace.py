import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import backbone.vits_face as vits_face
from codes.facedata import get_dynamic_loader

def train_face_model(model, loss_type, num_epochs=100, batch_size=32, lr=1e-4, 
                    class_range=(0, 49), device='cuda', save_dir='/home/jag/codes/Bi/checkpoints/face'):
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader = get_dynamic_loader(class_range=class_range, mode="train", batch_size=batch_size)
    val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=batch_size)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Move model to device
    model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Training tracking
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"Starting training with {loss_type} loss")
    print(f"Classes: {class_range[0]}-{class_range[1]} ({class_range[1]-class_range[0]+1} classes)")
    print(f"Device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - handle different loss types properly
            if loss_type == 'SFace':
                outputs = model(images, labels)
                loss_output = outputs[1]  # SFace returns (output, loss, ...)
                logits = outputs[0]
            else:
                # For ArcFace, CosFace, Softmax - they return (loss_logits, embeddings)
                logits, embeddings = model(images, labels)
                # The logits already have the loss function applied (margins added)
                # Apply CrossEntropy on the processed logits
                loss_output = nn.CrossEntropyLoss()(logits, labels)
            
            loss_output.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss_output.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss_output.item():.4f}',
                'Acc': f'{train_acc:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if loss_type == 'SFace':
                    outputs = model(images, labels)
                    logits = outputs[0]
                    loss = outputs[1]
                else:
                    logits, embeddings = model(images, labels)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_acc:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': epoch_train_loss,
                'val_acc': epoch_val_acc,
                'loss_type': loss_type,
                'class_range': class_range,
                'model_config': {
                    'loss_type': loss_type,
                    'GPU_ID': None,
                    'num_class': class_range[1] - class_range[0] + 1,
                    'image_size': 224,
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
            }
            
            best_path = os.path.join(save_dir, f'best_model_{loss_type.lower()}.pth')
            torch.save(checkpoint, best_path)
            print(f'New best model saved! Val Acc: {best_val_acc:.2f}%')
        
        # Save latest checkpoint
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'loss_type': loss_type,
            'class_range': class_range
        }
        
        latest_path = os.path.join(save_dir, f'latest_model_{loss_type.lower()}.pth')
        torch.save(latest_checkpoint, latest_path)
        
        print('-' * 60)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Models saved in: {save_dir}')
    
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best val acc: {checkpoint.get('best_val_acc', 'N/A')}")
    
    return checkpoint

# Example usage
if __name__ == "__main__":
    model_config = {
        'loss_type': 'ArcFace',
        'GPU_ID': [0],
        'num_class': 50,
        'image_size': 224,
        'patch_size': 16,
        'ac_patch_size': 16, 
        'pad': 0,
        'dim': 192,
        'depth': 12,
        'heads': 3,
        'mlp_dim': 768,  # 192 * 4
        'pool': 'cls',
        'channels': 3,
        'dim_head': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'lora_rank': 0  # No LoRA
    }
    
    model = vits_face.ViTs_face(**model_config)
    
    results = train_face_model(
        model=model,
        loss_type='ArcFace',
        num_epochs=50,
        batch_size=16,
        lr=1e-4,
        class_range=(0, 49),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )