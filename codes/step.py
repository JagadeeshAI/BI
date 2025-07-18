import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from data import get_dynamic_loader
from codes.utils import get_model, print_parameter_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forget_epochs = 50

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

def retention_loss(logits, labels):
    """Standard cross-entropy loss for retained classes"""
    return F.cross_entropy(logits, labels)

def forgetting_loss(logits, labels, BND=110):
    """ReLU(BND - cross_entropy) as per paper Eq. (10)"""
    ce_loss = F.cross_entropy(logits, labels)
    return F.relu(BND - ce_loss)

def incremental_loss(logits, labels):
    """Standard cross-entropy loss for new classes"""
    return F.cross_entropy(logits, labels)

def train_forgetting_incremental(model, retain_loader, forget_loader, new_loader, optimizer, 
                                beta_forget=0.05, beta_new=0.3, BND=50):
    """
    Sequential phased training:
    Phase 1 (0-10): Focus on forgetting + light new learning
    Phase 2 (11-30): Focus on new learning + maintain retention  
    Phase 3 (31-49): Recovery - balance both
    """
    
    for epoch in range(forget_epochs):
        model.train()
        
        # Phase determination with sequential approach
        if epoch <= 10:
            # Phase 1: Focus on forgetting + light new learning
            current_beta_forget = beta_forget
            current_beta_new = 0.2  # Very light new learning
            retain_weight = 2.0     # Emphasize retention
            phase_name = "Forgetting"
        elif epoch <= 30:
            # Phase 2: Focus on new learning + maintain retention
            current_beta_forget = 0.0
            current_beta_new = 0.8  # Strong new learning
            retain_weight = 2.0     # Keep retention strong
            phase_name = "Learning"
        else:
            # Phase 3: Recovery - balance both
            current_beta_forget = 0.0
            current_beta_new = 0.5  # Balanced
            retain_weight = 2.5     # Extra emphasis on retention
            phase_name = "Recovery"
        
        # Create iterator for cycling through datasets
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)
        new_iter = iter(new_loader)
        
        # Use the longest dataset as the base
        max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
        
        loop = tqdm(range(max_batches), desc=f"{phase_name} Epoch {epoch+1}")
        
        for batch_idx in loop:
            # Get batch from each dataset (cycle if needed)
            try:
                rx, ry = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rx, ry = next(retain_iter)
            
            try:
                fx, fy = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                fx, fy = next(forget_iter)
            
            try:
                nx, ny = next(new_iter)
            except StopIteration:
                new_iter = iter(new_loader)
                nx, ny = next(new_iter)
            
            # Move to device
            rx, ry = rx.to(device), ry.to(device)
            fx, fy = fx.to(device), fy.to(device)
            nx, ny = nx.to(device), ny.to(device)
            
            # Forward pass
            retain_logits = model(rx)
            loss_retain = retention_loss(retain_logits, ry)
            
            new_logits = model(nx)
            loss_new = incremental_loss(new_logits, ny)
            
            if current_beta_forget > 0:
                # Forgetting phase: retention + forgetting + light new learning
                forget_logits = model(fx)
                loss_forget = forgetting_loss(forget_logits, fy, BND)
                total_loss = (retain_weight * loss_retain + 
                             current_beta_forget * loss_forget + 
                             current_beta_new * loss_new)
                
                loop.set_postfix({
                    'retain': f'{loss_retain.item():.4f}',
                    'forget': f'{loss_forget.item():.4f}',
                    'new': f'{loss_new.item():.4f}',
                    'β_f': f'{current_beta_forget:.2f}',
                    'β_n': f'{current_beta_new:.2f}',
                    'r_w': f'{retain_weight:.1f}'
                })
            else:
                # Learning/Recovery phase: retention + new classes
                total_loss = (retain_weight * loss_retain + 
                             current_beta_new * loss_new)
                
                loop.set_postfix({
                    'retain': f'{loss_retain.item():.4f}',
                    'new': f'{loss_new.item():.4f}',
                    'β_n': f'{current_beta_new:.2f}',
                    'r_w': f'{retain_weight:.1f}',
                    'mode': phase_name
                })
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Evaluate after each epoch
        retain_val_loader = get_dynamic_loader(class_range=(10, 49), mode="val", batch_size=16)
        forget_val_loader = get_dynamic_loader(class_range=(0, 9), mode="val", batch_size=16)
        new_val_loader = get_dynamic_loader(class_range=(50, 59), mode="val", batch_size=16)
        combined_val_loader = get_dynamic_loader(class_range=(10, 59), mode="val", batch_size=16)
        
        retain_acc = evaluate(model, retain_val_loader, device, 100)
        forget_acc = evaluate(model, forget_val_loader, device, 100)
        new_acc = evaluate(model, new_val_loader, device, 100)
        combined_acc = evaluate(model, combined_val_loader, device, 100)
        
        print(f"Epoch {epoch+1}: Retain (10-49): {retain_acc:.2f}% | Forget (0-9): {forget_acc:.2f}% | New (50-59): {new_acc:.2f}% | Combined (10-59): {combined_acc:.2f}%")
        
        # Phase transition notifications
        if epoch == 10:
            print("🔄 Switching to LEARNING phase - focusing on new classes")
        elif epoch == 30:
            print("🔄 Switching to RECOVERY phase - balancing retention and new classes")

def main():
    os.makedirs("checkpoints/steps", exist_ok=True)
    num_classes = 100
    initial_ckpt = "checkpoints/oracle/0_49.pth"
    
    # Load model trained on classes 0-49
    model = get_model(num_classes=num_classes, use_lora=True, pretrained=False, lora_rank=8).to(device)
    print_parameter_stats(model)
    
    state_dict = torch.load(initial_ckpt, map_location=device, weights_only=True)
    filtered_state_dict = {k: v for k, v in state_dict.items()
                          if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)

    # Define classes: retain (10-49), forget (0-9), new (50-59)
    retain_classes = list(range(10, 50))   # Classes 10-49
    forget_classes = list(range(0, 10))    # Classes 0-9
    new_classes = list(range(50, 60))      # Classes 50-59
    
    print(f"🔄 Retaining classes {retain_classes[0]}–{retain_classes[-1]}")
    print(f"❌ Forgetting classes {forget_classes[0]}–{forget_classes[-1]}")
    print(f"➕ Adding new classes {new_classes[0]}–{new_classes[-1]}")
    
    # Create data loaders
    retain_loader = get_dynamic_loader(class_range=(retain_classes[0], retain_classes[-1]), mode="train", batch_size=64)
    forget_loader = get_dynamic_loader(class_range=(forget_classes[0], forget_classes[-1]), mode="train", batch_size=64)
    new_loader = get_dynamic_loader(class_range=(new_classes[0], new_classes[-1]), mode="train", batch_size=64)
    
    # Optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=forget_epochs)
    
    # Train with sequential phased approach
    train_forgetting_incremental(model, retain_loader, forget_loader, new_loader, optimizer, 
                                beta_forget=0.05, beta_new=0.3, BND=50)
    
    # Save checkpoint (10-59.pth)
    output_ckpt = "checkpoints/steps/10_59.pth"
    torch.save(model.state_dict(), output_ckpt)
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    
    # Retained classes (10-49)
    retain_val_loader = get_dynamic_loader(class_range=(retain_classes[0], retain_classes[-1]), mode="val", batch_size=16)
    retain_acc = evaluate(model, retain_val_loader, device, 100)
    print(f"✅ Retained classes (10-49): {retain_acc:.2f}%")
    
    # Forgotten classes (0-9) - should be low
    forget_val_loader = get_dynamic_loader(class_range=(forget_classes[0], forget_classes[-1]), mode="val", batch_size=16)
    forget_acc = evaluate(model, forget_val_loader, device, 100)
    print(f"❌ Forgotten classes (0-9): {forget_acc:.2f}%")
    
    # New classes (50-59) - should be high
    new_val_loader = get_dynamic_loader(class_range=(new_classes[0], new_classes[-1]), mode="val", batch_size=16)
    new_acc = evaluate(model, new_val_loader, device, 100)
    print(f"➕ New classes (50-59): {new_acc:.2f}%")
    
    # Combined performance (10-59)
    combined_val_loader = get_dynamic_loader(class_range=(10, 59), mode="val", batch_size=16)
    combined_acc = evaluate(model, combined_val_loader, device, 100)
    print(f"🎯 Combined performance (10-59): {combined_acc:.2f}%")
    
    print(f"\n✅ Done | Saved to {output_ckpt}")

if __name__ == "__main__":
    main()