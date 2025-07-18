import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime
from data import get_dynamic_loader
from codes.utils import get_model, print_parameter_stats

# Setup logging
def setup_logging():
    """Setup minimal logging for the CLU training process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/clu_training"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"clu_training_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('CLU_Training')
    logger.setLevel(logging.WARNING)  # Only warnings and errors
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler for logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    
    return logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forget_epochs = 50

def evaluate(model, dataloader, device, num_classes, logger=None):
    """Evaluate model performance on given dataloader"""
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy

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
                                step_num, retain_classes, forget_classes, new_classes,
                                beta_forget=0.05, beta_new=0.3, BND=50, logger=None):
    """
    Sequential phased training for one step
    """
    
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: Retain {retain_classes[0]}-{retain_classes[-1]} | Forget {forget_classes[0]}-{forget_classes[-1]} | Learn {new_classes[0]}-{new_classes[-1]}")
    print(f"{'='*60}")
    
    step_losses = {
        'retain': [],
        'forget': [],
        'new': [],
        'total': []
    }
    
    step_accuracies = {
        'retain': [],
        'forget': [],
        'new': [],
        'combined': []
    }
    
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
        
        # Phase transition logging
        if epoch == 0:
            print(f"🔄 Phase 1: Forgetting (β_forget: {current_beta_forget}, β_new: {current_beta_new})")
        elif epoch == 11:
            print(f"🔄 Phase 2: Learning (β_forget: {current_beta_forget}, β_new: {current_beta_new})")
        elif epoch == 31:
            print(f"🔄 Phase 3: Recovery (β_forget: {current_beta_forget}, β_new: {current_beta_new})")
        
        # Create iterator for cycling through datasets
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)
        new_iter = iter(new_loader)
        
        # Use the longest dataset as the base
        max_batches = max(len(retain_loader), len(forget_loader), len(new_loader))
        
        loop = tqdm(range(max_batches), desc=f"Step {step_num} - {phase_name} Epoch {epoch+1}")
        
        epoch_losses = {'retain': [], 'forget': [], 'new': [], 'total': []}
        
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
                
                epoch_losses['forget'].append(loss_forget.item())
                
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
                
                epoch_losses['forget'].append(0.0)  # No forgetting loss
                
                loop.set_postfix({
                    'retain': f'{loss_retain.item():.4f}',
                    'new': f'{loss_new.item():.4f}',
                    'β_n': f'{current_beta_new:.2f}',
                    'r_w': f'{retain_weight:.1f}',
                    'mode': phase_name
                })
            
            # Store losses
            epoch_losses['retain'].append(loss_retain.item())
            epoch_losses['new'].append(loss_new.item())
            epoch_losses['total'].append(total_loss.item())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Calculate epoch averages
        avg_retain_loss = sum(epoch_losses['retain']) / len(epoch_losses['retain'])
        avg_forget_loss = sum(epoch_losses['forget']) / len(epoch_losses['forget'])
        avg_new_loss = sum(epoch_losses['new']) / len(epoch_losses['new'])
        avg_total_loss = sum(epoch_losses['total']) / len(epoch_losses['total'])
        
        # Store step losses
        step_losses['retain'].append(avg_retain_loss)
        step_losses['forget'].append(avg_forget_loss)
        step_losses['new'].append(avg_new_loss)
        step_losses['total'].append(avg_total_loss)
        
        # Evaluate after each epoch
        retain_val_loader = get_dynamic_loader(class_range=(retain_classes[0], retain_classes[-1]), mode="val", batch_size=16)
        forget_val_loader = get_dynamic_loader(class_range=(forget_classes[0], forget_classes[-1]), mode="val", batch_size=16)
        new_val_loader = get_dynamic_loader(class_range=(new_classes[0], new_classes[-1]), mode="val", batch_size=16)
        combined_val_loader = get_dynamic_loader(class_range=(retain_classes[0], new_classes[-1]), mode="val", batch_size=16)
        
        retain_acc = evaluate(model, retain_val_loader, device, 100)
        forget_acc = evaluate(model, forget_val_loader, device, 100)
        new_acc = evaluate(model, new_val_loader, device, 100)
        combined_acc = evaluate(model, combined_val_loader, device, 100)
        
        # Store accuracies
        step_accuracies['retain'].append(retain_acc)
        step_accuracies['forget'].append(forget_acc)
        step_accuracies['new'].append(new_acc)
        step_accuracies['combined'].append(combined_acc)
        
        # Log epoch results (every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            epoch_log = (f"Step {step_num} - Epoch {epoch+1:2d} | "
                        f"Retain: {retain_acc:5.2f}% | "
                        f"Forget: {forget_acc:5.2f}% | "
                        f"New: {new_acc:5.2f}% | "
                        f"Combined: {combined_acc:5.2f}%")
            print(epoch_log)
    
    return step_losses, step_accuracies

def run_multi_step_clu(logger=None):
    """Run the complete 5-step CLU training process"""
    
    print("🚀 Starting Multi-Step CLU Training")
    print(f"Device: {device}")
    
    # Create checkpoint directories
    os.makedirs("checkpoints/steps", exist_ok=True)
    os.makedirs("checkpoints/oracle", exist_ok=True)
    
    num_classes = 100
    
    # Define the 5 steps
    steps_config = [
        {
            'step': 1,
            'input_ckpt': 'checkpoints/oracle/0_49.pth',
            'output_ckpt': 'checkpoints/steps/10_59.pth',
            'retain_classes': list(range(10, 50)),   # 10-49
            'forget_classes': list(range(0, 10)),    # 0-9
            'new_classes': list(range(50, 60)),      # 50-59
        },
        {
            'step': 2,
            'input_ckpt': 'checkpoints/steps/10_59.pth',
            'output_ckpt': 'checkpoints/steps/20_69.pth',
            'retain_classes': list(range(20, 60)),   # 20-59
            'forget_classes': list(range(10, 20)),   # 10-19
            'new_classes': list(range(60, 70)),      # 60-69
        },
        {
            'step': 3,
            'input_ckpt': 'checkpoints/steps/20_69.pth',
            'output_ckpt': 'checkpoints/steps/30_79.pth',
            'retain_classes': list(range(30, 70)),   # 30-69
            'forget_classes': list(range(20, 30)),   # 20-29
            'new_classes': list(range(70, 80)),      # 70-79
        },
        {
            'step': 4,
            'input_ckpt': 'checkpoints/steps/30_79.pth',
            'output_ckpt': 'checkpoints/steps/40_89.pth',
            'retain_classes': list(range(40, 80)),   # 40-79
            'forget_classes': list(range(30, 40)),   # 30-39
            'new_classes': list(range(80, 90)),      # 80-89
        },
        {
            'step': 5,
            'input_ckpt': 'checkpoints/steps/40_89.pth',
            'output_ckpt': 'checkpoints/steps/50_99.pth',
            'retain_classes': list(range(50, 90)),   # 50-89
            'forget_classes': list(range(40, 50)),   # 40-49
            'new_classes': list(range(90, 100)),     # 90-99
        }
    ]
    
    # Store results for all steps
    all_step_results = {}
    
    # Execute each step
    for step_config in steps_config:
        step_num = step_config['step']
        
        print(f"\n🔄 STEP {step_num}/5")
        
        # Load model
        model = get_model(num_classes=num_classes, use_lora=True, pretrained=False, lora_rank=8).to(device)
        
        if step_num == 1:
            print_parameter_stats(model)
        
        # Load checkpoint
        if not os.path.exists(step_config['input_ckpt']):
            print(f"❌ Input checkpoint not found: {step_config['input_ckpt']}")
            continue
        
        state_dict = torch.load(step_config['input_ckpt'], map_location=device, weights_only=True)
        filtered_state_dict = {k: v for k, v in state_dict.items()
                              if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Create data loaders
        retain_loader = get_dynamic_loader(
            class_range=(step_config['retain_classes'][0], step_config['retain_classes'][-1]), 
            mode="train", batch_size=64
        )
        forget_loader = get_dynamic_loader(
            class_range=(step_config['forget_classes'][0], step_config['forget_classes'][-1]), 
            mode="train", batch_size=64
        )
        new_loader = get_dynamic_loader(
            class_range=(step_config['new_classes'][0], step_config['new_classes'][-1]), 
            mode="train", batch_size=64
        )
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=forget_epochs)
        
        # Train
        step_losses, step_accuracies = train_forgetting_incremental(
            model, retain_loader, forget_loader, new_loader, optimizer, 
            step_num, step_config['retain_classes'], step_config['forget_classes'], step_config['new_classes'],
            beta_forget=0.05, beta_new=0.3, BND=50, logger=logger
        )
        
        # Save checkpoint
        torch.save(model.state_dict(), step_config['output_ckpt'])
        
        # Final evaluation for this step
        print(f"\n📊 Step {step_num} Final Results:")
        
        # Evaluate on all relevant class ranges
        retain_val_loader = get_dynamic_loader(class_range=(step_config['retain_classes'][0], step_config['retain_classes'][-1]), mode="val", batch_size=16)
        forget_val_loader = get_dynamic_loader(class_range=(step_config['forget_classes'][0], step_config['forget_classes'][-1]), mode="val", batch_size=16)
        new_val_loader = get_dynamic_loader(class_range=(step_config['new_classes'][0], step_config['new_classes'][-1]), mode="val", batch_size=16)
        combined_val_loader = get_dynamic_loader(class_range=(step_config['retain_classes'][0], step_config['new_classes'][-1]), mode="val", batch_size=16)
        
        retain_acc = evaluate(model, retain_val_loader, device, 100)
        forget_acc = evaluate(model, forget_val_loader, device, 100)
        new_acc = evaluate(model, new_val_loader, device, 100)
        combined_acc = evaluate(model, combined_val_loader, device, 100)
        
        # Log final results
        final_results = {
            'retain_acc': retain_acc,
            'forget_acc': forget_acc,
            'new_acc': new_acc,
            'combined_acc': combined_acc,
            'step_losses': step_losses,
            'step_accuracies': step_accuracies
        }
        
        all_step_results[step_num] = final_results
        
        result_log = (f"✅ Retain: {retain_acc:.2f}% | Forget: {forget_acc:.2f}% | New: {new_acc:.2f}% | Combined: {combined_acc:.2f}%")
        print(result_log)
        print(f"💾 Saved: {step_config['output_ckpt']}")
    
    # Final summary
    print(f"\n🎉 Multi-Step CLU Complete! Final model: 50-99 classes")
    return all_step_results

def main():
    """Main function to run the multi-step CLU training"""
    
    # Setup minimal logging
    logger = setup_logging()
    
    try:
        # Run the multi-step training
        results = run_multi_step_clu(logger)
        
        # Print final summary
        print("\n" + "="*50)
        print("📊 FINAL SUMMARY")
        print("="*50)
        
        for step_num, step_results in results.items():
            print(f"Step {step_num}: {step_results['combined_acc']:.2f}%")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()