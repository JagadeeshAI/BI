import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

class SCRUBUnlearner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def create_validation_set(self, target_classes, batch_size=16):
        """Create validation set with same distribution as forget set"""
        val_loader = get_dynamic_loader(
            class_range=(target_classes[0], target_classes[-1]), 
            mode="val", 
            batch_size=batch_size,
            num_workers=0
        )
        
        val_data = []
        val_labels = []
        max_samples = 100  # Limit validation samples
        
        for batch_data, batch_labels in val_loader:
            for i, label in enumerate(batch_labels):
                if label.item() in target_classes and len(val_data) < max_samples:
                    val_data.append(batch_data[i])
                    val_labels.append(label.item())
            
            if len(val_data) >= max_samples:
                break
        
        if val_data:
            val_data = torch.stack(val_data)
            val_labels = torch.tensor(val_labels)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
            return torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0
            )
        return None
    
    def validate(self, target_classes, retain_classes):
        """Validate on both forgotten classes and retained classes"""
        self.model.eval()
        
        # Validate on forgotten classes (should have low accuracy)
        forget_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)), 
            mode="val", batch_size=16, num_workers=0
        )
        
        forget_correct = 0
        forget_total = 0
        
        with torch.no_grad():
            for data, target in forget_loader:
                data, target = data.to(self.device), target.to(self.device)
                # Only evaluate on target classes
                mask = torch.isin(target, torch.tensor(target_classes).to(self.device))
                if mask.any():
                    data_filtered = data[mask]
                    target_filtered = target[mask]
                    output = self.model(data_filtered)
                    pred = output.argmax(dim=1)
                    forget_correct += pred.eq(target_filtered).sum().item()
                    forget_total += target_filtered.size(0)
                
                del data, target
                torch.cuda.empty_cache()
        
        forget_acc = 100. * forget_correct / forget_total if forget_total > 0 else 0
        
        # Validate on retained classes (should have high accuracy)
        retain_acc = 0
        if retain_classes:
            retain_loader = get_dynamic_loader(
                class_range=(min(retain_classes), max(retain_classes)), 
                mode="val", batch_size=16, num_workers=0
            )
            
            retain_correct = 0
            retain_total = 0
            
            with torch.no_grad():
                for data, target in retain_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    # Only evaluate on retain classes
                    mask = torch.isin(target, torch.tensor(retain_classes).to(self.device))
                    if mask.any():
                        data_filtered = data[mask]
                        target_filtered = target[mask]
                        output = self.model(data_filtered)
                        pred = output.argmax(dim=1)
                        retain_correct += pred.eq(target_filtered).sum().item()
                        retain_total += target_filtered.size(0)
                    
                    del data, target
                    torch.cuda.empty_cache()
            
            retain_acc = 100. * retain_correct / retain_total if retain_total > 0 else 0
        
        return forget_acc, retain_acc
    
    def create_target_loader(self, class_range, target_classes, batch_size=16):
        """Create data loader for target classes"""
        full_loader = get_dynamic_loader(
            class_range=class_range, 
            mode="train", 
            batch_size=batch_size,
            num_workers=0
        )
        
        target_data = []
        target_labels = []
        max_samples_per_class = 50  # Limit samples per class
        class_counts = {cls: 0 for cls in target_classes}
        
        for batch_data, batch_labels in full_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if (label_item in target_classes and 
                    class_counts[label_item] < max_samples_per_class):
                    target_data.append(batch_data[i])
                    target_labels.append(label_item)
                    class_counts[label_item] += 1
            
            if all(count >= max_samples_per_class for count in class_counts.values()):
                break
        
        if target_data:
            target_data = torch.stack(target_data)
            target_labels = torch.tensor(target_labels)
            target_dataset = torch.utils.data.TensorDataset(target_data, target_labels)
            return torch.utils.data.DataLoader(
                target_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
        return None
    
    def create_retain_loader(self, class_range, target_classes, batch_size=16):
        """Create data loader for retain classes - use full retain data"""
        # Determine retain classes based on step
        retain_class_start = max(target_classes) + 1
        retain_class_end = class_range[1]
        
        if retain_class_start > retain_class_end:
            return None
            
        full_loader = get_dynamic_loader(
            class_range=(retain_class_start, retain_class_end), 
            mode="train", 
            batch_size=batch_size,
            num_workers=0
        )
        
        retain_data = []
        retain_labels = []
        max_samples = 1000  # Use more retain samples
        
        for batch_data, batch_labels in full_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if retain_class_start <= label_item <= retain_class_end and len(retain_data) < max_samples:
                    retain_data.append(batch_data[i])
                    retain_labels.append(label_item)
            
            if len(retain_data) >= max_samples:
                break
        
        if retain_data:
            retain_data = torch.stack(retain_data)
            retain_labels = torch.tensor(retain_labels)
            retain_dataset = torch.utils.data.TensorDataset(retain_data, retain_labels)
            return torch.utils.data.DataLoader(
                retain_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
        return None
    
    def kl_divergence_loss(self, student_logits, teacher_logits):
        """Compute KL divergence between student and teacher outputs"""
        student_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean')
    
    def scrub_step(self, forget_loader, retain_loader, teacher_model, optimizer, alpha=1.0, gamma=1.0):
        """Perform one SCRUB training step with MAXIMUM retain emphasis"""
        self.model.train()
        teacher_model.eval()
        
        total_loss = 0
        forget_losses = []
        retain_losses = []
        batch_count = 0
        
        pbar = tqdm(forget_loader, desc="SCRUB Training", leave=False)
        retain_iter = iter(retain_loader) if retain_loader else None
        
        for forget_data, forget_target in pbar:
            forget_data, forget_target = forget_data.to(self.device), forget_target.to(self.device)
            
            # Process MANY MORE retain batches for maximum retention
            retain_loss_total = 0
            retain_batches = 0
            
            for _ in range(8):  # Increased from 4 to 8 for MAXIMUM retain emphasis
                if retain_iter:
                    try:
                        retain_data, retain_target = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        retain_data, retain_target = next(retain_iter)
                    
                    retain_data, retain_target = retain_data.to(self.device), retain_target.to(self.device)
                    
                    with torch.no_grad():
                        teacher_output_retain = teacher_model(retain_data)
                    
                    student_output_retain = self.model(retain_data)
                    retain_kl_loss = self.kl_divergence_loss(student_output_retain, teacher_output_retain)
                    retain_ce_loss = F.cross_entropy(student_output_retain, retain_target)
                    
                    retain_loss_batch = alpha * retain_kl_loss + gamma * retain_ce_loss
                    retain_loss_total += retain_loss_batch
                    retain_batches += 1
                    
                    retain_losses.append(retain_loss_batch.item())
                    
                    del retain_data, retain_target, teacher_output_retain, student_output_retain
            
            # MAX-STEP: Maximize divergence on forget data (with MINIMAL weight)
            with torch.no_grad():
                teacher_output_forget = teacher_model(forget_data)
            
            student_output_forget = self.model(forget_data)
            forget_kl_loss = -self.kl_divergence_loss(student_output_forget, teacher_output_forget)
            
            # EXTREME retain prioritization
            forget_weight = 0.01  # Extremely small - reduced from 0.05
            retain_weight = 10.0  # Much larger - increased from 3.0
            
            loss = (forget_weight * forget_kl_loss + 
                   retain_weight * (retain_loss_total / retain_batches if retain_batches > 0 else 0))
            
            forget_losses.append(forget_kl_loss.item())
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            
            # Very gentle gradient clipping to preserve stability
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Enhanced progress info
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Forget': f'{forget_kl_loss.item():.4f}',
                'Retain': f'{(retain_loss_total/retain_batches if retain_batches > 0 else 0):.4f}',
                'GradNorm': f'{grad_norm:.3f}'
            })
            
            del forget_data, forget_target, teacher_output_forget, student_output_forget, loss
            torch.cuda.empty_cache()
        
        # Log average losses for monitoring
        avg_forget_loss = sum(forget_losses) / len(forget_losses) if forget_losses else 0
        avg_retain_loss = sum(retain_losses) / len(retain_losses) if retain_losses else 0
        
        print(f"    Forget Loss: {avg_forget_loss:.4f} | Retain Loss: {avg_retain_loss:.4f}")
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def unlearn_step(self, step_num, checkpoint_path, save_path, target_classes, 
                    lr=1e-3, epochs=40, alpha=2.0, gamma=2.0, use_rewind=False):
        """Perform SCRUB unlearning for a specific step - Fixed version with initial validation"""
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle state dict key mismatches
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping key {k} due to shape mismatch or missing key")
        
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Create teacher model (frozen copy)
        teacher_model = get_model(num_classes=100, use_lora=False, pretrained=False)
        teacher_model.to(self.device)
        teacher_model.load_state_dict(self.model.state_dict(), strict=False)
        teacher_model.eval()
        
        # Freeze teacher parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        del checkpoint
        torch.cuda.empty_cache()
        
        # Determine class range for current step
        step_ranges = {
            1: (0, 59),
            2: (0, 69), 
            3: (0, 79),
            4: (0, 89),
            5: (0, 99)
        }
        class_range = step_ranges[step_num]
        
        # Calculate retain classes correctly
        retain_class_start = max(target_classes) + 1
        retain_class_end = class_range[1]
        retain_classes = list(range(retain_class_start, retain_class_end + 1)) if retain_class_start <= retain_class_end else []
        
        print(f"Starting SCRUB unlearning for step {step_num}")
        print(f"Forget classes: {target_classes[0]}-{target_classes[-1]} (total: {len(target_classes)})")
        if retain_classes:
            print(f"Retain classes: {retain_classes[0]}-{retain_classes[-1]} (total: {len(retain_classes)})")
        else:
            print("No retain classes for this step")
        
        # INITIAL VALIDATION - Check baseline performance before training
        print(f"\n{'='*60}")
        print("🔍 INITIAL VALIDATION (Before Training)")
        print(f"{'='*60}")
        initial_forget_acc, initial_retain_acc = self.validate(target_classes, retain_classes)
        print(f'Initial Forget Acc: {initial_forget_acc:.2f}% (current performance on forget classes)')
        print(f'Initial Retain Acc: {initial_retain_acc:.2f}% (current performance on retain classes)')
        print(f"📊 We need: Forget Acc ↓ (from {initial_forget_acc:.2f}%) | Retain Acc ↑ (maintain ~{initial_retain_acc:.2f}%)")
        
        # Create data loaders
        forget_loader = self.create_target_loader(class_range, target_classes, batch_size=8)
        retain_loader = self.create_retain_loader(class_range, target_classes, batch_size=8)
        
        if forget_loader is None:
            print(f"❌ No samples found for target classes {target_classes}")
            return
        
        print(f"\n📦 Data loaders created:")
        print(f"   Forget loader: {len(forget_loader)} batches")
        print(f"   Retain loader: {len(retain_loader) if retain_loader else 0} batches")
        
        # Optimizer with conservative learning rate
        optimizer = optim.Adam(self.model.parameters(), lr=lr/2, weight_decay=1e-4)
        
        # Store best model state
        best_retain_acc = initial_retain_acc  # Start with initial retain accuracy
        best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}  # Save initial state
        
        # Retain-only warm-up phase (EXTENDED to 10 epochs for better stabilization)
        print(f"\n{'='*60}")
        print("🔥 EXTENDED RETAIN-ONLY WARM-UP PHASE (Epochs 1-10)")
        print(f"{'='*60}")
        
        for warmup_epoch in range(10):  # Extended from 5 to 10
            print(f"\nWarm-up Epoch {warmup_epoch+1}/10")
            
            if retain_loader:
                self.model.train()
                warmup_loss = 0
                warmup_batches = 0
                
                # Use much smaller learning rate for warmup
                warmup_optimizer = optim.Adam(self.model.parameters(), lr=lr/10, weight_decay=1e-5)
                
                for retain_data, retain_target in retain_loader:
                    retain_data, retain_target = retain_data.to(self.device), retain_target.to(self.device)
                    
                    # Get teacher output for KL regularization
                    with torch.no_grad():
                        teacher_output = teacher_model(retain_data)
                    
                    warmup_optimizer.zero_grad()
                    student_output = self.model(retain_data)
                    
                    # Combined loss: CE + KL to stay close to teacher
                    retain_ce_loss = F.cross_entropy(student_output, retain_target)
                    retain_kl_loss = self.kl_divergence_loss(student_output, teacher_output)
                    retain_loss = retain_ce_loss + 0.5 * retain_kl_loss  # Add KL regularization
                    
                    retain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # Very gentle
                    warmup_optimizer.step()
                    
                    warmup_loss += retain_loss.item()
                    warmup_batches += 1
                    
                    del retain_data, retain_target, teacher_output, student_output
                    torch.cuda.empty_cache()
                
                avg_warmup_loss = warmup_loss / warmup_batches if warmup_batches > 0 else 0
                
                # Check progress more frequently
                if warmup_epoch % 2 == 0 or warmup_epoch >= 7:  # Check every 2 epochs, then every epoch
                    forget_acc, retain_acc = self.validate(target_classes, retain_classes)
                    print(f'  Retain-only Loss: {avg_warmup_loss:.4f}')
                    print(f'  Forget Acc: {forget_acc:.2f}% | Retain Acc: {retain_acc:.2f}%')
                    
                    # Update best model if retain improved
                    if retain_acc > best_retain_acc:
                        best_retain_acc = retain_acc
                        best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        print(f'  ⭐ New best retain accuracy: {retain_acc:.2f}%')
                    
                    # Stop early if retain performance is stable and good
                    if warmup_epoch >= 5 and retain_acc > initial_retain_acc * 0.95:  # Within 5% of initial
                        print(f'  ✅ Retain performance stabilized, ending warmup early')
                        break
        
        # Main SCRUB training loop - NO EARLY STOPPING
        print(f"\n{'='*60}")
        print("🚀 MAIN SCRUB TRAINING PHASE")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            # Progressive weight adjustment - more conservative
            if epoch < 15:  # Extended warm-up
                current_alpha, current_gamma = alpha * 1.0, gamma * 1.0  # Start with full weights
            elif epoch < 30:
                current_alpha, current_gamma = alpha * 1.2, gamma * 1.2  # Slightly increase
            else:
                current_alpha, current_gamma = alpha * 1.5, gamma * 1.5  # Max increase
            
            # SCRUB training step
            avg_loss = self.scrub_step(forget_loader, retain_loader, teacher_model, 
                                     optimizer, alpha=current_alpha, gamma=current_gamma)
            
            # Validation after every epoch
            forget_acc, retain_acc = self.validate(target_classes, retain_classes)
            
            # Calculate progress indicators
            forget_improvement = initial_forget_acc - forget_acc  # Higher is better (more forgetting)
            retain_maintenance = retain_acc - initial_retain_acc   # Positive is better (retention)
            
            print(f'  Forget Acc: {forget_acc:.2f}% (↓{forget_improvement:+.2f}% from initial)')
            print(f'  Retain Acc: {retain_acc:.2f}% ({retain_maintenance:+.2f}% from initial)')
            print(f'  Avg Loss: {avg_loss:.4f}')
            print(f'  Weights: α={current_alpha:.2f}, γ={current_gamma:.2f}')
            
            # Track best retain accuracy
            if retain_acc > best_retain_acc:
                best_retain_acc = retain_acc
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f'  ⭐ New best retain accuracy: {retain_acc:.2f}%')
            
            # Log progress indicators
            if forget_improvement > 10:
                print(f'  ✅ Excellent forgetting progress (-{forget_improvement:.1f}%)')
            elif forget_improvement > 5:
                print(f'  ✅ Good forgetting progress (-{forget_improvement:.1f}%)')
            
            if retain_acc > 65:
                print(f'  ✅ Excellent retention ({retain_acc:.1f}%)')
            elif retain_acc > 55:
                print(f'  ✅ Good retention ({retain_acc:.1f}%)')
            elif retain_acc < 45:
                print(f'  ⚠️  Retention dropping ({retain_acc:.1f}%)')
        
        # Load best model state at the end
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n🎯 Loaded best model with retain accuracy: {best_retain_acc:.2f}%")
        
        # Final validation with detailed comparison
        print(f"\n{'='*60}")
        print("📋 FINAL VALIDATION & COMPARISON")
        print(f"{'='*60}")
        final_forget_acc, final_retain_acc = self.validate(target_classes, retain_classes)
        
        forget_total_improvement = initial_forget_acc - final_forget_acc
        retain_total_change = final_retain_acc - initial_retain_acc
        
        print(f'📊 FORGET PERFORMANCE:')
        print(f'   Initial: {initial_forget_acc:.2f}% → Final: {final_forget_acc:.2f}%')
        print(f'   Improvement: {forget_total_improvement:+.2f}% {"✅" if forget_total_improvement > 0 else "❌"}')
        
        print(f'📊 RETAIN PERFORMANCE:')
        print(f'   Initial: {initial_retain_acc:.2f}% → Final: {final_retain_acc:.2f}%')
        print(f'   Change: {retain_total_change:+.2f}% {"✅" if retain_total_change >= -5 else "❌"}')
        
        # Save unlearned model with comprehensive metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'target_classes': target_classes,
            'initial_forget_acc': initial_forget_acc,
            'initial_retain_acc': initial_retain_acc,
            'final_forget_acc': final_forget_acc,
            'final_retain_acc': final_retain_acc,
            'best_retain_acc': best_retain_acc,
            'forget_improvement': forget_total_improvement,
            'retain_change': retain_total_change,
            'method': 'SCRUB_Enhanced'
        }, save_path)
        
        print(f"✅ Saved enhanced SCRUB model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='SCRUB Unlearning for ViT on CIFAR-100')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace',
                        help='Directory containing step checkpoints')
    parser.add_argument('--output_dir', type=str, default='scrub_unlearned_models',
                        help='Directory to save unlearned models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ViT model
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    
    unlearner = SCRUBUnlearner(model, device=args.device)
    
    # Define unlearning tasks
    unlearn_tasks = [
        (1, 'step1.pth', 'final_step1.pth', list(range(0, 10))),    # forget 0-9
        (2, 'step2.pth', 'final_step2.pth', list(range(0, 20))),    # forget 0-19  
        (3, 'step3.pth', 'final_step3.pth', list(range(0, 30))),    # forget 0-29
        (4, 'step4.pth', 'final_step4.pth', list(range(0, 40))),    # forget 0-39
        (5, 'step5.pth', 'final_step5.pth', list(range(0, 50))),    # forget 0-49
    ]
    
    for step_num, checkpoint_file, output_file, target_classes in unlearn_tasks:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        save_path = os.path.join(args.output_dir, output_file)
        
        print(f"\n{'='*60}")
        print(f"SCRUB Unlearning Step {step_num}")
        print(f"Target classes to forget: {target_classes}")
        print(f"{'='*60}")
        
        unlearner.unlearn_step(
            step_num=step_num,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            target_classes=target_classes,
            lr=5e-4,  # Even more conservative learning rate
            epochs=args.epochs,  # Use configurable epochs (default 40)
            alpha=3.0,  # Even stronger retain emphasis
            gamma=3.0,  # Even stronger retain emphasis
            use_rewind=False  # Disabled for full training
        )

if __name__ == "__main__":
    main()