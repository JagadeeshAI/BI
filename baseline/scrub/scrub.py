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
    
    def scrub_step(self, forget_loader, retain_loader, teacher_model, optimizer, alpha=2.0, gamma=2.0):
        """Perform one SCRUB training step with balanced forget/retain batches"""
        self.model.train()
        teacher_model.eval()
        
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(forget_loader, desc="SCRUB Training", leave=False)
        retain_iter = iter(retain_loader) if retain_loader else None
        
        for forget_data, forget_target in pbar:
            forget_data, forget_target = forget_data.to(self.device), forget_target.to(self.device)
            
            # Process multiple retain batches per forget batch for stability
            retain_loss_total = 0
            retain_batches = 0
            
            for _ in range(3):  # 3 retain batches per 1 forget batch
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
                    
                    retain_loss_total += alpha * retain_kl_loss + gamma * retain_ce_loss
                    retain_batches += 1
                    
                    del retain_data, retain_target, teacher_output_retain, student_output_retain
            
            # MAX-STEP: Maximize divergence on forget data (reduced weight)
            with torch.no_grad():
                teacher_output_forget = teacher_model(forget_data)
            
            student_output_forget = self.model(forget_data)
            forget_kl_loss = -self.kl_divergence_loss(student_output_forget, teacher_output_forget)
            
            # Combined loss with much stronger retain emphasis
            loss = 0.2 * forget_kl_loss + (retain_loss_total / retain_batches if retain_batches > 0 else 0)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            del forget_data, forget_target, teacher_output_forget, student_output_forget, loss
            torch.cuda.empty_cache()
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def rewind_to_best_checkpoint(self, checkpoints, val_loader, target_error):
        """Rewind to checkpoint with error closest to target"""
        best_checkpoint = None
        best_diff = float('inf')
        
        for checkpoint_data in checkpoints:
            self.model.load_state_dict(checkpoint_data['state_dict'])
            
            # Evaluate on validation set
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            error = 100. * (1 - correct / total) if total > 0 else 100.0
            diff = abs(error - target_error)
            
            if diff < best_diff:
                best_diff = diff
                best_checkpoint = checkpoint_data
        
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['state_dict'])
        
        return best_checkpoint is not None
    
    def unlearn_step(self, step_num, checkpoint_path, save_path, target_classes, 
                    lr=1e-3, epochs=20, alpha=1.0, gamma=1.0, use_rewind=True):
        """Perform SCRUB unlearning for a specific step"""
        
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
        
        # Create data loaders
        forget_loader = self.create_target_loader(class_range, target_classes, batch_size=8)
        retain_loader = self.create_retain_loader(class_range, target_classes, batch_size=8)
        
        if forget_loader is None:
            print(f"No samples found for target classes {target_classes}")
            return
        
        # Create validation set for rewinding
        val_loader = None
        if use_rewind:
            val_loader = self.create_validation_set(target_classes, batch_size=16)
        
        print(f"Starting SCRUB unlearning for step {step_num}")
        
        # Optimizer with very low learning rate for stability
        optimizer = optim.Adam(self.model.parameters(), lr=lr/10, weight_decay=1e-4)
        
        # Store checkpoints for rewinding
        checkpoints = []
        best_retain_acc = 0
        best_checkpoint_epoch = 0
        
        # Define validation ranges - compute retain classes properly
        retain_classes = []
        if step_num == 1:
            retain_classes = list(range(10, 60))
        elif step_num == 2:
            retain_classes = list(range(20, 70))
        elif step_num == 3:
            retain_classes = list(range(30, 80))
        elif step_num == 4:
            retain_classes = list(range(40, 90))
        elif step_num == 5:
            retain_classes = list(range(50, 100))
        
        # Main training loop with proper progress tracking
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # SCRUB training step
            avg_loss = self.scrub_step(forget_loader, retain_loader, teacher_model, 
                                     optimizer, alpha=3.0, gamma=3.0)  # Higher retain weights
            
            # Validation after every epoch
            forget_acc, retain_acc = self.validate(target_classes, retain_classes)
            
            print(f'  Forget Acc: {forget_acc:.2f}% (target: low, classes {target_classes[0]}-{target_classes[-1]})')
            print(f'  Retain Acc: {retain_acc:.2f}% (target: high, classes {retain_classes[0] if retain_classes else "N/A"}-{retain_classes[-1] if retain_classes else "N/A"})')
            print(f'  Avg Loss: {avg_loss:.4f}')
            
            # Store checkpoint and track best retain accuracy
            checkpoints.append({
                'epoch': epoch,
                'state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                'retain_acc': retain_acc,
                'forget_acc': forget_acc
            })
            
            if retain_acc > best_retain_acc:
                best_retain_acc = retain_acc
                best_checkpoint_epoch = epoch
            
            # Early stopping if good balance achieved
            if forget_acc < 10 and retain_acc > 60:
                print(f"✅ Good unlearning balance achieved at epoch {epoch+1}")
                break
            
            # Stop if retain accuracy drops too much
            if epoch > 5 and retain_acc < 25:
                print(f"⚠️ Retain accuracy too low, stopping at epoch {epoch+1}")
                break
        
        # Use checkpoint with best retain accuracy instead of rewinding
        if checkpoints and not use_rewind:
            best_checkpoint = checkpoints[best_checkpoint_epoch]
            self.model.load_state_dict(best_checkpoint['state_dict'])
            print(f"Loaded checkpoint from epoch {best_checkpoint_epoch + 1} with retain acc: {best_checkpoint['retain_acc']:.2f}%")
        elif use_rewind and val_loader and checkpoints:
            print("Performing rewinding to optimize forget error...")
            
            # Get reference error from final state on validation set
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            target_error = 100. * (1 - val_correct / val_total) if val_total > 0 else 100.0
            
            # Rewind to best checkpoint
            if self.rewind_to_best_checkpoint(checkpoints, val_loader, target_error):
                print(f"Rewound to checkpoint with error closest to {target_error:.2f}%")
        
        # Save unlearned model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'target_classes': target_classes,
            'method': 'SCRUB'
        }, save_path)
        
        print(f"Saved SCRUB unlearned model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='SCRUB Unlearning for ViT on CIFAR-100')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace',
                        help='Directory containing step checkpoints')
    parser.add_argument('--output_dir', type=str, default='scrub_unlearned_models',
                        help='Directory to save unlearned models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--use_rewind', action='store_true', default=True,
                        help='Use SCRUB+R (rewinding) variant')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ViT model
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    
    unlearner = SCRUBUnlearner(model, device=args.device)
    
    # Define unlearning tasks to match your requirements
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
            lr=1e-3,  # Lower learning rate for SCRUB
            epochs=25,
            alpha=3.0,  # Higher weight for retain KL loss
            gamma=3.0,  # Higher weight for retain CE loss
            use_rewind=False  # Disable rewinding by default
        )

if __name__ == "__main__":
    main()