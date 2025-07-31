import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import copy
import random
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

class ReservoirBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.buffer = []
        self.current_size = 0
        
    def add_samples(self, samples, logits, labels):
        """Add samples using reservoir sampling"""
        for sample, logit, label in zip(samples, logits, labels):
            sample = sample.cpu()
            logit = logit.cpu().detach()
            label = label.item() if hasattr(label, 'item') else label
            
            if self.current_size < self.max_size:
                self.buffer.append((sample, logit, label))
            else:
                # Reservoir sampling
                idx = random.randint(0, self.current_size)
                if idx < self.max_size:
                    self.buffer[idx] = (sample, logit, label)
            self.current_size += 1
    
    def sample_batch(self, batch_size):
        """Sample random batch from buffer"""
        if len(self.buffer) == 0:
            return None, None, None
            
        indices = random.choices(range(len(self.buffer)), k=min(batch_size, len(self.buffer)))
        
        samples, logits, labels = [], [], []
        for idx in indices:
            sample, logit, label = self.buffer[idx]
            samples.append(sample)
            logits.append(logit)
            labels.append(label)
            
        return torch.stack(samples), torch.stack(logits), torch.tensor(labels)

class SCRUBDERUnlearner:
    def __init__(self, model, device='cuda', buffer_size=2000, alpha=0.5, beta=0.5):
        self.device = device
        self.buffer = ReservoirBuffer(max_size=buffer_size)
        self.alpha = alpha  # Logit distillation weight (SCRUB-style)
        self.beta = beta    # Buffer classification weight (DER++ style)
        self.model = model.to(device)
        
    def augment_data(self, data):
        """Apply data augmentation as mentioned in DER++ paper"""
        # Simple augmentation - you can expand this
        if random.random() > 0.5:
            # Random horizontal flip
            data = torch.flip(data, dims=[-1])
        return data

    def validate(self, forget_classes, retain_classes):
        """Validate on forget and retain sets"""
        self.model.eval()
        
        # Forget classes validation
        forget_acc = 0
        if forget_classes:
            forget_loader = get_dynamic_loader(
                class_range=(min(forget_classes), max(forget_classes)),
                mode="val", batch_size=128, num_workers=0
            )
            forget_correct = 0
            forget_total = 0
            
            with torch.no_grad():
                for data, target in forget_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    mask = torch.isin(target, torch.tensor(forget_classes).to(self.device))
                    if mask.any():
                        data_filtered = data[mask]
                        target_filtered = target[mask]
                        output = self.model(data_filtered)
                        pred = output.argmax(dim=1)
                        forget_correct += pred.eq(target_filtered).sum().item()
                        forget_total += target_filtered.size(0)
            
            forget_acc = 100. * forget_correct / forget_total if forget_total > 0 else 0

        # Retain classes validation  
        retain_acc = 0
        if retain_classes:
            retain_loader = get_dynamic_loader(
                class_range=(min(retain_classes), max(retain_classes)),
                mode="val", batch_size=128, num_workers=0
            )
            retain_correct = 0
            retain_total = 0
            
            with torch.no_grad():
                for data, target in retain_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    mask = torch.isin(target, torch.tensor(retain_classes).to(self.device))
                    if mask.any():
                        data_filtered = data[mask]
                        target_filtered = target[mask]
                        output = self.model(data_filtered)
                        pred = output.argmax(dim=1)
                        retain_correct += pred.eq(target_filtered).sum().item()
                        retain_total += target_filtered.size(0)
            
            retain_acc = 100. * retain_correct / retain_total if retain_total > 0 else 0

        return forget_acc, retain_acc

    def create_validation_set(self, target_classes, batch_size=16):
        """Create validation set with same distribution as forget set (for SCRUB+R)"""
        val_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)), 
            mode="val", 
            batch_size=batch_size,
            num_workers=0
        )
        
        val_data = []
        val_labels = []
        max_samples = 100
        
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

    def create_forget_dataset(self, forget_classes, batch_size=32):
        """Create forget dataset"""
        if not forget_classes:
            return None
            
        forget_loader = get_dynamic_loader(
            class_range=(min(forget_classes), max(forget_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        forget_data = []
        forget_labels = []
        max_samples_per_class = 200  # Limit for efficiency
        class_counts = {cls: 0 for cls in forget_classes}
        
        for batch_data, batch_labels in forget_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if (label_item in forget_classes and 
                    class_counts[label_item] < max_samples_per_class):
                    forget_data.append(batch_data[i])
                    forget_labels.append(label_item)
                    class_counts[label_item] += 1
            
            if all(count >= max_samples_per_class for count in class_counts.values()):
                break

        if forget_data:
            forget_data = torch.stack(forget_data)
            forget_labels = torch.tensor(forget_labels)
            forget_dataset = torch.utils.data.TensorDataset(forget_data, forget_labels)
            return torch.utils.data.DataLoader(
                forget_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
        return None

    def create_retain_dataset(self, retain_classes, batch_size=32):
        """Create retain dataset"""
        if not retain_classes:
            return None
            
        retain_loader = get_dynamic_loader(
            class_range=(min(retain_classes), max(retain_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        retain_data = []
        retain_labels = []
        max_samples = 1000  # More retain samples for stability
        
        for batch_data, batch_labels in retain_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if label_item in retain_classes and len(retain_data) < max_samples:
                    retain_data.append(batch_data[i])
                    retain_labels.append(label_item)
            if len(retain_data) >= max_samples:
                break

        if retain_data:
            retain_data = torch.stack(retain_data)
            retain_labels = torch.tensor(retain_labels)
            retain_dataset = torch.utils.data.TensorDataset(retain_data, retain_labels)
            return torch.utils.data.DataLoader(
                retain_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
        return None

    def kl_divergence_loss(self, student_logits, teacher_logits):
        """Compute KL divergence between student and teacher outputs"""
        return F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )

    def scrub_der_max_step(self, forget_loader, teacher_model, optimizer):
        """SCRUB MAX-STEP: Train only on forget data to maximize divergence"""
        self.model.train()
        teacher_model.eval()
        
        max_loss = 0
        batch_count = 0
        
        for forget_data, forget_target in forget_loader:
            forget_data = forget_data.to(self.device)
            forget_target = forget_target.to(self.device)
            
            # Apply augmentation (DER++ style)
            forget_data_aug = self.augment_data(forget_data)
            
            with torch.no_grad():
                teacher_output = teacher_model(forget_data_aug)
            
            student_output = self.model(forget_data_aug)
            
            # SCRUB: Maximize divergence from teacher (negative KL)
            max_loss_batch = -self.kl_divergence_loss(student_output, teacher_output)
            
            optimizer.zero_grad()
            max_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            max_loss += max_loss_batch.item()
            batch_count += 1
            
            # Store current logits in buffer (DER++ style)
            with torch.no_grad():
                current_logits = self.model(forget_data).detach()
                self.buffer.add_samples(forget_data, current_logits, forget_target)
        
        return max_loss / batch_count if batch_count > 0 else 0

    def scrub_der_min_step(self, retain_loader, teacher_model, optimizer):
        """SCRUB MIN-STEP: Train only on retain data to minimize divergence + DER++ buffer replay"""
        self.model.train()
        teacher_model.eval()
        
        min_loss = 0
        batch_count = 0
        
        for retain_data, retain_target in retain_loader:
            retain_data = retain_data.to(self.device)
            retain_target = retain_target.to(self.device)
            
            # Apply augmentation (DER++ style)
            retain_data_aug = self.augment_data(retain_data)
            
            with torch.no_grad():
                teacher_output = teacher_model(retain_data_aug)
            
            student_output = self.model(retain_data_aug)
            
            # SCRUB: Minimize divergence from teacher + classification loss
            retain_kl_loss = self.alpha * self.kl_divergence_loss(student_output, teacher_output)
            retain_ce_loss = F.cross_entropy(student_output, retain_target)
            
            total_loss = retain_kl_loss + retain_ce_loss
            
            # DER++ Buffer replay component
            buffer_data, buffer_logits, buffer_labels = self.buffer.sample_batch(32)
            if buffer_data is not None:
                buffer_data = buffer_data.to(self.device)
                buffer_logits = buffer_logits.to(self.device)
                buffer_labels = buffer_labels.to(self.device)
                
                # Apply augmentation to buffer data
                buffer_data_aug = self.augment_data(buffer_data)
                
                # Forward pass on buffer data
                buffer_output = self.model(buffer_data_aug)
                
                # DER++: Logit distillation + classification on buffer
                buffer_distill_loss = self.alpha * F.mse_loss(buffer_output, buffer_logits)
                buffer_class_loss = self.beta * F.cross_entropy(buffer_output, buffer_labels)
                
                total_loss += buffer_distill_loss + buffer_class_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store current batch in buffer
            with torch.no_grad():
                current_logits = self.model(retain_data).detach()
                self.buffer.add_samples(retain_data, current_logits, retain_target)
            
            min_loss += total_loss.item()
            batch_count += 1
        
        return min_loss / batch_count if batch_count > 0 else 0

    def populate_buffer_initial(self, retain_classes):
        """Populate buffer initially with retain class samples"""
        if not retain_classes:
            return
            
        retain_loader = get_dynamic_loader(
            class_range=(min(retain_classes), max(retain_classes)), 
            mode="train", batch_size=32
        )
        
        self.model.eval()
        samples_added = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in retain_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Get logits for storage
                logits = self.model(batch_data)
                
                # Add to buffer
                self.buffer.add_samples(batch_data, logits, batch_labels)
                samples_added += len(batch_data)
                
                if samples_added >= 500:  # Limit initial population
                    break
        
        print(f"Buffer populated with {len(self.buffer.buffer)} samples from retain classes")

    def rewind_to_best_checkpoint(self, checkpoints, val_loader, target_error):
        """SCRUB+R: Rewind to checkpoint with error closest to target"""
        if not checkpoints or not val_loader:
            return False
            
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
            return True
        return False

    def unlearn(self, forget_classes, retain_classes, epochs=20, max_steps=3, min_steps=5, use_rewind=False):
        """Main SCRUB+DER++ unlearning with alternating min-max steps"""
        print(f"SCRUB+DER++ Unlearning: Forget {forget_classes}, Retain {retain_classes}")
        
        # Create teacher model (frozen copy of current model)
        teacher_model = get_model(num_classes=100, use_lora=False, pretrained=False)
        teacher_model.to(self.device)
        teacher_model.load_state_dict(self.model.state_dict())
        teacher_model.eval()
        
        # Freeze teacher parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Create datasets
        forget_loader = self.create_forget_dataset(forget_classes, batch_size=16)
        retain_loader = self.create_retain_dataset(retain_classes, batch_size=16)
        
        if forget_loader is None:
            print(f"No samples found for forget classes {forget_classes}")
            return
        
        # Populate buffer initially
        self.populate_buffer_initial(retain_classes)
        
        # Create validation set for rewinding
        val_loader = None
        if use_rewind:
            val_loader = self.create_validation_set(forget_classes)
        
        # Optimizer with low learning rate for stability
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        # Store checkpoints for rewinding
        checkpoints = []
        best_score = float('-inf')
        best_model_state = None
        
        print(f"Starting SCRUB+DER++ training with {epochs} epochs")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # SCRUB alternating min-max optimization
            for step in range(max(max_steps, min_steps)):
                if step < max_steps and forget_loader:
                    # MAX-STEP: Maximize divergence on forget data
                    max_loss = self.scrub_der_max_step(forget_loader, teacher_model, optimizer)
                    print(f"  MAX-STEP {step+1}: Loss = {max_loss:.4f}")
                
                if step < min_steps and retain_loader:
                    # MIN-STEP: Minimize divergence on retain data + DER++ buffer replay
                    min_loss = self.scrub_der_min_step(retain_loader, teacher_model, optimizer)
                    print(f"  MIN-STEP {step+1}: Loss = {min_loss:.4f}")
            
            # Additional min-steps for stability (as per SCRUB paper)
            for extra_step in range(2):
                if retain_loader:
                    min_loss = self.scrub_der_min_step(retain_loader, teacher_model, optimizer)
                    print(f"  EXTRA-MIN-STEP {extra_step+1}: Loss = {min_loss:.4f}")
            
            # Validation
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            
            # Unlearning score: balance forgetting and retention
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 40)
            
            print(f"  Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | Score: {unlearn_score:.1f}")
            
            # Store checkpoint
            checkpoints.append({
                'epoch': epoch,
                'state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                'forget_acc': forget_acc,
                'retain_acc': retain_acc,
                'score': unlearn_score
            })
            
            # Track best model
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  ★ New best model! Score: {unlearn_score:.1f}")
            
            # Early stopping conditions
            if forget_acc < 5 and retain_acc > 70:
                print(f"  ✅ Excellent unlearning achieved at epoch {epoch+1}")
                break
            
            if epoch > 10 and retain_acc < 20:
                print(f"  ⚠️ Retain accuracy too low, stopping at epoch {epoch+1}")
                break
        
        # Apply rewinding if requested
        if use_rewind and val_loader and checkpoints:
            print("\nPerforming SCRUB+R rewinding...")
            
            # Get reference error from final state
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
            
            if self.rewind_to_best_checkpoint(checkpoints, val_loader, target_error):
                print(f"Rewound to checkpoint with error closest to {target_error:.2f}%")
        elif best_model_state is not None:
            # Load best model if not rewinding
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model (Score: {best_score:.1f})")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    os.makedirs('scrub_der_models', exist_ok=True)
    
    # Load base model
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting SCRUB+DER++ Unlearning with base model (classes 0-49)")
    
    # Define unlearning steps
    steps = [
        (1, list(range(0, 10)), list(range(10, 50))),   # Forget 0-9, retain 10-49
        (2, list(range(10, 20)), list(range(20, 50))),  # Forget 10-19, retain 20-49  
        (3, list(range(20, 30)), list(range(30, 50))),  # Forget 20-29, retain 30-49
        (4, list(range(30, 40)), list(range(40, 50))),  # Forget 30-39, retain 40-49
        (5, list(range(40, 50)), [])                    # Forget 40-49, retain none
    ]
    
    for step_num, forget_classes, retain_classes in steps:
        print(f"\n{'='*80}")
        print(f"SCRUB+DER++ Step {step_num}")
        print(f"Forget classes: {forget_classes}")
        print(f"Retain classes: {retain_classes}")
        print(f"{'='*80}")
        
        # Initialize unlearner
        unlearner = SCRUBDERUnlearner(
            model, 
            device=device, 
            buffer_size=1000,
            alpha=0.5,  # KL divergence weight
            beta=0.5    # Buffer classification weight
        )
        
        # Perform unlearning
        unlearner.unlearn(
            forget_classes=forget_classes,
            retain_classes=retain_classes,
            epochs=15,
            max_steps=2,     # SCRUB max-steps per epoch
            min_steps=3,     # SCRUB min-steps per epoch  
            use_rewind=False # Set to True for SCRUB+R
        )
        
        # Save checkpoint
        checkpoint_path = f'scrub_der_models/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_forget_acc, final_retain_acc = unlearner.validate(forget_classes, retain_classes)
        print(f"Final - Forget: {final_forget_acc:.1f}% | Retain: {final_retain_acc:.1f}%")

if __name__ == "__main__":
    main()