from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import copy
import random
from codes.data import get_dynamic_loader
from codes.utils import get_model

class MemoryBuffer:
    def __init__(self, max_size_per_class=20):
        self.max_size_per_class = max_size_per_class
        self.buffer = {}
        
    def add_samples(self, samples, labels):
        for sample, label in zip(samples, labels):
            label = label.item() if hasattr(label, 'item') else label
            if label not in self.buffer:
                self.buffer[label] = []
            
            if len(self.buffer[label]) < self.max_size_per_class:
                self.buffer[label].append(sample.cpu())
            else:
                # Reservoir sampling
                idx = random.randint(0, len(self.buffer[label]))
                if idx < self.max_size_per_class:
                    self.buffer[label][idx] = sample.cpu()
    
    def sample_batch(self, batch_size, classes):
        samples, labels = [], []
        available_classes = [c for c in classes if c in self.buffer and len(self.buffer[c]) > 0]
        
        if not available_classes:
            return None, None
            
        for _ in range(batch_size):
            cls = random.choice(available_classes)
            sample = random.choice(self.buffer[cls])
            samples.append(sample)
            labels.append(cls)
            
        return torch.stack(samples), torch.tensor(labels)

class L2ULUnlearner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def compute_weight_importance(self, data_loader, target_classes):
        """Compute weight importance using MAS with memory optimization"""
        self.model.eval()
        importance = {}
        
        # Initialize importance dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)
        
        total_samples = 0
        max_samples = 100 
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if total_samples >= max_samples:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Only process samples from target classes
            mask = torch.isin(target, torch.tensor(target_classes).to(self.device))
            if not mask.any():
                continue
                
            data_filtered = data[mask]
            batch_size = min(data_filtered.size(0), max_samples - total_samples)
            
            # Process in smaller chunks to save memory
            chunk_size = 4
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_data = data_filtered[i:end_idx]
                
                for j in range(chunk_data.size(0)):
                    if total_samples >= max_samples:
                        break
                        
                    single_data = chunk_data[j:j+1]
                    single_data.requires_grad = True
                    
                    output = self.model(single_data)
                    loss = torch.norm(output, p=2, dim=1).sum()
                    
                    grads = torch.autograd.grad(loss, self.model.parameters(), 
                                              retain_graph=False, create_graph=False)
                    
                    for (name, param), grad in zip(self.model.named_parameters(), grads):
                        if param.requires_grad:
                            importance[name] += torch.abs(grad)
                    
                    total_samples += 1
                    
                    # Clear gradients and cache
                    del grads, loss, output
                    torch.cuda.empty_cache()
        
        # Normalize by number of samples
        for name in importance:
            if total_samples > 0:
                importance[name] /= total_samples
            # Normalize to [0, 1] and invert (1 - importance)
            max_imp = importance[name].max()
            if max_imp > 0:
                importance[name] = 1 - (importance[name] / max_imp)
        
        return importance
    
    def validate(self, target_classes, val_range):
        """Validate on both forgotten classes and retained classes"""
        self.model.eval()
        
        # Validate on forgotten classes (should have low accuracy)
        forget_loader = get_dynamic_loader(
            class_range=(target_classes[0], target_classes[-1]), 
            mode="val", batch_size=16, num_workers=0
        )
        
        forget_correct = 0
        forget_total = 0
        
        with torch.no_grad():
            for data, target in forget_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                forget_correct += pred.eq(target).sum().item()
                forget_total += target.size(0)
                
                # Clear cache after each batch
                del output, pred
                torch.cuda.empty_cache()
        
        forget_acc = 100. * forget_correct / forget_total if forget_total > 0 else 0
        
        # Validate on retained classes (should have high accuracy)
        retain_acc = 0
        if val_range[0] <= val_range[1]:
            retain_loader = get_dynamic_loader(
                class_range=val_range, 
                mode="val", batch_size=16, num_workers=0
            )
            
            retain_correct = 0
            retain_total = 0
            
            with torch.no_grad():
                for data, target in retain_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    retain_correct += pred.eq(target).sum().item()
                    retain_total += target.size(0)
                    
                    # Clear cache after each batch
                    del output, pred
                    torch.cuda.empty_cache()
            
            retain_acc = 100. * retain_correct / retain_total if retain_total > 0 else 0
        
        return forget_acc, retain_acc
    
    def create_target_loader(self, class_range, target_classes, batch_size=16):
        """Create data loader for target classes with memory optimization"""
        full_loader = get_dynamic_loader(
            class_range=class_range, 
            mode="train", 
            batch_size=batch_size,
            num_workers=0
        )
        
        target_data = []
        target_labels = []
        max_samples_per_class = 50
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
        else:
            return None

    def unlearn(self, forget_classes, retain_classes, class_range, lr=1e-3, epochs=50, weight_decay=1e-5):
        """Main L2UL unlearning with memory optimization"""
        print(f"L2UL Unlearning: Forget {forget_classes}, Retain {retain_classes}")
        
        # Create target loader
        target_loader = self.create_target_loader(class_range, forget_classes, batch_size=16)
        
        if target_loader is None:
            print(f"No samples found for target classes {forget_classes}")
            return
        
        print("Computing weight importance")
        weight_importance = self.compute_weight_importance(target_loader, forget_classes)
        
        # Clear cache after importance computation
        torch.cuda.empty_cache()
        
        # Optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, 
                             weight_decay=weight_decay)
        
        # Store original weights for regularization
        original_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_weights[name] = param.data.clone()
        
        # Define validation ranges
        val_range = (forget_classes[-1] + 1, class_range[1])
        
        best_score = float('-inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nL2UL Unlearning Epoch {epoch+1}/{epochs}")
            self.model.train()
            total_loss = 0
            
            # Training with tqdm
            pbar = tqdm(target_loader, desc=f"L2UL Training Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Misclassification loss
                unlearn_loss = -nn.CrossEntropyLoss()(output, target)
                unlearn_loss = torch.clamp(unlearn_loss, min=-1000.0, max=0.0)
                
                # Regularization using remaining classes
                reg_loss = 0
                if retain_classes and val_range[0] <= val_range[1]:
                    try:
                        reg_loader = get_dynamic_loader(
                            class_range=val_range, 
                            mode="train", 
                            batch_size=16,
                            num_workers=0
                        )
                        reg_data, reg_target = next(iter(reg_loader))
                        reg_data, reg_target = reg_data.to(self.device), reg_target.to(self.device)
                        reg_output = self.model(reg_data)
                        reg_loss = nn.CrossEntropyLoss()(reg_output, reg_target)
                        
                        del reg_data, reg_target, reg_output
                    except:
                        pass
                
                # Weight importance regularization
                importance_loss = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in weight_importance:
                        importance_loss += torch.sum(weight_importance[name] * 
                                                   (param - original_weights[name]) ** 2)
                
                # Combined loss
                total_loss_batch = unlearn_loss + 2.0 * reg_loss + 0.5 * importance_loss
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                
                # Clear variables and cache
                del output, unlearn_loss, total_loss_batch
                if 'importance_loss' in locals():
                    del importance_loss
                torch.cuda.empty_cache()
                
                pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}'})
            
            # Validation
            forget_acc, retain_acc = self.validate(forget_classes, val_range)
            target_range = retain_classes  # Target range for this step
            target_acc = self.validate_range(target_range)
            
            # Unlearning score: minimize forget accuracy, maximize retain accuracy
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 30)
            
            avg_loss = total_loss / len(target_loader)
            print(f"Validation - Loss: {avg_loss:.4f} | Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}% | Score: {unlearn_score:.1f}")
            
            # Save best model
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"★ New best model! Score: {unlearn_score:.1f}")
            
            # Early stopping
            if forget_acc < 5:
                print(f"✅ Unlearning successful at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best unlearned model (Score: {best_score:.1f})")

    def validate_range(self, class_range):
        """Validate on specific class range"""
        val_loader = get_dynamic_loader(
            class_range=(min(class_range), max(class_range)), 
            mode="val", batch_size=128, num_workers=0
        )
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                mask = torch.isin(target, torch.tensor(class_range).to(self.device))
                if mask.any():
                    data_filtered = data[mask]
                    target_filtered = target[mask]
                    output = self.model(data_filtered)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target_filtered).sum().item()
                    total += target_filtered.size(0)
        
        return 100. * correct / total if total > 0 else 0

class ViTERACE:
    def __init__(self, model, device='cuda'):
        self.device = device
        self.memory_buffer = MemoryBuffer(max_size_per_class=20)
        self.seen_classes = []
        self.model = model.to(device)
    
    def er_ace_loss(self, incoming_logits, incoming_targets, replay_logits=None, replay_targets=None):
        """ER-ACE asymmetric loss"""
        # For incoming data: mask to only allow current step classes
        mask_incoming = torch.zeros_like(incoming_logits, dtype=bool)
        current_step_classes = incoming_targets.unique()
        mask_incoming[:, current_step_classes] = True
        
        # Apply mask by setting unwanted logits to -inf
        masked_logits_incoming = incoming_logits.masked_fill(~mask_incoming, -1e9)
        loss_incoming = F.cross_entropy(masked_logits_incoming, incoming_targets)
        
        # For replay data: use all seen classes
        if replay_logits is not None and replay_targets is not None:
            mask_replay = torch.zeros_like(replay_logits, dtype=bool)
            mask_replay[:, self.seen_classes] = True
            masked_logits_replay = replay_logits.masked_fill(~mask_replay, -1e9)
            loss_replay = F.cross_entropy(masked_logits_replay, replay_targets)
            return loss_incoming + loss_replay
        
        return loss_incoming
    
    def validate_epoch(self, val_classes):
        """Validate on specified classes"""
        val_loader = get_dynamic_loader(
            class_range=(min(val_classes), max(val_classes)), 
            mode="val", batch_size=64
        )
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                
                # Mask outputs to only consider seen classes
                mask = torch.zeros_like(outputs, dtype=bool)
                mask[:, val_classes] = True
                masked_outputs = outputs.masked_fill(~mask, -1e9)
                
                _, predicted = torch.max(masked_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0
    
    def validate_range(self, class_range):
        """Validate on specific class range"""
        val_loader = get_dynamic_loader(
            class_range=(min(class_range), max(class_range)), 
            mode="val", batch_size=128, num_workers=0
        )
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                mask = torch.isin(target, torch.tensor(class_range).to(self.device))
                if mask.any():
                    data_filtered = data[mask]
                    target_filtered = target[mask]
                    output = self.model(data_filtered)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target_filtered).sum().item()
                    total += target_filtered.size(0)
        
        return 100. * correct / total if total > 0 else 0
    
    def populate_buffer(self, classes):
        """Populate buffer with samples from classes"""
        train_loader = get_dynamic_loader(
            class_range=(min(classes), max(classes)), 
            mode="train", batch_size=32
        )
        
        class_counts = {i: 0 for i in classes}
        
        for batch_data, batch_labels in train_loader:
            for sample, label in zip(batch_data, batch_labels):
                label = label.item() if hasattr(label, 'item') else label
                if label in classes and class_counts[label] < 20:
                    self.memory_buffer.add_samples([sample], [label])
                    class_counts[label] += 1
            
            if all(count >= 20 for count in class_counts.values()):
                break
    
    def learn_classes(self, new_classes, epochs=5):
        """Learn new classes using ER-ACE"""
        print(f"ER-ACE Learning: Adding classes {new_classes}")
        
        # Update seen classes
        self.seen_classes.extend(new_classes)
        
        # Setup optimizer with differentiated learning rates
        optimizer = torch.optim.Adam([
            {'params': self.model.head.parameters(), 'lr': 1e-4},
            {'params': [p for n, p in self.model.named_parameters() if 'head' not in n], 'lr': 1e-5}
        ])
        
        # Load new class data
        new_class_loader = get_dynamic_loader(
            class_range=(min(new_classes), max(new_classes)), 
            mode="train", batch_size=32
        )
        
        self.model.train()
        
        best_score = float('-inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nER-ACE Learning Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            # TQDM for training batches
            pbar = tqdm(new_class_loader, desc=f"ER-ACE Training Epoch {epoch+1}/{epochs}")
            
            for batch_data, batch_labels in pbar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                # Forward pass for incoming data
                incoming_logits = self.model(batch_data)
                
                # Get replay data from old classes
                old_classes = [c for c in self.seen_classes if c not in new_classes]
                replay_data, replay_labels = self.memory_buffer.sample_batch(32, old_classes)
                
                if replay_data is not None:
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    replay_logits = self.model(replay_data)
                    
                    # ER-ACE loss
                    loss = self.er_ace_loss(incoming_logits, batch_labels, replay_logits, replay_labels)
                else:
                    # Only incoming loss for first few batches
                    loss = self.er_ace_loss(incoming_logits, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Add current batch to memory buffer
                self.memory_buffer.add_samples(batch_data.cpu(), batch_labels.cpu())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation
            new_acc = self.validate_epoch(new_classes)
            old_classes = [c for c in self.seen_classes if c not in new_classes]
            retain_acc = self.validate_epoch(old_classes) if old_classes else 0
            target_range = self.seen_classes  # Current target range (retain + new)
            target_acc = self.validate_range(target_range)
            
            # Learning score: balance new learning and retention
            learn_score = 0.6 * new_acc + 0.4 * retain_acc
            
            avg_loss = total_loss / num_batches
            print(f"Validation - Loss: {avg_loss:.4f} | New: {new_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}% | Score: {learn_score:.1f}")
            
            # Save best model
            if learn_score > best_score:
                best_score = learn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"★ New best model! Score: {learn_score:.1f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best learned model (Score: {best_score:.1f})")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create steps directory
    os.makedirs('steps_ace_l2ul', exist_ok=True)
    
    # Load base model (0-49)
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting ER-ACE + L2UL pipeline with base model (classes 0-49)")
    
    # Pipeline for each step
    steps = [
        (1, list(range(0, 10)), list(range(10, 50)), list(range(50, 60))),  # 0-49 -> 10-59
        (2, list(range(10, 20)), list(range(20, 60)), list(range(60, 70))), # 10-59 -> 20-69
        (3, list(range(20, 30)), list(range(30, 70)), list(range(70, 80))), # 20-69 -> 30-79
        (4, list(range(30, 40)), list(range(40, 80)), list(range(80, 90))), # 30-79 -> 40-89
        (5, list(range(40, 50)), list(range(50, 90)), list(range(90, 100)))  # 40-89 -> 50-99
    ]
    
    current_classes = list(range(0, 50))  # Start with 0-49
    
    for step_num, forget_classes, retain_classes, new_classes in steps:
        print(f"\n{'='*80}")
        print(f"Step {step_num}: {min(current_classes)}-{max(current_classes)} -> {min(retain_classes + new_classes)}-{max(retain_classes + new_classes)}")
        print(f"{'='*80}")
        
        # Phase 1: L2UL Unlearning
        l2ul_unlearner = L2ULUnlearner(model, device)
        
        # Define class ranges per step
        step_ranges = {1: (0, 59), 2: (0, 69), 3: (0, 79), 4: (0, 89), 5: (0, 99)}
        class_range = step_ranges[step_num]
        
        # Initial validation before unlearning
        print("\nInitial validation before L2UL unlearning:")
        forget_acc, retain_acc = l2ul_unlearner.validate(forget_classes, (retain_classes[0], retain_classes[-1]))
        target_range = retain_classes + new_classes  # Final target range for this step
        target_acc = l2ul_unlearner.validate_range(target_range)
        print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # L2UL unlearning
        l2ul_unlearner.unlearn(forget_classes, retain_classes, class_range, 
                              lr=1e-3, epochs=20, weight_decay=1e-5)
        
        # Phase 2: ER-ACE Learning
        er_ace = ViTERACE(model, device)
        er_ace.seen_classes = retain_classes.copy()
        
        # Populate buffer with retained classes
        er_ace.populate_buffer(retain_classes)
        
        # Initial validation before learning new classes
        print(f"\nInitial validation before ER-ACE learning:")
        retain_acc = er_ace.validate_epoch(retain_classes)
        target_acc = er_ace.validate_range(target_range)
        print(f"Validation - Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # Learn new classes with ER-ACE
        er_ace.learn_classes(new_classes, epochs=5)
        
        # Update current classes
        current_classes = retain_classes + new_classes
        
        # Save checkpoint
        checkpoint_path = f'steps_ace_l2ul/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_acc = er_ace.validate_epoch(current_classes)
        print(f"Final accuracy on classes {min(current_classes)}-{max(current_classes)}: {final_acc:.2f}%")

if __name__ == "__main__":
    main()