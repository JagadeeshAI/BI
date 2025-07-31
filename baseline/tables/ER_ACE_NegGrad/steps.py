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

class NegGradPlusUnlearner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def validate(self, forget_classes, retain_classes):
        """Validation on forget and retain sets"""
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

    def create_forget_dataset(self, forget_classes, batch_size=128):
        """Create forget dataset"""
        if not forget_classes:
            return None
            
        forget_loader = get_dynamic_loader(
            class_range=(min(forget_classes), max(forget_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        forget_data = []
        forget_labels = []
        max_samples_per_class = 500
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

    def create_retain_dataset(self, retain_classes, batch_size=128):
        """Create retain dataset"""
        if not retain_classes:
            return None
            
        retain_loader = get_dynamic_loader(
            class_range=(min(retain_classes), max(retain_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        retain_data = []
        retain_labels = []
        max_samples = 2000
        
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

    def neggrad_plus_loss(self, forget_output, forget_target, 
                          retain_output=None, retain_target=None, beta=0.5):
        """
        NegGrad+ loss following paper's exact formulation:
        L = β * L_retain - (1-β) * L_forget
        beta=0: pure NegGrad (only forget)
        beta=1: pure retain (like fine-tune)
        beta=0.5: balanced NegGrad+
        """
        criterion = nn.CrossEntropyLoss()
        
        # Forget loss (gradient ascent component)
        forget_loss = criterion(forget_output, forget_target)
        total_loss = -(1 - beta) * forget_loss  # Negative for gradient ascent
        
        # Retain loss (gradient descent component)
        if retain_output is not None and beta > 0:
            retain_loss = criterion(retain_output, retain_target)
            total_loss += beta * retain_loss
        
        return total_loss, forget_loss.item(), retain_loss.item() if retain_output is not None else 0

    def neggrad_plus_training_step(self, forget_loader, retain_loader, optimizer, 
                                   epoch, total_epochs, beta=0.5):
        """NegGrad+ training step with proper paper formulation"""
        self.model.train()
        retain_iter = iter(retain_loader) if retain_loader else None
        
        # TQDM for training batches
        pbar = tqdm(forget_loader if forget_loader else retain_loader, 
                   desc=f"NegGrad+ Training Epoch {epoch}/{total_epochs}")
        
        epoch_stats = {'total_loss': 0, 'forget_loss': 0, 'retain_loss': 0, 'batches': 0}
        
        for i, batch_data in enumerate(pbar):
            # Process forget data
            if forget_loader:
                forget_data, forget_target = batch_data
                forget_data = forget_data.to(self.device)
                forget_target = forget_target.to(self.device)
                forget_output = self.model(forget_data)
            else:
                forget_output, forget_target = None, None
            
            # Process retain data
            retain_output, retain_target = None, None
            if retain_iter and beta > 0:
                try:
                    retain_data, retain_target = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_data, retain_target = next(retain_iter)
                
                retain_data = retain_data.to(self.device)
                retain_target = retain_target.to(self.device)
                retain_output = self.model(retain_data)
            
            # Compute NegGrad+ loss
            total_loss, forget_loss_val, retain_loss_val = self.neggrad_plus_loss(
                forget_output, forget_target, retain_output, retain_target, beta=beta
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update statistics
            epoch_stats['total_loss'] += total_loss.item()
            epoch_stats['forget_loss'] += forget_loss_val
            epoch_stats['retain_loss'] += retain_loss_val
            epoch_stats['batches'] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_loss.item():.3f}',
                'Forget': f'{forget_loss_val:.3f}',
                'Retain': f'{retain_loss_val:.3f}' if retain_loss_val > 0 else 'N/A'
            })
        
        return epoch_stats

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

    def unlearn(self, forget_classes, retain_classes, epochs=40, beta=0.5):
        """Main NegGrad+ unlearning with best model selection"""
        method_name = "NegGrad+" if beta > 0 else "NegGrad"
        print(f"{method_name} Unlearning (β={beta}): Forget {forget_classes}, Retain {retain_classes}")
        
        forget_loader = self.create_forget_dataset(forget_classes)
        retain_loader = self.create_retain_dataset(retain_classes)
        
        # More aggressive optimizer settings as per paper suggestions
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.01,  # Higher learning rate for effective unlearning
            momentum=0.9, 
            weight_decay=1e-4  # Less regularization
        )
        
        best_score = float('-inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n{method_name} Unlearning Epoch {epoch+1}/{epochs}")
            
            # Training with proper NegGrad+ formulation
            stats = self.neggrad_plus_training_step(
                forget_loader, retain_loader, optimizer, epoch+1, epochs, beta=beta
            )
            
            # Print epoch statistics
            avg_total = stats['total_loss'] / stats['batches']
            avg_forget = stats['forget_loss'] / stats['batches']
            avg_retain = stats['retain_loss'] / stats['batches'] if stats['retain_loss'] > 0 else 0
            print(f"Loss - Total: {avg_total:.3f} | Forget: {avg_forget:.3f} | Retain: {avg_retain:.3f}")
            
            # Validate after each epoch
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            target_range = retain_classes  # The final target range for this step
            target_acc = self.validate_range(target_range)
            
            # Unlearning score: minimize forget accuracy, maximize retain accuracy
            # Following paper's evaluation: good unlearning = low forget acc + high retain acc
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 30)
            
            print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}% | Score: {unlearn_score:.1f}")
            
            # Save best model
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"★ New best model! Score: {unlearn_score:.1f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best unlearned model (Score: {best_score:.1f})")

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
    os.makedirs('steps_ace_neggrad', exist_ok=True)
    
    # Load base model (0-49)
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting ER-ACE + NegGrad+ pipeline with base model (classes 0-49)")
    
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
        
        # Phase 1: NegGrad+ Unlearning (with proper paper formulation)
        neggrad_unlearner = NegGradPlusUnlearner(model, device)
        
        # Initial validation before unlearning
        print("\nInitial validation before NegGrad+ unlearning:")
        forget_acc, retain_acc = neggrad_unlearner.validate(forget_classes, retain_classes)
        target_range = retain_classes + new_classes  # Final target range for this step
        target_acc = neggrad_unlearner.validate_range(target_range)
        print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # NegGrad+ unlearning with β=0.5 (balanced approach as per paper)
        neggrad_unlearner.unlearn(forget_classes, retain_classes, epochs=40, beta=0.5)
        
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
        checkpoint_path = f'steps_ace_neggrad/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_acc = er_ace.validate_epoch(current_classes)
        print(f"Final accuracy on classes {min(current_classes)}-{max(current_classes)}: {final_acc:.2f}%")

if __name__ == "__main__":
    main()