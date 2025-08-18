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

class GAUnlearner:
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

    def ga_training_step(self, forget_loader, retain_loader, optimizer, epoch, total_epochs, retain_weight=1.0):
        """Gradient Ascent training step"""
        self.model.train()
        retain_iter = iter(retain_loader) if retain_loader else None
        criterion = nn.CrossEntropyLoss()
        
        # TQDM for training batches
        pbar = tqdm(forget_loader if forget_loader else retain_loader, 
                   desc=f"GA Training Epoch {epoch}/{total_epochs}")
        
        for i, batch_data in enumerate(pbar):
            total_loss = 0
            
            # Forget loss (gradient ascent - maximize loss)
            if forget_loader:
                forget_data, forget_target = batch_data
                forget_data = forget_data.to(self.device)
                forget_target = forget_target.to(self.device)
                
                forget_output = self.model(forget_data)
                forget_loss = criterion(forget_output, forget_target)
                
                # Negative loss for gradient ascent
                total_loss += -0.5 * forget_loss
            
            # Retain loss (gradient descent - minimize loss)
            if retain_iter:
                try:
                    retain_data, retain_target = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_data, retain_target = next(retain_iter)
                
                retain_data = retain_data.to(self.device)
                retain_target = retain_target.to(self.device)
                retain_output = self.model(retain_data)
                retain_loss = criterion(retain_output, retain_target)
                
                total_loss += retain_weight * retain_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{total_loss.item():.3f}'})

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

    def unlearn(self, forget_classes, retain_classes, epochs=40):
        """Main GA unlearning with best model selection"""
        print(f"GA Unlearning: Forget {forget_classes}, Retain {retain_classes}")
        
        forget_loader = self.create_forget_dataset(forget_classes)
        retain_loader = self.create_retain_dataset(retain_classes)
        
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.001,
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        best_score = float('-inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nGA Unlearning Epoch {epoch+1}/{epochs}")
            
            # Training with TQDM for batches
            self.ga_training_step(forget_loader, retain_loader, optimizer, epoch+1, epochs)
            
            # Validate after each epoch
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            target_range = retain_classes  # The final target range for this step
            target_acc = self.validate_range(target_range)
            
            # Unlearning score: minimize forget accuracy, maximize retain accuracy
            unlearn_score = retain_acc - forget_acc
            
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

class ViTEREWC:
    def __init__(self, model, device='cuda', lambda_ewc=1000):
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.memory_buffer = MemoryBuffer(max_size_per_class=20)
        self.seen_classes = []
        
        # EWC-specific attributes
        self.fisher_info = {}
        self.old_params = {}
        
        self.model = model.to(device)
    
    def compute_fisher_information(self, data_loader):
        """Compute Fisher Information Matrix"""
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Mask targets to only seen classes
            mask = torch.zeros(targets.size(0), dtype=bool)
            for i, target in enumerate(targets):
                if target.item() in self.seen_classes:
                    mask[i] = True
            
            if mask.sum() == 0:
                continue
                
            data, targets = data[mask], targets[mask]
            
            self.model.zero_grad()
            outputs = self.model(data)
            
            # Mask outputs to seen classes
            output_mask = torch.zeros_like(outputs, dtype=bool)
            output_mask[:, self.seen_classes] = True
            masked_outputs = outputs.masked_fill(~output_mask, -1e9)
            
            loss = F.cross_entropy(masked_outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize Fisher information
        num_samples = len(data_loader.dataset)
        for name in fisher:
            fisher[name] /= num_samples
            
        return fisher
    
    def ewc_loss(self):
        """Compute EWC regularization loss"""
        if not self.fisher_info or not self.old_params:
            return 0
        
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.old_params:
                loss += (self.fisher_info[name] * (param - self.old_params[name]) ** 2).sum()
        
        return self.lambda_ewc * loss
    
    def update_ewc_params(self, data_loader):
        """Update Fisher information and old parameters"""
        current_fisher = self.compute_fisher_information(data_loader)
        
        # Store old parameters
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.data.clone()
        
        # Accumulate Fisher information
        if not self.fisher_info:
            self.fisher_info = current_fisher
        else:
            for name in current_fisher:
                self.fisher_info[name] += current_fisher[name]
    
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
    
    def learn_classes(self, new_classes, epochs=40):
        """Learn new classes using ER-EWC"""
        print(f"ER-EWC Learning: Adding classes {new_classes}")
        
        # Update seen classes
        self.seen_classes.extend(new_classes)
        
        # Setup optimizer
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
            print(f"\nER-EWC Learning Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            
            # TQDM for training batches
            pbar = tqdm(new_class_loader, desc=f"ER-EWC Training Epoch {epoch+1}/{epochs}")
            
            for batch_data, batch_labels in pbar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                # Forward pass for incoming data
                incoming_logits = self.model(batch_data)
                
                # Get replay data from old classes
                old_classes = [c for c in self.seen_classes if c not in new_classes]
                replay_data, replay_labels = self.memory_buffer.sample_batch(32, old_classes)
                
                # Compute cross-entropy loss
                if replay_data is not None:
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    
                    # Combine incoming and replay data
                    all_data = torch.cat([batch_data, replay_data], dim=0)
                    all_labels = torch.cat([batch_labels, replay_labels], dim=0)
                    all_logits = self.model(all_data)
                    
                    # Mask to seen classes only
                    mask = torch.zeros_like(all_logits, dtype=bool)
                    mask[:, self.seen_classes] = True
                    masked_logits = all_logits.masked_fill(~mask, -1e9)
                    
                    ce_loss = F.cross_entropy(masked_logits, all_labels)
                else:
                    # Only incoming data
                    mask = torch.zeros_like(incoming_logits, dtype=bool)
                    mask[:, batch_labels.unique()] = True
                    masked_logits = incoming_logits.masked_fill(~mask, -1e9)
                    ce_loss = F.cross_entropy(masked_logits, batch_labels)
                
                # Compute EWC loss
                ewc_penalty = self.ewc_loss()
                
                # Total loss
                total_batch_loss = ce_loss + ewc_penalty
                
                # Backward pass
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                # Add current batch to memory buffer
                self.memory_buffer.add_samples(batch_data.cpu(), batch_labels.cpu())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{total_batch_loss.item():.3f}'})
            
            # Validation
            new_acc = self.validate_epoch(new_classes)
            old_classes = [c for c in self.seen_classes if c not in new_classes]
            retain_acc = self.validate_epoch(old_classes) if old_classes else 0
            target_range = self.seen_classes  # Current target range (retain + new)
            target_acc = self.validate_range(target_range)
            
            # Learning score
            learn_score = retain_acc - (100 - new_acc)
            
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
        
        # Update EWC parameters
        combined_loader = get_dynamic_loader(
            class_range=(min(self.seen_classes), max(self.seen_classes)), 
            mode="train", batch_size=32
        )
        self.update_ewc_params(combined_loader)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create steps directory
    os.makedirs('steps', exist_ok=True)
    
    # Load base model (0-49)
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting pipeline with base model (classes 0-49)")
    
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
        print(f"\n{'='*60}")
        print(f"Step {step_num}: {min(current_classes)}-{max(current_classes)} -> {min(retain_classes + new_classes)}-{max(retain_classes + new_classes)}")
        print(f"{'='*60}")
        
        # Phase 1: GA Unlearning
        ga_unlearner = GAUnlearner(model, device)
        
        # Initial validation before unlearning
        print("\nInitial validation before GA unlearning:")
        forget_acc, retain_acc = ga_unlearner.validate(forget_classes, retain_classes)
        target_range = retain_classes + new_classes  # Final target range for this step
        target_acc = ga_unlearner.validate_range(target_range)
        print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        ga_unlearner.unlearn(forget_classes, retain_classes, epochs=40)
        
        # Phase 2: ER-EWC Learning
        er_ewc = ViTEREWC(model, device, lambda_ewc=1000)
        er_ewc.seen_classes = retain_classes.copy()
        
        # Populate buffer with retained classes
        er_ewc.populate_buffer(retain_classes)
        
        # Initial validation before learning new classes
        print(f"\nInitial validation before ER-EWC learning:")
        retain_acc = er_ewc.validate_epoch(retain_classes)
        target_acc = er_ewc.validate_range(target_range)
        print(f"Validation - Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # Learn new classes
        er_ewc.learn_classes(new_classes, epochs=40)
        
        # Update current classes
        current_classes = retain_classes + new_classes
        
        # Save checkpoint
        checkpoint_path = f'steps/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_acc = er_ewc.validate_epoch(current_classes)
        print(f"Final accuracy on classes {min(current_classes)}-{max(current_classes)}: {final_acc:.2f}%")

if __name__ == "__main__":
    main()

