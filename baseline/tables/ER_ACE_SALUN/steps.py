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

class SALUNUnlearner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def validate(self, target_classes, retain_classes):
        """Validation following original paper methodology"""
        self.model.eval()
        
        # Forget classes validation
        forget_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)),
            mode="val", batch_size=128, num_workers=0
        )
        forget_correct = 0
        forget_total = 0
        
        with torch.no_grad():
            for data, target in forget_loader:
                data, target = data.to(self.device), target.to(self.device)
                mask = torch.isin(target, torch.tensor(target_classes).to(self.device))
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

    def create_combined_dataset(self, class_range, target_classes, batch_size=128):
        """Create combined dataset with forget (random labels) + retain data"""
        # Forget data with random labels (Paper Algorithm 1)
        forget_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        forget_data = []
        forget_labels = []
        original_labels = []
        
        max_samples_per_class = 500  # Same as paper
        class_counts = {cls: 0 for cls in target_classes}
        
        for batch_data, batch_labels in forget_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if (label_item in target_classes and 
                    class_counts[label_item] < max_samples_per_class):
                    
                    forget_data.append(batch_data[i])
                    original_labels.append(label_item)
                    
                    # Create random label ≠ original (RL method from paper)
                    num_classes = class_range[1] + 1
                    random_label = torch.randint(0, num_classes, (1,)).item()
                    while random_label == label_item:
                        random_label = torch.randint(0, num_classes, (1,)).item()
                    forget_labels.append(random_label)
                    
                    class_counts[label_item] += 1
            
            if all(count >= max_samples_per_class for count in class_counts.values()):
                break

        # Retain data with original labels
        retain_start = max(target_classes) + 1
        retain_end = class_range[1]
        retain_data = []
        retain_labels = []
        
        if retain_start <= retain_end:
            retain_loader = get_dynamic_loader(
                class_range=(retain_start, retain_end),
                mode="train", batch_size=batch_size, num_workers=0
            )
            
            max_retain_samples = 2000
            for batch_data, batch_labels in retain_loader:
                for i, label in enumerate(batch_labels):
                    if len(retain_data) < max_retain_samples:
                        retain_data.append(batch_data[i])
                        retain_labels.append(label.item())
                if len(retain_data) >= max_retain_samples:
                    break

        # Combine datasets
        if forget_data and retain_data:
            all_data = torch.stack(forget_data + retain_data)
            all_labels = torch.tensor(forget_labels + retain_labels)
            combined_dataset = torch.utils.data.TensorDataset(all_data, all_labels)
            combined_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            return combined_loader, torch.tensor(original_labels)
        
        return None, None

    def compute_weight_saliency_mask(self, class_range, target_classes, sparsity=0.5):
        """
        Compute gradient-based weight saliency mask (Equation 3)
        mS = 1(|∇θℓf(θ; Df)| ≥ γ)
        """
        print("Computing weight saliency mask...")
        
        # Create forget loader with ORIGINAL labels for gradient computation
        forget_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)),
            mode="train", batch_size=64, num_workers=0
        )
        
        forget_gradients = {}
        batch_count = 0
        
        self.model.train()
        for data, target in forget_loader:
            data, target = data.to(self.device), target.to(self.device)
            mask = torch.isin(target, torch.tensor(target_classes).to(self.device))
            
            if mask.any():
                data_filtered = data[mask]
                target_filtered = target[mask]
                
                self.model.zero_grad()
                output = self.model(data_filtered)
                
                # Use cross-entropy loss as in Equation (5)
                loss = F.cross_entropy(output, target_filtered)
                loss.backward()
                
                # Accumulate gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if name not in forget_gradients:
                            forget_gradients[name] = torch.zeros_like(param.grad)
                        forget_gradients[name] += param.grad.clone().abs()
                
                batch_count += 1
                if batch_count >= 10:  # Sufficient for gradient estimation
                    break

        # Normalize gradients
        for name in forget_gradients:
            forget_gradients[name] /= batch_count

        # Create hard-threshold saliency mask
        saliency_mask = {}
        for name, grad in forget_gradients.items():
            flat_grad = grad.flatten()
            threshold = torch.quantile(flat_grad, 1.0 - sparsity)
            mask = (grad >= threshold).float()
            saliency_mask[name] = mask
            
            selected_ratio = mask.mean().item()
            print(f"Layer {name}: {selected_ratio:.3f} weights selected")
        
        return saliency_mask

    def train_epoch(self, train_loader, criterion, optimizer, epoch, mask, args):
        """Training function following original code structure"""
        
        class AverageMeter:
            def __init__(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0
            
            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
        
        def accuracy(output, target, topk=(1,)):
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        self.model.train()
        
        for i, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply saliency mask to gradients (key SALUN step)
            if mask:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in mask:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            # Record metrics
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch}][{i+1}/{len(train_loader)}] "
                      f"Loss {losses.avg:.4f} Acc {top1.avg:.3f}")
        
        return top1.avg

    def unlearn(self, forget_classes, retain_classes, class_range, lr=0.01, epochs=10, sparsity=0.5):
        """Main SALUN unlearning following original paper methodology"""
        
        # Initial validation
        initial_forget_acc, initial_retain_acc = self.validate(forget_classes, retain_classes)
        print(f"Initial: Forget {initial_forget_acc:.1f}% | Retain {initial_retain_acc:.1f}%")
        
        # Compute weight saliency mask (Algorithm 1, Step 1)
        saliency_mask = self.compute_weight_saliency_mask(class_range, forget_classes, sparsity)
        
        # Create combined training dataset (Algorithm 1, Step 2)
        train_loader, original_labels = self.create_combined_dataset(
            class_range, forget_classes, batch_size=128)
        
        if train_loader is None:
            print("Failed to create training data")
            return
        
        # Setup optimizer (following paper settings)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # SALUN training loop (Algorithm 1, Step 3)
        best_model_state = None
        best_score = 0
        
        print("Starting SALUN training...")
        for epoch in range(epochs):
            # Training with saliency mask
            train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch, saliency_mask, None)
            
            # Validation
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            
            print(f"Epoch {epoch+1:2d}: Train {train_acc:.1f}% | Forget {forget_acc:.1f}% | Retain {retain_acc:.1f}%")
            
            # Model selection
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 30)
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            final_forget_acc, final_retain_acc = self.validate(forget_classes, retain_classes)
            print(f"Best model: Forget {final_forget_acc:.1f}% | Retain {final_retain_acc:.1f}%")

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
    os.makedirs('steps_ace_salun', exist_ok=True)
    
    # Load base model (0-49)
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting ER-ACE + SALUN pipeline with base model (classes 0-49)")
    
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
        
        # Phase 1: SALUN Unlearning
        salun_unlearner = SALUNUnlearner(model, device)
        
        # Define class ranges per step
        step_ranges = {1: (0, 59), 2: (0, 69), 3: (0, 79), 4: (0, 89), 5: (0, 99)}
        class_range = step_ranges[step_num]
        
        # Initial validation before unlearning
        print("\nInitial validation before SALUN unlearning:")
        forget_acc, retain_acc = salun_unlearner.validate(forget_classes, retain_classes)
        target_range = retain_classes + new_classes  # Final target range for this step
        target_acc = salun_unlearner.validate_range(target_range)
        print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # SALUN unlearning
        salun_unlearner.unlearn(forget_classes, retain_classes, class_range, 
                               lr=0.01, epochs=10, sparsity=0.5)
        
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
        checkpoint_path = f'steps_ace_salun/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_acc = er_ace.validate_epoch(current_classes)
        print(f"Final accuracy on classes {min(current_classes)}-{max(current_classes)}: {final_acc:.2f}%")

if __name__ == "__main__":
    main()