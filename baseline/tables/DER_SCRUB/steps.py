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

class SCRUBUnlearner:
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
        max_samples_per_class = 200  # Reduced for stability
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
        max_samples = 1500  # More retain samples for stability
        
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

    def augment_data(self, data):
        """Apply data augmentation as mentioned in DER++ paper"""
        if random.random() > 0.5:
            # Random horizontal flip
            data = torch.flip(data, dims=[-1])
        return data

    def scrub_max_step(self, forget_loader, teacher_model, optimizer, buffer):
        """SCRUB MAX-STEP: Train only on forget data to maximize divergence"""
        if not forget_loader:
            return 0
            
        self.model.train()
        teacher_model.eval()
        
        max_loss = 0
        batch_count = 0
        
        # Progress bar for forget batches
        pbar = tqdm(forget_loader, desc="MAX-STEP (Forget)", leave=False)
        
        for forget_data, forget_target in pbar:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            max_loss += max_loss_batch.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{max_loss_batch.item():.3f}'})
            
            # Store current logits in buffer (DER++ style)
            with torch.no_grad():
                current_logits = self.model(forget_data).detach()
                buffer.add_samples(forget_data, current_logits, forget_target)
        
        return max_loss / batch_count if batch_count > 0 else 0

    def scrub_min_step(self, retain_loader, teacher_model, optimizer, buffer, alpha=1.0, beta=1.0):
        """SCRUB MIN-STEP: Train only on retain data + DER++ buffer replay"""
        if not retain_loader:
            return 0
            
        self.model.train()
        teacher_model.eval()
        
        min_loss = 0
        batch_count = 0
        
        # Progress bar for retain batches
        pbar = tqdm(retain_loader, desc="MIN-STEP (Retain)", leave=False)
        
        for retain_data, retain_target in pbar:
            retain_data = retain_data.to(self.device)
            retain_target = retain_target.to(self.device)
            
            # Apply augmentation (DER++ style)
            retain_data_aug = self.augment_data(retain_data)
            
            with torch.no_grad():
                teacher_output = teacher_model(retain_data_aug)
            
            student_output = self.model(retain_data_aug)
            
            # SCRUB: Minimize divergence from teacher + classification loss
            retain_kl_loss = alpha * self.kl_divergence_loss(student_output, teacher_output)
            retain_ce_loss = F.cross_entropy(student_output, retain_target)
            
            total_loss = retain_kl_loss + retain_ce_loss
            
            # DER++ Buffer replay component
            buffer_data, buffer_logits, buffer_labels = buffer.sample_batch(16)
            if buffer_data is not None:
                buffer_data = buffer_data.to(self.device)
                buffer_logits = buffer_logits.to(self.device)
                buffer_labels = buffer_labels.to(self.device)
                
                # Apply augmentation to buffer data
                buffer_data_aug = self.augment_data(buffer_data)
                
                # Forward pass on buffer data
                buffer_output = self.model(buffer_data_aug)
                
                # DER++: Logit distillation + classification on buffer
                buffer_distill_loss = alpha * F.mse_loss(buffer_output, buffer_logits)
                buffer_class_loss = beta * F.cross_entropy(buffer_output, buffer_labels)
                
                total_loss += buffer_distill_loss + buffer_class_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{total_loss.item():.3f}'})
            
            # Store current batch in buffer
            with torch.no_grad():
                current_logits = self.model(retain_data).detach()
                buffer.add_samples(retain_data, current_logits, retain_target)
            
            min_loss += total_loss.item()
            batch_count += 1
        
        return min_loss / batch_count if batch_count > 0 else 0

    def scrub_training_step(self, forget_loader, retain_loader, teacher_model, optimizer, 
                           buffer, epoch, total_epochs, max_steps=1, min_steps=3, alpha=1.0, beta=1.0):
        """SCRUB+DER++ training step with alternating min-max optimization"""
        
        epoch_stats = {'max_loss': 0, 'min_loss': 0, 'max_batches': 0, 'min_batches': 0}
        
        print(f"SCRUB+DER++ Unlearning Epoch {epoch}/{total_epochs}")
        
        # Perform alternating max-steps and min-steps
        for step in range(max(max_steps, min_steps)):
            if step < max_steps and forget_loader:
                # MAX-STEP: Only on forget data
                max_loss = self.scrub_max_step(forget_loader, teacher_model, optimizer, buffer)
                epoch_stats['max_loss'] += max_loss
                epoch_stats['max_batches'] += 1
            
            if step < min_steps and retain_loader:
                # MIN-STEP: Only on retain data + buffer
                min_loss = self.scrub_min_step(retain_loader, teacher_model, optimizer, buffer, alpha, beta)
                epoch_stats['min_loss'] += min_loss
                epoch_stats['min_batches'] += 1
        
        # Additional min-steps for stability (as per SCRUB paper)
        for extra_step in range(3):
            if retain_loader:
                extra_min_loss = self.scrub_min_step(retain_loader, teacher_model, optimizer, buffer, alpha, beta)
                epoch_stats['min_loss'] += extra_min_loss
                epoch_stats['min_batches'] += 1
        
        # Average the losses
        avg_max_loss = epoch_stats['max_loss'] / epoch_stats['max_batches'] if epoch_stats['max_batches'] > 0 else 0
        avg_min_loss = epoch_stats['min_loss'] / epoch_stats['min_batches'] if epoch_stats['min_batches'] > 0 else 0
        
        return {'max_loss': avg_max_loss, 'min_loss': avg_min_loss}

    def validate_range(self, class_range):
        """Validate on specific class range"""
        if not class_range:
            return 0
            
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

    def unlearn(self, forget_classes, retain_classes, epochs=40, alpha=1.0, beta=1.0):
        """Main SCRUB+DER++ unlearning with alternating optimization"""
        print(f"SCRUB+DER++ Unlearning: Forget {forget_classes}, Retain {retain_classes}")
        
        # Create teacher model (frozen copy of current model)
        teacher_model = get_model(num_classes=100, use_lora=False, pretrained=False)
        teacher_model.to(self.device)
        teacher_model.load_state_dict(self.model.state_dict())
        teacher_model.eval()
        
        # Freeze teacher parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        forget_loader = self.create_forget_dataset(forget_classes)
        retain_loader = self.create_retain_dataset(retain_classes)
        
        # Initialize DER++ buffer
        buffer = ReservoirBuffer(max_size=1000)
        
        # Populate buffer initially with retain samples
        if retain_loader:
            self.model.eval()
            for batch_data, batch_labels in retain_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(batch_data)
                    buffer.add_samples(batch_data, logits, batch_labels)
                break  # Just one batch for initial population
        
        # Conservative optimizer settings for stability
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=1e-4,  # Much lower learning rate
            weight_decay=1e-4
        )
        
        best_score = float('-inf')
        best_model_state = None
        
        # Training loop with individual epoch progress
        for epoch in range(1, epochs + 1):
            # SCRUB alternating training step
            stats = self.scrub_training_step(
                forget_loader, retain_loader, teacher_model, optimizer, 
                buffer, epoch, epochs, max_steps=1, min_steps=3, alpha=alpha, beta=beta
            )
            
            # Calculate total loss (similar to NegGrad+ format)
            total_loss = stats['min_loss'] - stats['max_loss']  # MIN - MAX (retain - forget)
            
            # Print epoch statistics
            print(f"Loss - Total: {total_loss:.3f} | Forget: {abs(stats['max_loss']):.3f} | Retain: {stats['min_loss']:.3f}")
            
            # Validate after each epoch
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            target_range = retain_classes  # The final target range for this step
            target_acc = self.validate_range(target_range)
            
            # SCRUB unlearning score: minimize forget accuracy, maximize retain accuracy
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 40)
            
            print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range) if target_range else 'N/A'}-{max(target_range) if target_range else 'N/A'}: {target_acc:.1f}% | Score: {unlearn_score:.1f}")
            
            # Save best model
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"★ New best model! Score: {unlearn_score:.1f}")
            
            # Early stopping conditions
            if forget_acc < 15 and retain_acc > 70:
                print(f"✅ Good unlearning achieved at epoch {epoch}")
                break
                
            if retain_acc < 30 and epoch > 10:
                print(f"⚠️ Retain accuracy too low, stopping at epoch {epoch}")
                break
            
            print()  # Add blank line after each epoch
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best unlearned model (Score: {best_score:.1f})")

class ViTDERPlusPlus:
    def __init__(self, model, device='cuda'):
        self.device = device
        self.buffer = ReservoirBuffer(max_size=1000)
        self.seen_classes = []
        self.model = model.to(device)
    
    def der_plus_plus_loss(self, current_logits, current_labels, buffer_logits=None, 
                          buffer_stored_logits=None, buffer_labels=None, alpha=0.5, beta=0.5):
        """DER++ loss with logit distillation and buffer classification"""
        # Current task loss
        loss_current = F.cross_entropy(current_logits, current_labels)
        
        if buffer_logits is not None and buffer_stored_logits is not None and buffer_labels is not None:
            # Logit distillation loss - key to avoiding catastrophic forgetting
            loss_distill = F.mse_loss(buffer_logits, buffer_stored_logits)
            
            # Buffer classification loss  
            loss_buffer = F.cross_entropy(buffer_logits, buffer_labels)
            
            total_loss = loss_current + alpha * loss_distill + beta * loss_buffer
            return total_loss, loss_current.item(), loss_distill.item(), loss_buffer.item()
        
        return loss_current, loss_current.item(), 0.0, 0.0
    
    def validate_epoch(self, val_classes):
        """Validate on specified classes"""
        if not val_classes:
            return 0
            
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
                
                # Filter to only include samples from val_classes
                mask = torch.isin(labels, torch.tensor(val_classes).to(self.device))
                if mask.any():
                    data_filtered = data[mask]
                    labels_filtered = labels[mask]
                    
                    outputs = self.model(data_filtered)
                    predicted = outputs.argmax(dim=1)
                    
                    correct += (predicted == labels_filtered).sum().item()
                    total += labels_filtered.size(0)
        
        return 100 * correct / total if total > 0 else 0
    
    def validate_range(self, class_range):
        """Validate on specific class range"""
        if not class_range:
            return 0
            
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
        if not classes:
            return
            
        train_loader = get_dynamic_loader(
            class_range=(min(classes), max(classes)), 
            mode="train", batch_size=32
        )
        
        self.model.eval()
        class_counts = {i: 0 for i in classes}
        samples_per_class = 50  # Increase buffer samples
        
        print(f"Populating buffer with {samples_per_class} samples per class for classes {classes}")
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            with torch.no_grad():
                logits = self.model(batch_data)
            
            for sample, logit, label in zip(batch_data, logits, batch_labels):
                label = label.item()
                if label in classes and class_counts[label] < samples_per_class:
                    self.buffer.add_samples([sample], [logit], [label])
                    class_counts[label] += 1
            
            if all(count >= samples_per_class for count in class_counts.values()):
                break
        
        print(f"Buffer populated with {sum(class_counts.values())} samples")
    
    def learn_classes(self, new_classes, epochs=40, alpha=0.5, beta=0.5):
        """Learn new classes using DER++"""
        print(f"DER++ Learning: Adding classes {new_classes}")
        
        # Update seen classes
        self.seen_classes.extend(new_classes)
        
        # Setup optimizer with more conservative learning rates
        optimizer = torch.optim.Adam([
            {'params': self.model.head.parameters(), 'lr': 5e-5},  # Lower LR for head
            {'params': [p for n, p in self.model.named_parameters() if 'head' not in n], 'lr': 1e-5}
        ], weight_decay=1e-4)
        
        # Load new class data
        new_class_loader = get_dynamic_loader(
            class_range=(min(new_classes), max(new_classes)), 
            mode="train", batch_size=32
        )
        
        self.model.train()
        
        best_score = float('-inf')
        best_model_state = None
        
        # Training loop with individual epoch progress
        for epoch in range(1, epochs + 1):
            print(f"DER++ Learning Epoch {epoch}/{epochs}")
            total_loss = 0
            total_current = 0
            total_distill = 0
            total_buffer = 0
            num_batches = 0
            
            # TQDM for training batches
            pbar = tqdm(new_class_loader, desc=f"Training Epoch {epoch}", leave=False)
            
            for batch_data, batch_labels in pbar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                # Forward pass for current data
                current_logits = self.model(batch_data)
                
                # Sample from buffer
                buffer_data, buffer_stored_logits, buffer_labels = self.buffer.sample_batch(16)
                
                if buffer_data is not None:
                    buffer_data = buffer_data.to(self.device)
                    buffer_stored_logits = buffer_stored_logits.to(self.device)
                    buffer_labels = buffer_labels.to(self.device)
                    
                    # Forward pass for buffer data
                    buffer_logits = self.model(buffer_data)
                    
                    # DER++ loss
                    loss, loss_current, loss_distill, loss_buffer = self.der_plus_plus_loss(
                        current_logits, batch_labels, 
                        buffer_logits, buffer_stored_logits, buffer_labels, alpha, beta
                    )
                else:
                    loss, loss_current, loss_distill, loss_buffer = self.der_plus_plus_loss(
                        current_logits, batch_labels
                    )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                # Store current batch in buffer (with current logits)
                with torch.no_grad():
                    current_logits_detached = self.model(batch_data).detach()
                    self.buffer.add_samples(batch_data, current_logits_detached, batch_labels)
                
                # Track losses
                total_loss += loss.item()
                total_current += loss_current
                total_distill += loss_distill
                total_buffer += loss_buffer
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Total': f'{loss.item():.3f}',
                    'Curr': f'{loss_current:.3f}',
                    'Dist': f'{loss_distill:.3f}',
                    'Buff': f'{loss_buffer:.3f}'
                })
            
            # Validation after each epoch
            new_acc = self.validate_epoch(new_classes)
            old_classes = [c for c in self.seen_classes if c not in new_classes]
            retain_acc = self.validate_epoch(old_classes) if old_classes else 0
            target_range = self.seen_classes  # Current target range (retain + new)
            target_acc = self.validate_range(target_range)
            
            # Learning score: prioritize overall target accuracy more
            learn_score = 0.7 * target_acc + 0.2 * new_acc + 0.1 * retain_acc
            
            avg_loss = total_loss / num_batches
            avg_current = total_current / num_batches
            avg_distill = total_distill / num_batches  
            avg_buffer = total_buffer / num_batches
            
            print(f"Loss - Total: {avg_loss:.3f} | Current: {avg_current:.3f} | Distill: {avg_distill:.3f}")
            print(f"Validation - New: {new_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}% | Score: {learn_score:.1f}")
            
            # Save best model
            if learn_score > best_score:
                best_score = learn_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"★ New best model! Score: {learn_score:.1f}")
            
            # Early stopping if target accuracy is good
            if target_acc > 75 and epoch > 10:
                print(f"✅ Good target accuracy achieved at epoch {epoch}")
                break
                
            # Stop if target accuracy is declining significantly
            if target_acc < 25 and epoch > 15:
                print(f"⚠️ Target accuracy too low, stopping at epoch {epoch}")
                break
            
            print()  # Add blank line after each epoch
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best learned model (Score: {best_score:.1f})")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create steps directory
    os.makedirs('steps_scrub_der', exist_ok=True)
    
    # Load base model (0-49)
    model = get_model(num_classes=100, pretrained=True)
    base_checkpoint = torch.load('/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
                                map_location=device, weights_only=True)
    model.load_state_dict(base_checkpoint)
    
    print("Starting SCRUB+DER++ pipeline with base model (classes 0-49)")
    
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
        
        # Phase 1: SCRUB Unlearning
        scrub_unlearner = SCRUBUnlearner(model, device)
        
        # Initial validation before unlearning
        print("\nInitial validation before SCRUB unlearning:")
        forget_acc, retain_acc = scrub_unlearner.validate(forget_classes, retain_classes)
        target_range = retain_classes + new_classes  # Final target range for this step
        target_acc = scrub_unlearner.validate_range(target_range)
        print(f"Validation - Forget: {forget_acc:.1f}% | Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # SCRUB unlearning with conservative parameters
        scrub_unlearner.unlearn(forget_classes, retain_classes, epochs=40, alpha=1.0, beta=1.0)
        
        # Phase 2: DER++ Learning
        der_plus_plus = ViTDERPlusPlus(model, device)
        der_plus_plus.seen_classes = retain_classes.copy()
        
        # Populate buffer with retained classes
        der_plus_plus.populate_buffer(retain_classes)
        
        # Initial validation before learning new classes
        print(f"\nInitial validation before DER++ learning:")
        retain_acc = der_plus_plus.validate_epoch(retain_classes)
        target_acc = der_plus_plus.validate_range(target_range)
        print(f"Validation - Retain: {retain_acc:.1f}% | {min(target_range)}-{max(target_range)}: {target_acc:.1f}%")
        
        # Learn new classes with DER++ using more conservative parameters
        der_plus_plus.learn_classes(new_classes, epochs=50, alpha=0.3, beta=0.3)
        
        # Update current classes
        current_classes = retain_classes + new_classes
        
        # Save checkpoint
        checkpoint_path = f'steps_scrub_der/step{step_num}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final validation
        final_acc = der_plus_plus.validate_epoch(current_classes)
        print(f"Final accuracy on classes {min(current_classes)}-{max(current_classes)}: {final_acc:.2f}%")

if __name__ == "__main__":
    main()