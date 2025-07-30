import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import random
import os
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
                self.current_size += 1
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

class ViTDERPlusPlus:
    def __init__(self, model_path=None, device='cuda', buffer_size=2000, alpha=0.5, beta=0.5):
        self.device = device
        self.buffer = ReservoirBuffer(max_size=buffer_size)
        self.seen_classes = list(range(50))  # Start with 0-49
        self.alpha = alpha  # Logit distillation weight
        self.beta = beta    # Buffer classification weight
        
        # Load model
        self.model = get_model(num_classes=100, pretrained=True)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(device)
    
    def der_plus_plus_loss(self, current_logits, current_labels, buffer_logits=None, buffer_stored_logits=None, buffer_labels=None):
        """
        DER++ Loss: L_current + α * logit_distillation + β * buffer_classification
        
        This is how DER++ avoids catastrophic forgetting:
        1. Logit Distillation (α term): Forces current model to produce similar logits to stored logits
           - Preserves "dark knowledge" - the relationships between classes
           - Maintains similarity structure across tasks
        2. Buffer Classification (β term): Ensures model still classifies old examples correctly
        3. Current Task Loss: Learns new classes
        """
        # Current task loss
        loss_current = F.cross_entropy(current_logits, current_labels)
        
        if buffer_logits is not None and buffer_stored_logits is not None and buffer_labels is not None:
            # Logit distillation loss - key to avoiding catastrophic forgetting
            loss_distill = F.mse_loss(buffer_logits, buffer_stored_logits)
            
            # Buffer classification loss
            loss_buffer = F.cross_entropy(buffer_logits, buffer_labels)
            
            total_loss = loss_current + self.alpha * loss_distill + self.beta * loss_buffer
            return total_loss, loss_current.item(), loss_distill.item(), loss_buffer.item()
        
        return loss_current, loss_current.item(), 0.0, 0.0
    
    def validate_epoch(self, val_classes):
        """Validate on specified classes"""
        val_loader = get_dynamic_loader(class_range=(min(val_classes), max(val_classes)), mode="val", batch_size=64)
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                
                # Mask to ALL seen classes, not just validation classes
                mask = torch.zeros_like(outputs, dtype=bool)
                mask[:, self.seen_classes] = True
                masked_outputs = outputs.masked_fill(~mask, -1e9)
                
                _, predicted = torch.max(masked_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def populate_buffer_initial(self):
        """Populate buffer with samples from classes 0-49"""
        train_loader = get_dynamic_loader(class_range=(0, 49), mode="train", batch_size=32)
        
        self.model.eval()
        samples_added = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Get logits for storage
                logits = self.model(batch_data)
                
                # Add to buffer with logits
                self.buffer.add_samples(batch_data, logits, batch_labels)
                samples_added += len(batch_data)
                
                if samples_added >= 1000:  # Limit initial population
                    break
        
        print(f"Buffer populated with {len(self.buffer.buffer)} samples from classes 0-49")
    
    def train_step(self, step_num, epochs=5, lr=1e-4):
        """Train one step with DER++"""
        print(f"\n=== Step {step_num}: Adding classes {50 + (step_num-1)*10} to {50 + step_num*10 - 1} ===")
        
        # Update seen classes
        new_classes = list(range(50 + (step_num-1)*10, 50 + step_num*10))
        self.seen_classes.extend(new_classes)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.head.parameters(), 'lr': lr},
            {'params': [p for n, p in self.model.named_parameters() if 'head' not in n], 'lr': lr/10}
        ])
        
        # Load new class data
        start_class = 50 + (step_num-1)*10
        end_class = 50 + step_num*10 - 1
        new_class_loader = get_dynamic_loader(class_range=(start_class, end_class), mode="train", batch_size=32)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_current = 0
            total_distill = 0
            total_buffer = 0
            num_batches = 0
            
            pbar = tqdm(new_class_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_data, batch_labels in pbar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                # Forward pass for current data
                current_logits = self.model(batch_data)
                
                # Sample from buffer
                buffer_data, buffer_stored_logits, buffer_labels = self.buffer.sample_batch(32)
                
                if buffer_data is not None:
                    buffer_data = buffer_data.to(self.device)
                    buffer_stored_logits = buffer_stored_logits.to(self.device)
                    buffer_labels = buffer_labels.to(self.device)
                    
                    # Forward pass for buffer data
                    buffer_logits = self.model(buffer_data)
                    
                    # DER++ loss
                    loss, loss_current, loss_distill, loss_buffer = self.der_plus_plus_loss(
                        current_logits, batch_labels, 
                        buffer_logits, buffer_stored_logits, buffer_labels
                    )
                else:
                    loss, loss_current, loss_distill, loss_buffer = self.der_plus_plus_loss(
                        current_logits, batch_labels
                    )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
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
                
                pbar.set_postfix({
                    'Total': f'{loss.item():.3f}',
                    'Curr': f'{loss_current:.3f}',
                    'Dist': f'{loss_distill:.3f}',
                    'Buff': f'{loss_buffer:.3f}'
                })
            
            # Detailed validation per epoch
            avg_loss = total_loss / num_batches
            avg_current = total_current / num_batches
            avg_distill = total_distill / num_batches
            avg_buffer = total_buffer / num_batches
            
            # Validation on different class groups
            old_classes = self.seen_classes[:-10] if len(self.seen_classes) > 10 else []
            new_classes = self.seen_classes[-10:]
            all_classes = self.seen_classes
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} "
                  f"(Curr: {avg_current:.3f}, Dist: {avg_distill:.3f}, Buff: {avg_buffer:.3f})")
            
            if old_classes:
                old_acc = self.validate_epoch(old_classes)
                print(f"  Old classes {min(old_classes)}-{max(old_classes)}: {old_acc:.2f}%")
            
            new_acc = self.validate_epoch(new_classes)
            print(f"  New classes {min(new_classes)}-{max(new_classes)}: {new_acc:.2f}%")
            
            all_acc = self.validate_epoch(all_classes)
            print(f"  Overall {min(all_classes)}-{max(all_classes)}: {all_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_dir = '/home/jag/codes/Bi/baseline/DER/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'step{step_num}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def evaluate(self, test_classes=None):
        """Evaluate on specified classes"""
        if test_classes is None:
            test_classes = self.seen_classes
            
        test_loader = get_dynamic_loader(class_range=(min(test_classes), max(test_classes)), mode="val", batch_size=64)
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                
                # Mask to test classes
                mask = torch.zeros_like(outputs, dtype=bool)
                mask[:, test_classes] = True
                masked_outputs = outputs.masked_fill(~mask, -1e9)
                
                _, predicted = torch.max(masked_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy on classes {min(test_classes)}-{max(test_classes)}: {accuracy:.2f}%")
        return accuracy

def main():
    # Initialize DER++
    der_plus_plus = ViTDERPlusPlus(
        model_path='/home/jag/codes/Bi/checkpoints/oracle/0_49.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        buffer_size=2000,
        alpha=0.5,  # Logit distillation weight
        beta=0.5    # Buffer classification weight
    )
    
    # Populate buffer with initial classes
    print("Populating buffer with classes 0-49...")
    der_plus_plus.populate_buffer_initial()
    
    # Baseline evaluation
    print("\nBaseline evaluation:")
    der_plus_plus.evaluate(list(range(50)))
    
    # Training steps
    for step in range(1, 6):
        der_plus_plus.train_step(step, epochs=5)
        
        # Evaluate on all seen classes
        print(f"\nEvaluation after step {step}:")
        der_plus_plus.evaluate()
        
        # Check forgetting on old classes
        if step > 1:
            old_classes = list(range(50))
            print(f"Forgetting check - Original classes 0-49:")
            der_plus_plus.evaluate(old_classes)

if __name__ == "__main__":
    main()