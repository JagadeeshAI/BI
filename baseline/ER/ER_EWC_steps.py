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

class MemoryBuffer:
    def __init__(self, max_size_per_class=20):
        self.max_size_per_class = max_size_per_class
        self.buffer = {}
        
    def add_samples(self, samples, labels):
        for sample, label in zip(samples, labels):
            # Handle both tensor and int labels
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

class ViTEREWC:
    def __init__(self, model_path=None, device='cuda', lambda_ewc=1000):
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.memory_buffer = MemoryBuffer(max_size_per_class=20)
        self.seen_classes = list(range(50))  
        
        # EWC-specific attributes
        self.fisher_info = {}
        self.old_params = {}
        
        # Load model using get_model function
        self.model = get_model(num_classes=100, pretrained=True)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(device)
    
    def compute_fisher_information(self, data_loader):
        """Compute Fisher Information Matrix for current task"""
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
        """Update Fisher information and old parameters after learning a task"""
        # Compute Fisher information for current task
        current_fisher = self.compute_fisher_information(data_loader)
        
        # Store old parameters
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.data.clone()
        
        # Accumulate Fisher information (for multiple tasks)
        if not self.fisher_info:
            self.fisher_info = current_fisher
        else:
            for name in current_fisher:
                self.fisher_info[name] += current_fisher[name]
    
    def validate_epoch(self, val_classes):
        """Validate on specified classes and return accuracy"""
        val_loader = get_dynamic_loader(class_range=(min(val_classes), max(val_classes)), mode="val", batch_size=64)
        
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
        
        accuracy = 100 * correct / total
        return accuracy
    
    def populate_buffer_initial(self):
        """Populate buffer with samples from classes 0-49"""
        train_loader = get_dynamic_loader(class_range=(0, 49), mode="train", batch_size=32)
        
        # Collect samples by class
        class_counts = {i: 0 for i in range(50)}
        
        for batch_data, batch_labels in train_loader:
            for sample, label in zip(batch_data, batch_labels):
                label = label.item() if hasattr(label, 'item') else label
                if class_counts[label] < 20:  # Max 20 samples per class
                    self.memory_buffer.add_samples([sample], [label])
                    class_counts[label] += 1
            
            # Stop if we have enough samples for all classes
            if all(count >= 20 for count in class_counts.values()):
                break
        
        print(f"Buffer populated: {sum(len(self.memory_buffer.buffer[i]) for i in range(50))} samples")
        
        # Initialize EWC with initial task
        self.update_ewc_params(train_loader)
    
    def train_step(self, step_num, epochs=5, lr=1e-4):
        """Train for one step (add 10 new classes)"""
        print(f"\n=== Step {step_num}: Adding classes {50 + (step_num-1)*10} to {50 + step_num*10 - 1} ===")
        
        # Update seen classes
        new_classes = list(range(50 + (step_num-1)*10, 50 + step_num*10))
        self.seen_classes.extend(new_classes)
        
        # Setup optimizer - freeze earlier layers more aggressively
        optimizer = torch.optim.Adam([
            {'params': self.model.head.parameters(), 'lr': lr},
            {'params': [p for n, p in self.model.named_parameters() if 'head' not in n], 'lr': lr/20}
        ])
        
        # Load new class data using dynamic loader
        start_class = 50 + (step_num-1)*10
        end_class = 50 + step_num*10 - 1
        new_class_loader = get_dynamic_loader(class_range=(start_class, end_class), mode="train", batch_size=32)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_ewc_loss = 0
            num_batches = 0
            
            # Training loop with tqdm
            pbar = tqdm(new_class_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_data, batch_labels in pbar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                # Forward pass for incoming data
                incoming_logits = self.model(batch_data)
                
                # Get replay data from old classes only
                old_classes = self.seen_classes[:-10] if len(self.seen_classes) > 10 else []
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
                total_ce_loss += ce_loss.item()
                total_ewc_loss += ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Total': f'{total_batch_loss.item():.4f}',
                    'CE': f'{ce_loss.item():.4f}',
                    'EWC': f'{ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty:.4f}'
                })
                
                # Add current batch to memory buffer
                self.memory_buffer.add_samples(batch_data.cpu(), batch_labels.cpu())
            
            # Calculate train accuracy on current seen classes
            train_acc = self.validate_epoch(self.seen_classes)
            
            # Calculate validation accuracy on current seen classes  
            val_acc = self.validate_epoch(self.seen_classes)
            
            avg_loss = total_loss / num_batches
            avg_ce_loss = total_ce_loss / num_batches
            avg_ewc_loss = total_ewc_loss / num_batches
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, EWC: {avg_ewc_loss:.4f}) - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Update EWC parameters after training on new task
        combined_loader = get_dynamic_loader(class_range=(min(self.seen_classes), max(self.seen_classes)), mode="train", batch_size=32)
        self.update_ewc_params(combined_loader)
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = '/home/jag/codes/Bi/baseline/ER/checkpoints/ewc'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
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
                
                # Mask outputs to only consider seen classes
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
    # Initialize ER-EWC with ViT
    er_ewc = ViTEREWC(
        model_path='/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lambda_ewc=1000  # EWC regularization strength
    )
    
    # Populate buffer with initial classes (0-49)
    print("Populating memory buffer with classes 0-49...")
    er_ewc.populate_buffer_initial()
    
    # Evaluate baseline (classes 0-49)
    print("\nBaseline evaluation:")
    er_ewc.evaluate(list(range(50)))
    
    # Training steps
    for step in range(1, 6):  # Steps 1-5
        er_ewc.train_step(step, epochs=5)
        
        # Evaluate on all seen classes
        print(f"\nEvaluation after step {step}:")
        er_ewc.evaluate()
        
        # Evaluate on old classes only (forgetting check)
        if step > 1:
            old_classes = list(range(50 + (step-2)*10))
            print(f"Forgetting check - Old classes 0-{max(old_classes)}:")
            er_ewc.evaluate(old_classes)

if __name__ == "__main__":
    main()