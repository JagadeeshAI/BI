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

class ViTERACE:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.memory_buffer = MemoryBuffer(max_size_per_class=20)
        self.seen_classes = list(range(50))  
        
        # Load model using get_model function
        self.model = get_model(num_classes=100, pretrained=True)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(device)
    
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
    
    def er_ace_loss(self, incoming_logits, incoming_targets, replay_logits=None, replay_targets=None):
        """ER-ACE asymmetric loss"""
        # For incoming data: mask to only allow current step classes
        mask_incoming = torch.zeros_like(incoming_logits, dtype=bool)
        current_step_classes = incoming_targets.unique()
        mask_incoming[:, current_step_classes] = True
        
        # Apply mask by setting unwanted logits to -inf
        masked_logits_incoming = incoming_logits.masked_fill(~mask_incoming, -1e9)
        loss_incoming = F.cross_entropy(masked_logits_incoming, incoming_targets)
        
        # For replay data: use all seen classes (not all 100)
        if replay_logits is not None and replay_targets is not None:
            mask_replay = torch.zeros_like(replay_logits, dtype=bool)
            mask_replay[:, self.seen_classes] = True
            masked_logits_replay = replay_logits.masked_fill(~mask_replay, -1e9)
            loss_replay = F.cross_entropy(masked_logits_replay, replay_targets)
            return loss_incoming + loss_replay
        
        return loss_incoming
    
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
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Add current batch to memory buffer
                self.memory_buffer.add_samples(batch_data.cpu(), batch_labels.cpu())
            
            # Calculate train accuracy on current seen classes
            train_acc = self.validate_epoch(self.seen_classes)
            
            # Calculate validation accuracy on current seen classes  
            val_acc = self.validate_epoch(self.seen_classes)
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = '/home/jag/codes/Bi/baseline/ER/checkpoints'
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
    # Initialize ER-ACE with ViT
    erace = ViTERACE(
        model_path='/home/jag/codes/Bi/checkpoints/oracle/0_49.pth', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Populate buffer with initial classes (0-49)
    print("Populating memory buffer with classes 0-49...")
    erace.populate_buffer_initial()
    
    # Evaluate baseline (classes 0-49)
    print("\nBaseline evaluation:")
    erace.evaluate(list(range(50)))
    
    # Training steps
    for step in range(1, 6):  # Steps 1-5
        erace.train_step(step, epochs=5)
        
        # Evaluate on all seen classes
        print(f"\nEvaluation after step {step}:")
        erace.evaluate()
        
        # Evaluate on old classes only (forgetting check)
        if step > 1:
            old_classes = list(range(50 + (step-2)*10))
            print(f"Forgetting check - Old classes 0-{max(old_classes)}:")
            erace.evaluate(old_classes)

if __name__ == "__main__":
    main()