import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

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
            mode="val", batch_size=16, num_workers=0  # Reduced batch size and workers
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
        # Use smaller batch size and no workers to reduce memory
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
            
            # Break if we have enough samples
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
    
    def unlearn_step(self, step_num, checkpoint_path, save_path, target_classes, 
                    lr=1e-3, epochs=50, weight_decay=1e-5):
        """Perform unlearning for a specific step with memory optimization"""
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Clear checkpoint from memory
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
        
        # Create target loader with memory optimization
        target_loader = self.create_target_loader(class_range, target_classes, batch_size=16)
        
        if target_loader is None:
            print(f"No samples found for target classes {target_classes}")
            return
        
        print(f"Computing weight importance")
        weight_importance = self.compute_weight_importance(target_loader, target_classes)
        
        # Clear cache after importance computation
        torch.cuda.empty_cache()
        
        # Optimizer with original learning rate
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, 
                             weight_decay=weight_decay)
        
        # Store original weights for regularization
        original_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_weights[name] = param.data.clone()
        
        print(f"Starting unlearning for step {step_num}")
        
        # Define validation ranges
        val_range = (target_classes[-1] + 1, step_ranges[step_num][1])
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Training with tqdm
            pbar = tqdm(target_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                       leave=True, dynamic_ncols=True)
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Misclassification loss
                unlearn_loss = -nn.CrossEntropyLoss()(output, target)
                unlearn_loss = torch.clamp(unlearn_loss, min=-1000.0, max=0.0)
                
                # Regularization using remaining classes
                reg_loss = 0
                remaining_range = (target_classes[-1] + 1, class_range[1])
                if remaining_range[0] <= remaining_range[1]:
                    try:
                        # Create smaller reg loader
                        reg_loader = get_dynamic_loader(
                            class_range=remaining_range, 
                            mode="train", 
                            batch_size=16,
                            num_workers=0
                        )
                        reg_data, reg_target = next(iter(reg_loader))
                        reg_data, reg_target = reg_data.to(self.device), reg_target.to(self.device)
                        reg_output = self.model(reg_data)
                        reg_loss = nn.CrossEntropyLoss()(reg_output, reg_target)
                        
                        # Clear reg data immediately
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
            
            # Validation every 5 epochs to save time and memory
            if epoch % 5 == 0 or epoch == epochs - 1:
                forget_acc, retain_acc = self.validate(target_classes, val_range)
                
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Forget Acc: {forget_acc:.2f}% (target: 0%)')
                print(f'  Retain Acc: {retain_acc:.2f}% (target: high)')
                print(f'  Loss: {total_loss:.4f}')
                
                # Early stopping
                if forget_acc < 5:
                    print(f"✅ Unlearning successful at epoch {epoch+1}")
                    break
        
        # Save unlearned model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'target_classes': target_classes
        }, save_path)
        
        print(f"Saved unlearned model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='L2UL Unlearning for ViT on CIFAR-100')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace',
                        help='Directory containing step checkpoints')
    parser.add_argument('--output_dir', type=str, default='unlearned_models',
                        help='Directory to save unlearned models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ViT model
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    
    unlearner = L2ULUnlearner(model, device=args.device)
    
    # Define unlearning tasks
    unlearn_tasks = [
        (1, 'step1.pth', 'final_step1.pth', list(range(0, 10))),   # 0-9
        (2, 'step2.pth', 'final_step2.pth', list(range(0, 20))),   # 0-19  
        (3, 'step3.pth', 'final_step3.pth', list(range(0, 30))),   # 0-29
        (4, 'step4.pth', 'final_step4.pth', list(range(0, 40))),   # 0-39
        (5, 'step5.pth', 'final_step5.pth', list(range(0, 50))),   # 0-49
    ]
    
    for step_num, checkpoint_file, output_file, target_classes in unlearn_tasks:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        save_path = os.path.join(args.output_dir, output_file)
        
        print(f"\n{'='*50}")
        print(f"Unlearning Step {step_num}")
        print(f"Target classes to forget: {len(target_classes)} classes")
        print(f"{'='*50}")
        
        unlearner.unlearn_step(
            step_num=step_num,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            target_classes=target_classes,
            lr=1e-3,
            epochs=20  # Reduced epochs for testing
        )

if __name__ == "__main__":
    main()