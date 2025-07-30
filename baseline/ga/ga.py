import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

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

    def ga_training_step(self, forget_loader, retain_loader, optimizer, retain_weight=1.0):
        """
        Gradient Ascent training step:
        - Gradient ascent on forget data (maximize loss)
        - Gradient descent on retain data (minimize loss)
        """
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
        forget_acc_meter = AverageMeter()
        retain_acc_meter = AverageMeter()
        
        self.model.train()
        retain_iter = iter(retain_loader) if retain_loader else None
        criterion = nn.CrossEntropyLoss()
        
        for i, batch_data in enumerate(forget_loader if forget_loader else retain_loader):
            total_loss = 0
            
            # Forget loss (gradient ascent - maximize loss)
            if forget_loader:
                forget_data, forget_target = batch_data
                forget_data = forget_data.to(self.device)
                forget_target = forget_target.to(self.device)
                
                forget_output = self.model(forget_data)
                forget_loss = criterion(forget_output, forget_target)
                
                # Stronger negative loss for gradient ascent
                total_loss += -0.5 * forget_loss
                
                # Track forget accuracy
                forget_prec1 = accuracy(forget_output.data, forget_target)[0]
                forget_acc_meter.update(forget_prec1.item(), forget_data.size(0))
            
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
                
                # Positive loss for gradient descent
                total_loss += retain_weight * retain_loss
                
                # Track retain accuracy
                retain_prec1 = accuracy(retain_output.data, retain_target)[0]
                retain_acc_meter.update(retain_prec1.item(), retain_data.size(0))
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            losses.update(total_loss.item(), batch_data[0].size(0) if forget_loader else retain_data.size(0))
        
        return forget_acc_meter.avg, retain_acc_meter.avg

    def unlearn_step(self, step_num, checkpoint_path, save_path, 
                     lr=0.001, epochs=5, retain_weight=1.0):
        """
        Main GA unlearning procedure to shift class ranges
        """
        
        # Load pretrained model
        print(f"Loading checkpoint: {checkpoint_path}")
        sys.stdout.flush()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
        self.model.load_state_dict(filtered_state_dict, strict=False)
        
        # Define class ranges for each step
        # Original: step1(0-59) -> final_step1(10-59)
        forget_ranges = {
            1: list(range(0, 10)),   # forget 0-9
            2: list(range(0, 20)),   # forget 0-19  
            3: list(range(0, 30)),   # forget 0-29
            4: list(range(0, 40)),   # forget 0-39
            5: list(range(0, 50))    # forget 0-49
        }
        
        retain_ranges = {
            1: list(range(10, 60)),  # retain 10-59
            2: list(range(20, 70)),  # retain 20-69
            3: list(range(30, 80)),  # retain 30-79
            4: list(range(40, 90)),  # retain 40-89
            5: list(range(50, 100))  # retain 50-99
        }
        
        forget_classes = forget_ranges[step_num]
        retain_classes = retain_ranges[step_num]
        
        print(f"Step {step_num}: Forgetting classes {forget_classes[0]}-{forget_classes[-1]}, "
              f"Retaining classes {retain_classes[0]}-{retain_classes[-1]}")
        sys.stdout.flush()
        
        # Initial validation
        print("Running initial validation...")
        sys.stdout.flush()
        initial_forget_acc, initial_retain_acc = self.validate(forget_classes, retain_classes)
        print(f"Initial: Forget {initial_forget_acc:.1f}% | Retain {initial_retain_acc:.1f}%")
        sys.stdout.flush()
        
        # Create datasets
        print("Creating datasets...")
        sys.stdout.flush()
        forget_loader = self.create_forget_dataset(forget_classes, batch_size=128)
        retain_loader = self.create_retain_dataset(retain_classes, batch_size=128)
        
        if forget_loader is None and retain_loader is None:
            print("Failed to create datasets")
            return
        
        if forget_loader:
            print(f"Forget dataset size: {len(forget_loader)} batches")
        if retain_loader:
            print(f"Retain dataset size: {len(retain_loader)} batches")
        sys.stdout.flush()
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr,
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        # GA training loop
        best_model_state = None
        best_score = float('-inf')
        
        print(f"Starting GA training...")
        sys.stdout.flush()
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            sys.stdout.flush()
            
            # Training with GA on forget data and GD on retain data
            train_forget_acc, train_retain_acc = self.ga_training_step(
                forget_loader, retain_loader, optimizer, retain_weight=retain_weight)
            
            # Validation
            print("Running validation...")
            sys.stdout.flush()
            forget_acc, retain_acc = self.validate(forget_classes, retain_classes)
            
            print(f"Epoch {epoch+1:2d}: Train_F {train_forget_acc:.1f}% Train_R {train_retain_acc:.1f}% | "
                  f"Val_F {forget_acc:.1f}% Val_R {retain_acc:.1f}%")
            sys.stdout.flush()
            
            # Model selection (minimize forget accuracy, maximize retain accuracy)
            unlearn_score = (100 - forget_acc) + 0.5 * retain_acc
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"New best model! Score: {unlearn_score:.2f}")
                sys.stdout.flush()
        
        # Load best model
        if best_model_state is not None:
            print("Loading best model...")
            sys.stdout.flush()
            self.model.load_state_dict(best_model_state)
            final_forget_acc, final_retain_acc = self.validate(forget_classes, retain_classes)
            print(f"Final: Forget {final_forget_acc:.1f}% | Retain {final_retain_acc:.1f}%")
        
        # Save unlearned model
        print(f"Saving model to {save_path}...")
        sys.stdout.flush()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'forget_classes': forget_classes,
            'retain_classes': retain_classes,
            'method': 'GA'
        }, save_path)
        print(f"Saved to {save_path}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Gradient Ascent Unlearning')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace')
    parser.add_argument('--output_dir', type=str, default='ga_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--retain_weight', type=float, default=1.0, 
                       help='Weight for retain loss')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set stdout to be unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    unlearner = GAUnlearner(model, device=args.device)
    
    tasks = [
        (1, 'step1.pth', 'ga_final_step1.pth'),  # 0-59 -> 10-59
        (2, 'step2.pth', 'ga_final_step2.pth'),  # 0-69 -> 20-69
        (3, 'step3.pth', 'ga_final_step3.pth'),  # 0-79 -> 30-79
        (4, 'step4.pth', 'ga_final_step4.pth'),  # 0-89 -> 40-89
        (5, 'step5.pth', 'ga_final_step5.pth'),  # 0-99 -> 50-99
    ]
    
    for step_num, checkpoint_file, output_file in tasks:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        save_path = os.path.join(args.output_dir, output_file)
        
        print(f"\n{'='*60}")
        print(f"GA Step {step_num}: Class Range Shift")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        unlearner.unlearn_step(
            step_num=step_num,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            lr=args.lr,
            epochs=args.epochs,
            retain_weight=args.retain_weight
        )


if __name__ == "__main__":
    main()