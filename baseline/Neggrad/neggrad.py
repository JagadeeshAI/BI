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

class NegGradUnlearner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def validate(self, target_classes, retain_classes):
        """Validation on forget and retain sets"""
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

    def create_forget_dataset(self, class_range, target_classes, batch_size=128):
        """Create forget dataset"""
        forget_loader = get_dynamic_loader(
            class_range=(min(target_classes), max(target_classes)),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        forget_data = []
        forget_labels = []
        max_samples_per_class = 500
        class_counts = {cls: 0 for cls in target_classes}
        
        for batch_data, batch_labels in forget_loader:
            for i, label in enumerate(batch_labels):
                label_item = label.item()
                if (label_item in target_classes and 
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

    def create_retain_dataset(self, class_range, target_classes, batch_size=128):
        """Create retain dataset"""
        retain_start = max(target_classes) + 1
        retain_end = class_range[1]
        
        if retain_start > retain_end:
            return None
            
        retain_loader = get_dynamic_loader(
            class_range=(retain_start, retain_end),
            mode="train", batch_size=batch_size, num_workers=0
        )
        
        retain_data = []
        retain_labels = []
        max_samples = 2000
        
        for batch_data, batch_labels in retain_loader:
            for i, label in enumerate(batch_labels):
                if len(retain_data) < max_samples:
                    retain_data.append(batch_data[i])
                    retain_labels.append(label.item())
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

    def neggrad_training_step(self, forget_loader, retain_loader, optimizer, alpha=0.9):
        """
        NegGrad+ training step:
        - alpha=0: Pure NegGrad (gradient ascent on forget data only)
        - alpha>0: NegGrad+ (balanced forget + retain)
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
        top1 = AverageMeter()
        
        self.model.train()
        retain_iter = iter(retain_loader) if retain_loader else None
        criterion = nn.CrossEntropyLoss()
        
        # Fixed tqdm configuration for better visibility
        total_batches = len(forget_loader)
        pbar = tqdm(forget_loader, 
                   desc="NegGrad Training", 
                   leave=True, 
                   ncols=100, 
                   disable=False, 
                   file=sys.stdout,
                   dynamic_ncols=False)
        
        for i, (forget_data, forget_target) in enumerate(pbar):
            forget_data = forget_data.to(self.device)
            forget_target = forget_target.to(self.device)
            
            # Forget loss (gradient ascent - maximize loss)
            forget_output = self.model(forget_data)
            forget_loss = criterion(forget_output, forget_target)
            
            total_loss = -0.1 * forget_loss  # Scaled negative for gradient ascent
            
            # Retain loss (gradient descent - minimize loss)
            if alpha > 0 and retain_iter:
                try:
                    retain_data, retain_target = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_data, retain_target = next(retain_iter)
                
                retain_data = retain_data.to(self.device)
                retain_target = retain_target.to(self.device)
                retain_output = self.model(retain_data)
                retain_loss = criterion(retain_output, retain_target)
                
                # NegGrad+ combination: scaled forget ascent + retain descent
                total_loss = -0.1 * forget_loss + alpha * retain_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            prec1 = accuracy(forget_output.data, forget_target)[0]
            losses.update(total_loss.item(), forget_data.size(0))
            top1.update(prec1.item(), forget_data.size(0))
            
            # Update progress bar with forced refresh
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.3f}',
                'Batch': f'{i+1}/{total_batches}'
            })
            pbar.refresh()
            sys.stdout.flush()
            
            # Additional manual progress print every 20 batches for backup visibility
            if i % 20 == 0:
                print(f"\nProgress: {i+1}/{total_batches} batches - Loss: {losses.avg:.4f}, Acc: {top1.avg:.3f}")
                sys.stdout.flush()
        
        # Final progress update
        print(f"\nTraining completed: {total_batches}/{total_batches} batches - Final Loss: {losses.avg:.4f}, Final Acc: {top1.avg:.3f}")
        sys.stdout.flush()
        
        return top1.avg

    def unlearn_step(self, step_num, checkpoint_path, save_path, target_classes,
                     lr=0.01, epochs=10, alpha=0.9):
        """
        Main NegGrad/NegGrad+ unlearning procedure
        alpha=0: Pure NegGrad
        alpha>0: NegGrad+ with retain regularization
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
        
        # Define class ranges per step
        step_ranges = {1: (0, 59), 2: (0, 69), 3: (0, 79), 4: (0, 89), 5: (0, 99)}
        class_range = step_ranges[step_num]
        
        # Define retain classes for validation
        retain_starts = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
        retain_classes = list(range(retain_starts[step_num], class_range[1] + 1))
        
        # Initial validation
        print("Running initial validation...")
        sys.stdout.flush()
        initial_forget_acc, initial_retain_acc = self.validate(target_classes, retain_classes)
        print(f"Initial: Forget {initial_forget_acc:.1f}% | Retain {initial_retain_acc:.1f}%")
        sys.stdout.flush()
        
        # Create datasets
        print("Creating datasets...")
        sys.stdout.flush()
        forget_loader = self.create_forget_dataset(class_range, target_classes, batch_size=128)
        retain_loader = self.create_retain_dataset(class_range, target_classes, batch_size=128)
        
        if forget_loader is None:
            print("Failed to create forget data")
            return
        
        print(f"Forget dataset size: {len(forget_loader)} batches")
        if retain_loader:
            print(f"Retain dataset size: {len(retain_loader)} batches")
        sys.stdout.flush()
        
        # Setup optimizer (more conservative settings)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.001,  # Much lower learning rate
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        # NegGrad/NegGrad+ training loop
        best_model_state = None
        best_score = 0
        method_name = "NegGrad+" if alpha > 0 else "NegGrad"
        
        print(f"Starting {method_name} training (alpha={alpha})...")
        sys.stdout.flush()
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            sys.stdout.flush()
            
            # Training with gradient ascent on forget data
            train_acc = self.neggrad_training_step(
                forget_loader, retain_loader, optimizer, alpha=alpha)
            
            # Validation
            print("Running validation...")
            sys.stdout.flush()
            forget_acc, retain_acc = self.validate(target_classes, retain_classes)
            
            print(f"Epoch {epoch+1:2d}: Train {train_acc:.1f}% | "
                  f"Forget {forget_acc:.1f}% | Retain {retain_acc:.1f}%")
            sys.stdout.flush()
            
            # Model selection (prioritize low forget accuracy + good retain accuracy)
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 30)
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"New best model found! Score: {unlearn_score:.2f}")
                sys.stdout.flush()
        
        # Load best model
        if best_model_state is not None:
            print("Loading best model...")
            sys.stdout.flush()
            self.model.load_state_dict(best_model_state)
            final_forget_acc, final_retain_acc = self.validate(target_classes, retain_classes)
            print(f"Best model: Forget {final_forget_acc:.1f}% | Retain {final_retain_acc:.1f}%")
        
        # Save unlearned model
        print(f"Saving model to {save_path}...")
        sys.stdout.flush()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'target_classes': target_classes,
            'method': method_name,
            'alpha': alpha
        }, save_path)
        print(f"Saved to {save_path}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='NegGrad/NegGrad+ Unlearning')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace')
    parser.add_argument('--output_dir', type=str, default='neggrad_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--alpha', type=float, default=0.9, 
                       help='Alpha for NegGrad+ (0=pure NegGrad, >0=NegGrad+)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set stdout to be unbuffered for immediate output
    sys.stdout.reconfigure(line_buffering=True)
    
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    unlearner = NegGradUnlearner(model, device=args.device)
    
    tasks = [
        (1, 'step1.pth', 'neggrad_step1.pth', list(range(0, 10))),
        (2, 'step2.pth', 'neggrad_step2.pth', list(range(0, 20))),
        (3, 'step3.pth', 'neggrad_step3.pth', list(range(0, 30))),
        (4, 'step4.pth', 'neggrad_step4.pth', list(range(0, 40))),
        (5, 'step5.pth', 'neggrad_step5.pth', list(range(0, 50))),
    ]
    
    for step_num, checkpoint_file, output_file, target_classes in tasks:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        save_path = os.path.join(args.output_dir, output_file)
        
        method_name = "NegGrad+" if args.alpha > 0 else "NegGrad"
        print(f"\n{'='*60}")
        print(f"{method_name} Step {step_num}: Classes {target_classes}")
        print(f"Alpha: {args.alpha}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        unlearner.unlearn_step(
            step_num=step_num,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            target_classes=target_classes,
            lr=args.lr,
            epochs=args.epochs,
            alpha=args.alpha
        )


if __name__ == "__main__":
    main()