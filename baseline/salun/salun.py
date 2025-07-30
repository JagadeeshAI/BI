import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model


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

    def unlearn_step(self, step_num, checkpoint_path, save_path, target_classes,
                     lr=0.01, epochs=10, sparsity=0.5):
        """Main SALUN unlearning following original paper methodology"""
        
        # Load pretrained model
        print(f"Loading checkpoint: {checkpoint_path}")
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
        initial_forget_acc, initial_retain_acc = self.validate(target_classes, retain_classes)
        print(f"Initial: Forget {initial_forget_acc:.1f}% | Retain {initial_retain_acc:.1f}%")
        
        # Compute weight saliency mask (Algorithm 1, Step 1)
        saliency_mask = self.compute_weight_saliency_mask(class_range, target_classes, sparsity)
        
        # Create combined training dataset (Algorithm 1, Step 2)
        train_loader, original_labels = self.create_combined_dataset(
            class_range, target_classes, batch_size=128)
        
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
            forget_acc, retain_acc = self.validate(target_classes, retain_classes)
            
            print(f"Epoch {epoch+1:2d}: Train {train_acc:.1f}% | Forget {forget_acc:.1f}% | Retain {retain_acc:.1f}%")
            
            # Model selection
            unlearn_score = (100 - forget_acc) + 0.5 * max(0, retain_acc - 30)
            if unlearn_score > best_score:
                best_score = unlearn_score
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            final_forget_acc, final_retain_acc = self.validate(target_classes, retain_classes)
            print(f"Best model: Forget {final_forget_acc:.1f}% | Retain {final_retain_acc:.1f}%")
        
        # Save unlearned model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step_num,
            'target_classes': target_classes,
            'method': 'SALUN',
            'sparsity': sparsity
        }, save_path)
        print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='SALUN Implementation')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/home/jag/codes/Bi/baseline/ER/checkpoints/ace')
    parser.add_argument('--output_dir', type=str, default='salun_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Saliency sparsity')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    unlearner = SALUNUnlearner(model, device=args.device)
    
    tasks = [
        (1, 'step1.pth', 'salun_step1.pth', list(range(0, 10))),
        (2, 'step2.pth', 'salun_step2.pth', list(range(0, 20))),
        (3, 'step3.pth', 'salun_step3.pth', list(range(0, 30))),
        (4, 'step4.pth', 'salun_step4.pth', list(range(0, 40))),
        (5, 'step5.pth', 'salun_step5.pth', list(range(0, 50))),
    ]
    
    for step_num, checkpoint_file, output_file, target_classes in tasks:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        save_path = os.path.join(args.output_dir, output_file)
        
        print(f"\n{'='*60}")
        print(f"SALUN Step {step_num}: Classes {target_classes}")
        print(f"{'='*60}")
        
        unlearner.unlearn_step(
            step_num=step_num,
            checkpoint_path=checkpoint_path,
            save_path=save_path,
            target_classes=target_classes,
            lr=args.lr,
            epochs=10,
            sparsity=args.sparsity
        )


if __name__ == "__main__":
    main()