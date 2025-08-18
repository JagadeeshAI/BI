import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import random
import copy
from codes.utils import get_model, load_model_weights
from codes.data import get_dynamic_loader

class UniCLUNFramework:
    def __init__(self, device='cuda', buffer_size=5120):
        self.device = device
        self.buffer_size = buffer_size
        self.replay_buffer = []
    
    def update_buffer(self, data_loader, target_classes, forget_classes):
        """Update replay buffer using reservoir sampling"""
        # Remove forget class samples
        self.replay_buffer = [item for item in self.replay_buffer if item[1] not in forget_classes]
        
        # Add new target class samples
        new_samples = []
        for images, labels in data_loader:
            for i, label in enumerate(labels):
                if label.item() in target_classes:
                    new_samples.append((images[i].cpu(), label.item()))
        
        # Reservoir sampling
        for sample in new_samples:
            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append(sample)
            else:
                j = random.randint(0, len(self.replay_buffer))
                if j < self.buffer_size:
                    self.replay_buffer[j] = sample
    
    def get_buffer_batch(self, batch_size=64):
        """Sample batch from replay buffer"""
        if len(self.replay_buffer) == 0:
            return None, None
        
        batch_samples = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        images = torch.stack([item[0] for item in batch_samples])
        labels = torch.tensor([item[1] for item in batch_samples])
        return images.to(self.device), labels.to(self.device)
    
    def get_features(self, model, x):
        """Extract features before classification head"""
        if hasattr(model, 'head'):
            # Save original head and replace with identity
            original_head = model.head
            model.head = nn.Identity()
            features = model(x)
            model.head = original_head
            return features
        else:
            return model(x)
    
    def compute_confidence_weight(self, teacher, x, y, rho=1.0):
        """Compute confidence weight ω(xi) from Eq. 3"""
        with torch.no_grad():
            logits = teacher(x)
            probs = F.softmax(logits / rho, dim=1)
            confidence = probs.gather(1, y.unsqueeze(1)).squeeze()
        return confidence
    
    def contrastive_loss(self, z_student, z_teacher, labels, tau=0.5):
        """Contrastive distillation loss from Eq. 5"""
        z_student = F.normalize(z_student, p=2, dim=1)
        z_teacher = F.normalize(z_teacher, p=2, dim=1)
        
        sim_matrix = torch.matmul(z_student, z_teacher.T) / tau
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()
    
    def supervised_contrastive_loss(self, z_student, labels, tau=0.5):
        """Supervised contrastive loss from Eq. 7"""
        z_student = F.normalize(z_student, p=2, dim=1)
        sim_matrix = torch.matmul(z_student, z_student.T) / tau
        
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        pos_count = pos_mask.sum(dim=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        
        valid_samples = pos_count > 0
        if valid_samples.sum() > 0:
            return loss[valid_samples].mean()
        return torch.tensor(0.0, device=self.device)
    
    def compute_unified_loss(self, student, cl_teacher, ul_teacher, images, labels, target_classes, forget_classes, gamma=1.0):
        """Compute unified loss from Eq. 11"""
        # Hyperparameters
        alpha1, alpha2, alpha3 = 0.5, 0.5, 0.5
        
        # Forward pass
        student_logits = student(images)
        
        # Classification loss (Eq. 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Online distillation loss (Eq. 4) - only for buffer samples
        od_loss = torch.tensor(0.0, device=self.device)
        buffer_images, buffer_labels = self.get_buffer_batch(batch_size=64)
        if buffer_images is not None:
            with torch.no_grad():
                teacher_logits = cl_teacher(buffer_images)
            student_buffer_logits = student(buffer_images)
            
            weights = self.compute_confidence_weight(cl_teacher, buffer_images, buffer_labels)
            mse_loss = F.mse_loss(student_buffer_logits, teacher_logits, reduction='none').mean(dim=1)
            od_loss = (weights * mse_loss).mean()
        
        # Get features for contrastive learning
        with torch.no_grad():
            teacher_features = self.get_features(cl_teacher, images)
        student_features = self.get_features(student, images)
        
        # Contrastive distillation loss (Eq. 5)
        cd_loss = self.contrastive_loss(student_features, teacher_features, labels)
        
        # Supervised contrastive loss (Eq. 7)
        scd_loss = self.supervised_contrastive_loss(student_features, labels)
        
        # Continual learning loss (Eq. 8)
        cl_loss = ce_loss + alpha1 * od_loss + alpha2 * cd_loss + alpha3 * scd_loss
        
        # Unlearning loss (Eq. 9) - KL divergence with UL teacher for forget classes
        ul_loss = torch.tensor(0.0, device=self.device)
        forget_mask = torch.tensor([label.item() in forget_classes for label in labels], device=self.device)
        
        if forget_mask.sum() > 0:
            forget_images = images[forget_mask]
            with torch.no_grad():
                ul_teacher_logits = ul_teacher(forget_images)
            student_forget_logits = student(forget_images)
            
            kl_loss = F.kl_div(
                F.log_softmax(student_forget_logits, dim=1),
                F.softmax(ul_teacher_logits, dim=1),
                reduction='batchmean'
            )
            ul_loss = kl_loss
        
        # Unified loss (Eq. 11)
        total_loss = gamma * cl_loss + (1 - gamma) * ul_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'od': od_loss.item(),
            'cd': cd_loss.item(),
            'scd': scd_loss.item(),
            'ul': ul_loss.item()
        }
    
    def momentum_update(self, cl_teacher, student, momentum=0.999):
        """Update CL teacher using momentum from student (Eq. in paper)"""
        with torch.no_grad():
            for teacher_param, student_param in zip(cl_teacher.parameters(), student.parameters()):
                # Momentum update: θT ← mθT + (1-m)θS
                teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

def evaluate_step(student, step_num, device):
    """Evaluate student performance for current step"""
    student.eval()
    
    # Define expected ranges for each step
    step_configs = {
        1: {'target': (10, 59), 'forget': (0, 9)},
        2: {'target': (20, 69), 'forget': (10, 19)},
        3: {'target': (30, 79), 'forget': (20, 29)},
        4: {'target': (40, 89), 'forget': (30, 39)},
        5: {'target': (50, 99), 'forget': (40, 49)}
    }
    
    config = step_configs[step_num]
    
    # Test on target classes
    target_loader = get_dynamic_loader(class_range=config['target'], mode="val", batch_size=32)
    target_correct, target_total = 0, 0
    
    with torch.no_grad():
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)
            target_correct += (preds == labels).sum().item()
            target_total += labels.size(0)
    
    target_acc = 100 * target_correct / target_total if target_total > 0 else 0
    
    # Test on forget classes
    forget_loader = get_dynamic_loader(class_range=config['forget'], mode="val", batch_size=32)
    forget_correct, forget_total = 0, 0
    
    with torch.no_grad():
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)
            forget_correct += (preds == labels).sum().item()
            forget_total += labels.size(0)
    
    forget_acc = 100 * forget_correct / forget_total if forget_total > 0 else 0
    
    print(f"Step {step_num} Evaluation:")
    print(f"  Target classes {config['target']}: {target_acc:.2f}%")
    print(f"  Forget classes {config['forget']}: {forget_acc:.2f}%")
    
    return target_acc, forget_acc

def run_uniclun_step(framework, cl_teacher, student, step_num, target_range, base_ckpt_path, num_epochs=40):
    """Run one UniCLUN step"""
    device = framework.device
    
    # Define classes
    target_classes = list(range(target_range[0], target_range[1] + 1))
    if step_num == 1:
        forget_classes = list(range(0, 10))  # Forget 0-9
    else:
        forget_classes = list(range((step_num-2)*10, (step_num-1)*10))  # Previous step classes
    
    print(f"\nStep {step_num}: Target {target_range}, Forget {forget_classes}")
    
    # Create UL teacher (random initialization) - THIS IS KEY!
    ul_teacher = get_model(num_classes=100, use_lora=False, pretrained=False).to(device)
    ul_teacher.eval()  # Always in eval mode
    
    # Get data loader
    train_loader = get_dynamic_loader(class_range=target_range, mode="train", batch_size=64)
    
    # Update replay buffer
    framework.update_buffer(train_loader, target_classes, forget_classes)
    
    # Training setup
    optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_target_acc = 0.0
    
    epoch_pbar = tqdm(range(num_epochs), desc=f"Step {step_num}", unit="epoch")
    
    for epoch in epoch_pbar:
        student.train()
        total_loss = 0
        batch_losses = []
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Determine gamma (learning vs unlearning weight)
            has_forget_samples = any(label.item() in forget_classes for label in labels)
            gamma = 0.5 if has_forget_samples else 1.0  # More unlearning when forget samples present
            
            # Compute unified loss
            loss, loss_dict = framework.compute_unified_loss(
                student, cl_teacher, ul_teacher, images, labels, 
                target_classes, forget_classes, gamma
            )
            
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'Loss': f"{batch_loss:.4f}",
                'Avg': f"{np.mean(batch_losses[-10:]):.4f}"  # Rolling average
            })
        
        batch_pbar.close()
        
        # Evaluate
        eval_pbar = tqdm([1], desc="Evaluating", leave=False)
        for _ in eval_pbar:
            target_acc, forget_acc = evaluate_step(student, step_num, device)
        eval_pbar.close()
        
        avg_loss = total_loss / len(train_loader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Loss': f"{avg_loss:.4f}",
            'Target%': f"{target_acc:.1f}",
            'Forget%': f"{forget_acc:.1f}",
            'Best%': f"{best_target_acc:.1f}"
        })
        
        # Save best model based on target accuracy
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            save_path = f"./baseline/students/step{step_num}.pth"
            torch.save(student.state_dict(), save_path)
            epoch_pbar.write(f"✅ New best: {target_acc:.2f}% → {save_path}")
        
        scheduler.step()
        
        # Update CL teacher from student using momentum
        framework.momentum_update(cl_teacher, student, momentum=0.999)
    
    epoch_pbar.close()
    
    return cl_teacher, student

def main():
    """Main UniCLUN training pipeline"""
    os.makedirs("./baseline/students", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt_path = "/home/jag/codes/Bi/checkpoints/oracle/0_49.pth"
    
    # Initialize framework
    framework = UniCLUNFramework(device=device, buffer_size=5120)
    
    # Initialize models - CL teacher starts from 0-49.pth
    cl_teacher = get_model(num_classes=100, use_lora=False, pretrained=False).to(device)
    load_model_weights(cl_teacher, base_ckpt_path)
    
    # Student starts as copy of CL teacher
    student = get_model(num_classes=100, use_lora=False, pretrained=False).to(device)
    student.load_state_dict(cl_teacher.state_dict())
    
    # Step configurations
    steps = [
        (1, (10, 59)),
        (2, (20, 69)),
        (3, (30, 79)),
        (4, (40, 89)),
        (5, (50, 99))
    ]
    
    print("🚀 Starting UniCLUN Sequential Training")
    print(f"Starting from checkpoint: {base_ckpt_path}")
    
    for step_num, target_range in steps:
        print(f"\n{'='*60}")
        print(f"Starting Step {step_num}: Classes {target_range[0]}-{target_range[1]}")
        print(f"{'='*60}")
        
        # Run UniCLUN step
        cl_teacher, student = run_uniclun_step(
            framework, cl_teacher, student, step_num, target_range, base_ckpt_path
        )
        
        print(f"✅ Step {step_num} completed!")
    
    print(f"\n🎉 All 5 UniCLUN steps completed!")
    print(f"📁 Student models saved in: ./baseline/students/")

if __name__ == "__main__":
    main()