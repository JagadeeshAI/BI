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
from codes.utils import get_model
from codes.data import get_dynamic_loader

def create_uniclun_framework(num_classes=100, device='cuda', buffer_size=5120):
    """Create UniCLUN framework with proper model initialization"""
    
    def load_teachers(good_teacher_path, bad_teacher_path):
        """Load good and bad teachers for current step"""
        good_teacher = get_model(num_classes=num_classes, use_lora=False, pretrained=False).to(device)
        bad_teacher = get_model(num_classes=num_classes, use_lora=False, pretrained=False).to(device)
        
        good_teacher.load_state_dict(torch.load(good_teacher_path, map_location=device, weights_only=True))
        bad_teacher.load_state_dict(torch.load(bad_teacher_path, map_location=device, weights_only=True))
        
        good_teacher.eval()
        bad_teacher.eval()
        return good_teacher, bad_teacher
    
    def update_buffer(buffer, data_loader, target_classes, forget_classes):
        """Update replay buffer - add target class samples, remove forget class samples"""
        # Remove forget class samples
        buffer[:] = [item for item in buffer if item[1] not in forget_classes]
        
        # Add new target class samples using reservoir sampling
        new_samples = []
        for images, labels in data_loader:
            for i, label in enumerate(labels):
                if label.item() in target_classes:
                    new_samples.append((images[i].cpu(), label.item()))
        
        # Reservoir sampling to maintain buffer size
        for sample in new_samples:
            if len(buffer) < buffer_size:
                buffer.append(sample)
            else:
                j = random.randint(0, len(buffer))
                if j < buffer_size:
                    buffer[j] = sample
    
    def get_buffer_batch(buffer, batch_size=64):
        """Sample batch from buffer"""
        if len(buffer) == 0:
            return None, None
        
        batch_samples = random.sample(buffer, min(batch_size, len(buffer)))
        images = torch.stack([item[0] for item in batch_samples])
        labels = torch.tensor([item[1] for item in batch_samples])
        return images.to(device), labels.to(device)
    
    def get_features(model, x):
        """Extract features from penultimate layer"""
        # For ViT models, get features before the head
        if hasattr(model, 'head'):
            # Save original head
            original_head = model.head
            # Replace with identity
            model.head = nn.Identity()
            features = model(x)
            # Restore original head
            model.head = original_head
            return features
        else:
            # Fallback for other architectures
            return model(x)
    
    def compute_confidence_weight(good_teacher, x, y, rho=1.0):
        """Compute confidence weight ω(xi) from Eq. 3"""
        with torch.no_grad():
            logits = good_teacher(x)
            probs = F.softmax(logits / rho, dim=1)
            confidence = probs.gather(1, y.unsqueeze(1)).squeeze()
        return confidence
    
    def contrastive_loss(z_student, z_teacher, labels, tau=0.5):
        """Contrastive distillation loss from Eq. 5"""
        # Normalize embeddings
        z_student = F.normalize(z_student, p=2, dim=1)
        z_teacher = F.normalize(z_teacher, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_student, z_teacher.T) / tau
        
        # Create positive mask
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask.float()
        
        # Compute contrastive loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()
    
    def supervised_contrastive_loss(z_student, labels, tau=0.5):
        """Supervised contrastive loss from Eq. 7"""
        # Normalize embeddings
        z_student = F.normalize(z_student, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_student, z_student.T) / tau
        
        # Create positive mask (same class)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask.float()
        # Remove diagonal
        pos_mask.fill_diagonal_(0)
        
        # Compute supervised contrastive loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        pos_count = pos_mask.sum(dim=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        
        # Only compute loss for samples with positives
        valid_samples = pos_count > 0
        if valid_samples.sum() > 0:
            return loss[valid_samples].mean()
        return torch.tensor(0.0, device=device)
    
    return {
        'load_teachers': load_teachers,
        'update_buffer': update_buffer,
        'get_buffer_batch': get_buffer_batch,
        'get_features': get_features,
        'compute_confidence_weight': compute_confidence_weight,
        'contrastive_loss': contrastive_loss,
        'supervised_contrastive_loss': supervised_contrastive_loss
    }

def compute_unified_loss(student, good_teacher, bad_teacher, framework_funcs, buffer, images, labels, target_classes, forget_classes, gamma=1.0):
    """Compute unified loss from Eq. 11"""
    device = images.device
    
    # Hyperparameters from paper
    alpha1, alpha2, alpha3 = 0.5, 0.5, 0.5
    
    # Forward pass
    student_logits = student(images)
    
    # Get features for contrastive learning
    with torch.no_grad():
        teacher_features = framework_funcs['get_features'](good_teacher, images)
    student_features = framework_funcs['get_features'](student, images)
    
    # Classification loss (Eq. 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Online distillation loss (Eq. 4) - only for buffer samples
    od_loss = torch.tensor(0.0, device=device)
    buffer_images, buffer_labels = framework_funcs['get_buffer_batch'](buffer, batch_size=64)
    if buffer_images is not None:
        with torch.no_grad():
            teacher_logits = good_teacher(buffer_images)
        student_buffer_logits = student(buffer_images)
        
        # Confidence weighting
        weights = framework_funcs['compute_confidence_weight'](good_teacher, buffer_images, buffer_labels)
        mse_loss = F.mse_loss(student_buffer_logits, teacher_logits, reduction='none').mean(dim=1)
        od_loss = (weights * mse_loss).mean()
    
    # Contrastive distillation loss (Eq. 5)
    cd_loss = framework_funcs['contrastive_loss'](student_features, teacher_features, labels)
    
    # Supervised contrastive loss (Eq. 7)
    scd_loss = framework_funcs['supervised_contrastive_loss'](student_features, labels)
    
    # Continual learning loss (Eq. 8)
    cl_loss = ce_loss + alpha1 * od_loss + alpha2 * cd_loss + alpha3 * scd_loss
    
    # Unlearning loss (Eq. 9) - KL divergence with bad teacher for forget classes
    ul_loss = torch.tensor(0.0, device=device)
    forget_mask = torch.tensor([label.item() in forget_classes for label in labels], device=device)
    
    if forget_mask.sum() > 0:
        forget_images = images[forget_mask]
        with torch.no_grad():
            bad_teacher_logits = bad_teacher(forget_images)
        student_forget_logits = student(forget_images)
        
        # KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(student_forget_logits, dim=1),
            F.softmax(bad_teacher_logits, dim=1),
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

def evaluate_comprehensive(student, step_num):
    """Evaluate on all 100 classes to show performance distribution"""
    device = next(student.parameters()).device
    student.eval()
    
    # Test on all classes 0-99
    test_loader = get_dynamic_loader(class_range=(0, 99), mode="val", batch_size=32)
    
    class_correct = torch.zeros(100)
    class_total = torch.zeros(100)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (preds[i] == labels[i]).item()
                class_total[label] += 1
    
    # Calculate accuracies
    class_accuracies = []
    for i in range(100):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(acc.item())
        else:
            class_accuracies.append(0.0)
    
    # Print results by ranges
    step_ranges = {
        1: {'target': (10, 59), 'forget': (0, 9), 'other': (60, 99)},
        2: {'target': (20, 69), 'forget': (10, 19), 'other': [(0, 9), (70, 99)]},
        3: {'target': (30, 79), 'forget': (20, 29), 'other': [(0, 19), (80, 99)]},
        4: {'target': (40, 89), 'forget': (30, 39), 'other': [(0, 29), (90, 99)]},
        5: {'target': (50, 99), 'forget': (40, 49), 'other': (0, 39)}
    }
    
    if step_num in step_ranges:
        config = step_ranges[step_num]
        
        # Target classes (should be high)
        target_start, target_end = config['target']
        target_acc = np.mean(class_accuracies[target_start:target_end+1])
        
        # Forget classes (should be low)
        if isinstance(config['forget'], tuple):
            forget_start, forget_end = config['forget']
            forget_acc = np.mean(class_accuracies[forget_start:forget_end+1])
        else:
            forget_acc = np.mean([class_accuracies[i] for i in config['forget']])
        
        # Other classes (should be low)
        if isinstance(config['other'], list):
            other_acc = np.mean([np.mean(class_accuracies[start:end+1]) for start, end in config['other']])
        else:
            other_start, other_end = config['other']
            other_acc = np.mean(class_accuracies[other_start:other_end+1])
        
        print(f"\nStep {step_num} Comprehensive Evaluation:")
        print(f"Target classes {config['target']}: {target_acc:.2f}%")
        print(f"Forget classes {config['forget']}: {forget_acc:.2f}%")
        print(f"Other classes: {other_acc:.2f}%")
        
    return class_accuracies

def train_step(student, framework_funcs, step_num, target_range, num_epochs=40):
    """Train student for one step"""
    device = next(student.parameters()).device
    buffer = []
    
    target_classes = list(range(target_range[0], target_range[1] + 1))
    
    # Determine forget classes based on step
    if step_num == 1:
        forget_classes = list(range(0, 10))  # Forget 0-9
    else:
        forget_classes = list(range((step_num-2)*10, (step_num-1)*10))  # Previous step classes
    
    print(f"Step {step_num}: Target {target_range}, Forget {forget_classes}")
    
    # Load appropriate teachers
    good_teacher_path = f"./baseline/teachers/student_{target_range[0]}_{target_range[1]}_good_teacher.pth"
    bad_teacher_path = f"./baseline/teachers/student_{target_range[0]}_{target_range[1]}_bad_teacher.pth"
    good_teacher, bad_teacher = framework_funcs['load_teachers'](good_teacher_path, bad_teacher_path)
    
    # Get data loaders
    train_loader = get_dynamic_loader(class_range=target_range, mode="train", batch_size=64)
    
    # Update buffer
    framework_funcs['update_buffer'](buffer, train_loader, target_classes, forget_classes)
    
    # Training setup
    optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(train_loader, desc=f"Step {step_num} Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Compute unified loss
            loss, loss_dict = compute_unified_loss(
                student, good_teacher, bad_teacher, framework_funcs, buffer,
                images, labels, target_classes, forget_classes
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = student(images).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Validation
        val_acc = evaluate_comprehensive(student, step_num)
        train_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.3f}, Loss {total_loss/len(train_loader):.4f}")
        
        # Use target class accuracy for best model selection
        if step_num == 1:
            current_val_acc = np.mean(val_acc[10:60])  # Classes 10-59
        elif step_num == 2:
            current_val_acc = np.mean(val_acc[20:70])  # Classes 20-69
        elif step_num == 3:
            current_val_acc = np.mean(val_acc[30:80])  # Classes 30-79
        elif step_num == 4:
            current_val_acc = np.mean(val_acc[40:90])  # Classes 40-89
        else:  # step_num == 5
            current_val_acc = np.mean(val_acc[50:100])  # Classes 50-99
        
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            save_path = f"./baseline/students/step{step_num}.pth"
            torch.save(student.state_dict(), save_path)
            print(f"Best model saved: {save_path} (Target Acc: {current_val_acc:.2f}%)")
        
        scheduler.step()

def main():
    os.makedirs("./baseline/students", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize student model
    student = get_model(num_classes=100, use_lora=False, pretrained=True).to(device)
    
    # Create framework functions
    framework_funcs = create_uniclun_framework(device=device)
    
    # Step configurations
    steps = [
        (1, (10, 59)),
        (2, (20, 69)),
        (3, (30, 79)),
        (4, (40, 89)),
        (5, (50, 99))
    ]
    
    for step_num, target_range in steps:
        print(f"\n{'='*50}")
        print(f"Starting Step {step_num}: Classes {target_range[0]}-{target_range[1]}")
        print(f"{'='*50}")
        
        train_step(student, framework_funcs, step_num, target_range)
        
        print(f"Step {step_num} completed!")

if __name__ == "__main__":
    main()