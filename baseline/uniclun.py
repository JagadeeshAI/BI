import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from codes.data import get_dynamic_loader
from codes.utils import get_model, load_model_weights, print_parameter_stats
import random
from tqdm import tqdm
import math

# -------------------
# GPU-Optimized Mixed Dataset
# -------------------
class GPUOptimizedMixedDataset(Dataset):
    def __init__(self, learn_range, unlearn_range, retain_range, mode="train", data_percentage=1.0, device=None):
        """GPU-optimized dataset with forced device placement"""
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if str(device) == 'cuda':
            device = torch.device("cuda:0")
            
        self.device = device
        
        print(f"🚀 Initializing dataset on device: {self.device}")
        
        self.samples = []
        max_samples_per_range = 2000
        
        # Load learn samples
        learn_loader = get_dynamic_loader(learn_range, mode=mode, batch_size=32, data_percentage=data_percentage)
        learn_count = 0
        for batch_imgs, batch_labels in learn_loader:
            for i in range(len(batch_imgs)):
                if learn_count >= max_samples_per_range:
                    break
                img = batch_imgs[i].to(self.device, non_blocking=True)
                label = batch_labels[i].to(self.device, non_blocking=True)
                ulabel = torch.tensor(0, device=self.device)  # Learn/Retain
                self.samples.append((img, label, ulabel))
                learn_count += 1
            if learn_count >= max_samples_per_range:
                break
        
        # Load unlearn samples
        unlearn_loader = get_dynamic_loader(unlearn_range, mode=mode, batch_size=32, data_percentage=data_percentage)
        unlearn_count = 0
        for batch_imgs, batch_labels in unlearn_loader:
            for i in range(len(batch_imgs)):
                if unlearn_count >= max_samples_per_range:
                    break
                img = batch_imgs[i].to(self.device, non_blocking=True)
                label = batch_labels[i].to(self.device, non_blocking=True)
                ulabel = torch.tensor(1, device=self.device)  # Unlearn
                self.samples.append((img, label, ulabel))
                unlearn_count += 1
            if unlearn_count >= max_samples_per_range:
                break
        
        # Load retain samples  
        retain_loader = get_dynamic_loader(retain_range, mode=mode, batch_size=32, data_percentage=data_percentage)
        retain_count = 0
        for batch_imgs, batch_labels in retain_loader:
            for i in range(len(batch_imgs)):
                if retain_count >= max_samples_per_range:
                    break
                img = batch_imgs[i].to(self.device, non_blocking=True)
                label = batch_labels[i].to(self.device, non_blocking=True)
                ulabel = torch.tensor(0, device=self.device)  # Learn/Retain
                self.samples.append((img, label, ulabel))
                retain_count += 1
            if retain_count >= max_samples_per_range:
                break
        
        random.shuffle(self.samples)
        
        print(f"📊 Dataset loaded: {len(self.samples)} samples")
        print(f"   📚 Learn: {learn_count} samples")  
        print(f"   🗑️ Unlearn: {unlearn_count} samples")
        print(f"   💾 Retain: {retain_count} samples")
        print(f"   🔥 All data pre-loaded to GPU: {self.device}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, ulabel = self.samples[idx]
        return img, label, ulabel

# -------------------
# Buffer
# -------------------
class Buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.examples, self.labels, self.ulabels = None, None, None

    def reservoir(self, num_seen_examples, buffer_size):
        if num_seen_examples < buffer_size:
            return num_seen_examples
        rand = np.random.randint(0, num_seen_examples + 1)
        return rand if rand < buffer_size else -1

    def add_data(self, examples, labels, ulabels):
        if self.examples is None:
            self.examples = torch.zeros((self.buffer_size, *examples.shape[1:]), device=self.device)
            self.labels = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
            self.ulabels = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
        for i in range(examples.shape[0]):
            index = self.reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i]
                self.labels[index] = labels[i]
                self.ulabels[index] = ulabels[i]

    def get_data(self, size):
        if self.num_seen_examples == 0:
            return None, None, None
        size = min(size, self.num_seen_examples, self.buffer_size)
        choice = np.random.choice(min(self.num_seen_examples, self.buffer_size), size=size, replace=False)
        return self.examples[choice], self.labels[choice], self.ulabels[choice]

    def remove_class_range(self, class_range):
        if self.examples is None:
            return
        start_class, end_class = class_range
        mask = ~((self.labels >= start_class) & (self.labels <= end_class))
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            self.examples[:len(valid_indices)] = self.examples[valid_indices]
            self.labels[:len(valid_indices)] = self.labels[valid_indices]
            self.ulabels[:len(valid_indices)] = self.ulabels[valid_indices]
            self.examples[len(valid_indices):] = 0
            self.labels[len(valid_indices):] = 0
            self.ulabels[len(valid_indices):] = 0
            self.num_seen_examples = len(valid_indices)

# -------------------
# Projector Network (Missing from your implementation)
# -------------------
class ProjectorHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)

# -------------------
# Enhanced UniCLUN Model (Paper-aligned)
# -------------------
class UniCLUNModel(nn.Module):
    def __init__(self, backbone, num_classes=100, buffer_size=200):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = next(backbone.parameters()).device

        # Get feature dimension - Handle both CNN and ViT
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            if hasattr(self.backbone, 'features'):
                # ResNet/CNN case
                features = self.backbone.features(dummy_input)
                self.feature_dim = features.view(features.size(0), -1).size(1)
            else:
                # Vision Transformer case
                # Get features from before the final head
                self.feature_dim = self.backbone.embed_dim

        # Teachers - Following paper's approach
        self.cl_teacher = get_model(num_classes=num_classes, pretrained=False).to(self.device)
        self.ul_teacher = get_model(num_classes=num_classes, pretrained=False).to(self.device)
        
        # Initialize CL teacher with backbone weights
        self.cl_teacher.load_state_dict(backbone.state_dict())
        
        # Initialize UL teacher with random weights (bad teacher)
        # This is key for unlearning - it should provide "bad" guidance
        
        # Add projector heads for contrastive learning (Missing in your implementation)
        self.student_projector = ProjectorHead(self.feature_dim).to(self.device)
        self.teacher_projector = ProjectorHead(self.feature_dim).to(self.device)
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())

        self.buffer = Buffer(buffer_size, self.device)

        # Paper's hyperparameters (Table 11)
        self.alpha1 = 1.0   # Online distillation weight (Lod)
        self.alpha2 = 0.5   # Contrastive distillation weight (Lcd)  
        self.alpha3 = 0.5   # Supervised contrastive weight (Lscd)
        self.beta = 0.1     # UL weight - much weaker than your original
        self.gamma = 1.0    # Replay weight
        self.temperature = 0.5  # For contrastive learning
        
        print(f"🔥 Enhanced UniCLUN model initialized on device: {self.device}")

    def get_features(self, x, model=None):
        """Extract features from backbone - supports both CNN and ViT"""
        if model is None:
            model = self.backbone
            
        if hasattr(model, 'features'):
            # ResNet/CNN case
            features = model.features(x)
            return features.view(features.size(0), -1)
        else:
            # Vision Transformer case
            B = x.shape[0]
            x = model.patch_embed(x)
            cls_tokens = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + model.pos_embed
            x = model.pos_drop(x)
            
            # Pass through transformer blocks
            x = model.blocks(x)
            x = model.norm(x)
            
            # Return CLS token features (before final head)
            return x[:, 0]

    def contrastive_distillation_loss(self, student_features, teacher_features, labels):
        """Equation 5 from paper - Contrastive distillation loss"""
        # Get embeddings
        student_emb = self.student_projector(student_features)
        teacher_emb = self.teacher_projector(teacher_features)
        
        # Normalize embeddings
        student_emb = F.normalize(student_emb, p=2, dim=1)
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        
        batch_size = student_emb.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            # Find positive teacher embeddings (same label)
            positive_mask = (labels == labels[i])
            if positive_mask.sum() <= 1:  # Only current sample
                continue
                
            positive_teacher = teacher_emb[positive_mask]
            
            # Compute similarities
            sim_positive = torch.exp(torch.mm(student_emb[i:i+1], positive_teacher.t()) / self.temperature)
            sim_all = torch.exp(torch.mm(student_emb[i:i+1], teacher_emb.t()) / self.temperature)
            
            # Contrastive loss
            loss += -torch.log(sim_positive.sum() / sim_all.sum())
        
        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)

    def supervised_contrastive_loss(self, student_features, labels):
        """Equation 7 from paper - Supervised contrastive loss"""
        embeddings = self.student_projector(student_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        batch_size = embeddings.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            # Find positive samples (same label)
            positive_mask = (labels == labels[i])
            positive_mask[i] = False  # Exclude self
            
            if not positive_mask.any():
                continue
                
            positive_embeddings = embeddings[positive_mask]
            
            # Compute similarities
            sim_positive = torch.exp(torch.mm(embeddings[i:i+1], positive_embeddings.t()) / self.temperature)
            sim_all = torch.exp(torch.mm(embeddings[i:i+1], embeddings.t()) / self.temperature)
            sim_all = sim_all - torch.exp(torch.tensor(1.0 / self.temperature))  # Remove self-similarity
            
            # Supervised contrastive loss
            loss += -torch.log(sim_positive.sum() / sim_all.sum())
        
        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)

    def compute_sample_weights(self, inputs, labels):
        """Equation 3 from paper - Compute sample weights"""
        with torch.no_grad():
            teacher_outputs = self.cl_teacher(inputs)
            # Temperature scaling for sharpness
            rho = 1.0
            weights = F.softmax(teacher_outputs / rho, dim=1)
            # Get weights for correct labels
            sample_weights = weights[range(len(labels)), labels]
        return sample_weights

    def forward(self, x):
        return self.backbone(x)

    def observe(self, inputs, labels, ulabels):
        """Enhanced observe method following paper's methodology"""
        
        # Main forward pass
        student_outputs = self.forward(inputs)
        student_features = self.get_features(inputs)
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Separate learn/retain and unlearn samples
        learn_retain_mask = (ulabels == 0)
        unlearn_mask = (ulabels == 1)
        
        # === CONTINUAL LEARNING LOSSES ===
        if learn_retain_mask.any():
            lr_inputs = inputs[learn_retain_mask]
            lr_labels = labels[learn_retain_mask]
            lr_outputs = student_outputs[learn_retain_mask]
            lr_features = student_features[learn_retain_mask]
            
            # 1. Classification loss (Lce)
            ce_loss = F.cross_entropy(lr_outputs, lr_labels)
            total_loss += ce_loss
            
            # 2. Online distillation loss (Lod) - Equation 4
            with torch.no_grad():
                teacher_outputs = self.cl_teacher(lr_inputs)
                teacher_features = self.get_features(lr_inputs, self.cl_teacher)
                sample_weights = self.compute_sample_weights(lr_inputs, lr_labels)
            
            # Weighted MSE loss
            weighted_mse = sample_weights * torch.sum((lr_outputs - teacher_outputs) ** 2, dim=1)
            lod_loss = torch.mean(weighted_mse)
            total_loss += self.alpha1 * lod_loss
            
            # 3. Contrastive distillation loss (Lcd) - Equation 5
            if len(lr_features) > 1:  # Need at least 2 samples for contrastive
                lcd_loss = self.contrastive_distillation_loss(lr_features, teacher_features, lr_labels)
                total_loss += self.alpha2 * lcd_loss
            
            # 4. Supervised contrastive loss (Lscd) - Equation 7
            if len(lr_features) > 1:
                lscd_loss = self.supervised_contrastive_loss(lr_features, lr_labels)
                total_loss += self.alpha3 * lscd_loss
        
        # === UNLEARNING LOSSES ===
        if unlearn_mask.any():
            ul_inputs = inputs[unlearn_mask]
            ul_outputs = student_outputs[unlearn_mask]
            
            # Paper's approach: KL divergence with bad teacher (Equation 9)
            with torch.no_grad():
                bad_teacher_outputs = self.ul_teacher(ul_inputs)
                good_teacher_outputs = self.cl_teacher(ul_inputs)
            
            # Dynamic weight for unlearning (paper uses ωu)
            omega_u = 0.8  # Favor bad teacher for unlearning
            
            # KL divergence losses
            kl_bad = F.kl_div(F.log_softmax(ul_outputs, dim=1), 
                             F.softmax(bad_teacher_outputs, dim=1), 
                             reduction='batchmean')
            kl_good = F.kl_div(F.log_softmax(ul_outputs, dim=1), 
                              F.softmax(good_teacher_outputs, dim=1), 
                              reduction='batchmean')
            
            # Combined unlearning loss (Equation 9)
            unlearn_loss = omega_u * kl_bad + (1 - omega_u) * kl_good
            total_loss += self.beta * unlearn_loss
        
        # === EXPERIENCE REPLAY ===
        if self.buffer.num_seen_examples > 0:
            buf_inputs, buf_labels, buf_ulabels = self.buffer.get_data(min(32, inputs.shape[0]))
            if buf_inputs is not None:
                buf_outputs = self.forward(buf_inputs)
                buf_retain_mask = (buf_ulabels == 0)
                
                if buf_retain_mask.any():
                    # Strong retention loss for buffer
                    buf_ce_loss = F.cross_entropy(buf_outputs[buf_retain_mask], buf_labels[buf_retain_mask])
                    total_loss += self.gamma * buf_ce_loss
                    
                    # Additional distillation for buffer samples
                    with torch.no_grad():
                        buf_teacher_outputs = self.cl_teacher(buf_inputs[buf_retain_mask])
                    buf_distill_loss = F.mse_loss(buf_outputs[buf_retain_mask], buf_teacher_outputs)
                    total_loss += 0.5 * buf_distill_loss
        
        # Add to buffer (only add retain samples, not unlearn samples)
        retain_samples_mask = (ulabels == 0)
        if retain_samples_mask.any():
            self.buffer.add_data(inputs[retain_samples_mask].detach(), 
                               labels[retain_samples_mask].detach(), 
                               ulabels[retain_samples_mask].detach())
        
        return total_loss

    def update_teachers(self, momentum=0.99):
        """Stronger momentum update to preserve knowledge better"""
        with torch.no_grad():
            # Update CL teacher with higher momentum to preserve knowledge
            for param_student, param_teacher in zip(self.backbone.parameters(), self.cl_teacher.parameters()):
                param_teacher.data.mul_(momentum).add_(param_student.data, alpha=1-momentum)
            
            # Update teacher projector
            for param_student, param_teacher in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
                param_teacher.data.mul_(momentum).add_(param_student.data, alpha=1-momentum)

# -------------------
# Validation
# -------------------
def validate_model(model, forget_range, retain_range, new_range, overall_range, device):
    model.eval()
    results = {}
    ranges = {
        'Forget': forget_range,
        'Retain': retain_range,
        'New': new_range,
        'Overall': overall_range
    }
    
    with torch.no_grad():
        for name, class_range in ranges.items():
            val_loader = get_dynamic_loader(class_range, mode="val", batch_size=128)
            correct, total = 0, 0
            
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            results[name] = accuracy
    
    model.train()
    return results

# -------------------
# Enhanced Training with Better Learning Rate Scheduling
# -------------------
def sliding_window_step(checkpoint_path, learn_range, unlearn_range, retain_range,
                       save_path, n_epochs=50, lr=0.0005, batch_size=128):
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This code requires GPU.")
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    print(f"\n🔥 Enhanced GPU Training Setup:")
    print(f"   🔥 Device: {device}")
    print(f"   🔥 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   🔥 Batch Size: {batch_size}")
    print(f"   🔥 Learning Rate: {lr}")
    
    print(f"\n🔄 Sliding window step:")
    print(f"   📚 Learn classes: {learn_range}")
    print(f"   🗑️  Unlearn classes: {unlearn_range}")
    print(f"   💾 Retain classes: {retain_range}")

    # Load model
    backbone = get_model(num_classes=100, pretrained=False)
    load_model_weights(backbone, checkpoint_path)
    backbone.to(device)

    # Create enhanced UniCLUN model
    model = UniCLUNModel(backbone, num_classes=100, buffer_size=500)  # Larger buffer
    model.to(device)
    model.buffer.remove_class_range(unlearn_range)

    # Dataset
    train_dataset = GPUOptimizedMixedDataset(learn_range, unlearn_range, retain_range, 
                                           mode="train", data_percentage=1.0, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False, drop_last=True)

    # Enhanced optimizer with lower learning rate and weight decay
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=lr, weight_decay=1e-3, 
                                 betas=(0.9, 0.999), eps=1e-8)
    
    # Learning rate scheduler - Cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.1)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    print(f"🚀 Starting enhanced training with {len(train_loader)} batches per epoch")
    
    # Track best accuracy to save best model
    best_overall_acc = 0.0
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", 
                           leave=False, position=0, ncols=100)
        
        for batch_idx, (inputs, labels, ulabels) in enumerate(progress_bar):
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) 
                ulabels = ulabels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                loss = model.observe(inputs, labels, ulabels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update teachers with higher momentum for stability
            model.update_teachers(momentum=0.995)
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
            
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        progress_bar.close()
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        overall_range = (retain_range[0], learn_range[1])
        val_results = validate_model(model.backbone, unlearn_range, retain_range, learn_range, overall_range, device)
        
        # Save best model
        if val_results['Overall'] > best_overall_acc:
            best_overall_acc = val_results['Overall']
            torch.save(model.backbone.state_dict(), save_path.replace('.pth', '_best.pth'))
        
        print(f"\n📊 Epoch {epoch+1}/{n_epochs} Results:")
        print(f"   📉 Loss: {avg_loss:.4f}")
        print(f"   🔧 LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   🗑️ Forget: {val_results['Forget']:.1f}%")
        print(f"   💾 Retain: {val_results['Retain']:.1f}%") 
        print(f"   📚 New: {val_results['New']:.1f}%")
        print(f"   🎯 Overall: {val_results['Overall']:.1f}% (Best: {best_overall_acc:.1f}%)")
    
    # Save final model
    torch.save(model.backbone.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
    print(f"✅ Best model saved to {save_path.replace('.pth', '_best.pth')}")
    
    torch.cuda.empty_cache()
    return model

# -------------------
# Enhanced Pipeline
# -------------------
def run_sliding_window_pipeline(initial_checkpoint="checkpoints/oracle/0_49.pth"):
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please use a GPU for training.")
        return
    
    print(f"🔥 GPU Available: {torch.cuda.get_device_name()}")
    print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    steps = [
        {"checkpoint": initial_checkpoint, "learn_range": (50, 59), "unlearn_range": (0, 9), "retain_range": (10, 49), "save_path": "10-59.pth", "name": "Step 1: 0-49 → 10-59"},
        {"checkpoint": "10-59.pth", "learn_range": (60, 69), "unlearn_range": (10, 19), "retain_range": (20, 59), "save_path": "20-69.pth", "name": "Step 2: 10-59 → 20-69"},
        {"checkpoint": "20-69.pth", "learn_range": (70, 79), "unlearn_range": (20, 29), "retain_range": (30, 69), "save_path": "30-79.pth", "name": "Step 3: 20-69 → 30-79"},
        {"checkpoint": "30-79.pth", "learn_range": (80, 89), "unlearn_range": (30, 39), "retain_range": (40, 79), "save_path": "40-89.pth", "name": "Step 4: 30-79 → 40-89"},
        {"checkpoint": "40-89.pth", "learn_range": (90, 99), "unlearn_range": (40, 49), "retain_range": (50, 89), "save_path": "50-99.pth", "name": "Step 5: 40-89 → 50-99"}
    ]
    
    pipeline_pbar = tqdm(steps, desc="🚀 Enhanced Pipeline", unit="step", position=1, leave=True)
    
    for step in pipeline_pbar:
        pipeline_pbar.set_description(f"🚀 {step['name']}")
        try:
            sliding_window_step(
                checkpoint_path=step["checkpoint"],
                learn_range=step["learn_range"],
                unlearn_range=step["unlearn_range"],
                retain_range=step["retain_range"],
                save_path=step["save_path"],
                n_epochs=50,
                lr=0.0005,  # Lower learning rate for stability
                batch_size=128
            )
            pipeline_pbar.set_postfix({"Status": "✅"})
            
        except Exception as e:
            pipeline_pbar.set_postfix({"Status": f"❌ {str(e)[:20]}..."})
            print(f"\n❌ Error in {step['name']}: {e}")
            break
    
    pipeline_pbar.close()
    print("\n🎉 Enhanced sliding window pipeline completed!")

if __name__ == "__main__":
    run_sliding_window_pipeline("checkpoints/oracle/0_49.pth")