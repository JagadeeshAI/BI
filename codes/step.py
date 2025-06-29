import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from data import get_dynamic_loader
from codes.utils import get_model

# --------------------- Hyperparameters ---------------------
alpha_inc = 0.5
alpha_forget = 2.0
beta = 1.0
gamma = 0.5
inc_epochs = 10
forget_epochs = 15
warmup_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Loss Functions ---------------------
def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def forgetting_loss(logits, forget_classes):
    probs = F.softmax(logits, dim=1)
    mask = torch.zeros_like(probs)
    mask[:, forget_classes] = 1.0
    forget_probs = (probs * mask).sum(dim=1)
    return -torch.log(1.0 - forget_probs + 1e-6).mean()

def bnd_loss(features, base_features):
    loss = 0.0
    for f, bf in zip(features, base_features):
        loss += (f.mean() - bf.mean()).pow(2).mean() + (f.std() - bf.std()).pow(2).mean()
    return loss

# --------------------- Feature Hook ---------------------
class FeatureHook:
    def __init__(self):
        self.features = []
    def __call__(self, module, input, output):
        self.features.append(output.detach())
    def clear(self):
        self.features.clear()

# --------------------- Accuracy Evaluation ---------------------
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

# --------------------- Training Steps ---------------------
def train_increment(model, loader_new, loader_old, optimizer, criterion):
    model.train()
    total_loss = 0
    old_iter = iter(loader_old)

    for images_new, labels_new in tqdm(loader_new, desc="Incremental Learning", leave=False):
        images_new, labels_new = images_new.to(device), labels_new.to(device)
        try:
            images_old, labels_old = next(old_iter)
        except StopIteration:
            old_iter = iter(loader_old)
            images_old, labels_old = next(old_iter)
        images_old, labels_old = images_old.to(device), labels_old.to(device)

        loss = (alpha_inc * criterion(model(images_old), labels_old) +
                (1 - alpha_inc) * criterion(model(images_new), labels_new))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader_new)

def train_forgetting(model, baseline, retain_loader, forget_loader, optimizer, forget_classes):
    hooks, base_hooks = [], []
    for i in range(12):
        h = FeatureHook()
        bh = FeatureHook()
        model.blocks[i].mlp.register_forward_hook(h)
        baseline.blocks[i].mlp.register_forward_hook(bh)
        hooks.append(h)
        base_hooks.append(bh)

    for epoch in range(forget_epochs):
        model.train()
        loop = tqdm(zip(retain_loader, forget_loader), total=min(len(retain_loader), len(forget_loader)),
                    desc=f"Forget Epoch {epoch+1}")
        for (rx, ry), (fx, _) in loop:
            rx, ry = rx.to(device), ry.to(device)
            fx = fx.to(device)

            for h in hooks + base_hooks:
                h.clear()

            rlogits = model(rx)
            flogits = model(fx)
            _ = baseline(rx)

            loss_r = retention_loss(rlogits, ry)
            loss_f = forgetting_loss(flogits, forget_classes) if epoch >= warmup_epochs else 0.0
            loss_bnd = bnd_loss([h.features[-1] for h in hooks],
                                [h.features[-1] for h in base_hooks]) if epoch >= warmup_epochs else 0.0

            total_loss = beta * loss_r + alpha_forget * loss_f + gamma * loss_bnd

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

# --------------------- MAIN PIPELINE ---------------------
def main():
    os.makedirs("checkpoints/steps", exist_ok=True)
    num_classes = 100
    initial_ckpt = "checkpoints/oracle/0_49.pth"
    prev_ckpt = initial_ckpt

    for step in range(5):
        print(f"\n🚀 STEP {step + 1}/5")
        forget_classes = list(range(10 * step, 10 * step + 10))
        retain_classes = list(range(10 * step + 10, 10 * step + 50))
        add_classes = list(range(50 + step * 10, 50 + step * 10 + 10))

        # Load model
        model = get_model(num_classes=num_classes, use_lora=True, pretrained=False, lora_rank=2).to(device)
        state_dict = torch.load(prev_ckpt, map_location=device,weights_only=True)
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(filtered_state_dict, strict=False)

        # INCREMENT
        print(f"🔼 Adding classes {add_classes[0]}–{add_classes[-1]}")
        loader_new = get_dynamic_loader(class_range=(add_classes[0], add_classes[-1]), mode="train", batch_size=64)
        loader_old = get_dynamic_loader(class_range=(forget_classes[0], add_classes[0] - 1), mode="train", batch_size=64)
        val_loader = get_dynamic_loader(class_range=(retain_classes[0], add_classes[-1]), mode="val", batch_size=64)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        for epoch in range(inc_epochs):
            print(f"📅 Increment Epoch {epoch + 1}/{inc_epochs}")
            train_loss = train_increment(model, loader_new, loader_old, optimizer, criterion)
            acc = evaluate(model, val_loader, device, num_classes)
            print(f"📊 Train Loss: {train_loss:.4f} | Val Acc (Retain+New): {acc:.2f}%")

        # FORGET
        print(f"❌ Forgetting classes {forget_classes[0]}–{forget_classes[-1]}")
        baseline = deepcopy(model).eval()
        retain_loader = get_dynamic_loader(class_range=(retain_classes[0], add_classes[-1]), mode="train", batch_size=64)
        forget_loader = get_dynamic_loader(class_range=(forget_classes[0], forget_classes[-1]), mode="train", batch_size=64)
        train_forgetting(model, baseline, retain_loader, forget_loader, optimizer, forget_classes)

        # Save checkpoint
        step_ckpt = f"checkpoints/steps/step{step + 1}.pth"
        torch.save(model.state_dict(), step_ckpt)
        prev_ckpt = step_ckpt

        # Final Eval
        acc = evaluate(model, val_loader, device, num_classes)
        print(f"✅ Step {step + 1} done | Saved to {step_ckpt} | Val Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
