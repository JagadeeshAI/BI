# uniCLUN.py — Full Pipeline Following UniCLUN Paper for 10–59 Step (No projector version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from tqdm import tqdm
from codes.data import get_dynamic_loader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (ViT with 100-way classifier)
def get_model():
    model = create_model('deit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 100)
    return model.to(device)

# UniCLUN loss (without projector)
def uniclun_loss(student_logits, labels, cl_logits, ul_logits, buffer_mask, forget_mask, gamma):
    ce_loss = F.cross_entropy(student_logits, labels)

    distill_loss = F.kl_div(
        F.log_softmax(student_logits[buffer_mask], dim=1),
        F.softmax(cl_logits[buffer_mask], dim=1),
        reduction='batchmean'
    ) if buffer_mask.any() else torch.tensor(0.0, device=device)

    forget_kl = 0.0
    if forget_mask.any():
        kl_ul = F.kl_div(
            F.log_softmax(student_logits[forget_mask], dim=1),
            F.softmax(ul_logits[forget_mask], dim=1),
            reduction='batchmean'
        )
        kl_cl = F.kl_div(
            F.log_softmax(student_logits[forget_mask], dim=1),
            F.softmax(cl_logits[forget_mask], dim=1),
            reduction='batchmean'
        )
        forget_kl = 0.5 * kl_ul + 0.5 * kl_cl

    return gamma * (ce_loss + distill_loss) + (1 - gamma) * forget_kl

# Accuracy metric
def evaluate(model, loader):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_correct / total_samples

# Load models
print("🔁 Loading CL Teacher (10–49)")
cl_teacher = get_model()
cl_teacher.load_state_dict(torch.load("baseline/teachers/checkpoint_deit_10_49.pth"))
cl_teacher.eval()

print("🧹 Loading UL Teacher (10–59)")
ul_teacher = get_model()
ul_teacher.load_state_dict(torch.load("baseline/teachers/checkpoint_deit_10_59.pth"))
ul_teacher.eval()

print("🎓 Loading Oracle Student (0–49)")
student = get_model()
student.load_state_dict(torch.load("checkpoints/oracle/0_49.pth"))
student.train()

# DataLoaders
train_loader = get_dynamic_loader(class_range=(50, 60), mode="train", batch_size=64)
buffer_loader = get_dynamic_loader(class_range=(10, 50), mode="train", batch_size=64)
val_loader = get_dynamic_loader(class_range=(10, 60), mode="val", batch_size=64)

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
EPOCHS = 10
GAMMA = 0.5

print("🚀 Starting UniCLUN Training")
for epoch in range(EPOCHS):
    student.train()
    total_loss = 0.0

    loop = tqdm(zip(train_loader, buffer_loader), total=min(len(train_loader), len(buffer_loader)), desc=f"Epoch {epoch+1}")
    for (new_x, new_y), (buf_x, buf_y) in loop:
        x = torch.cat([new_x, buf_x], dim=0).to(device)
        y = torch.cat([new_y, buf_y], dim=0).to(device)

        student_logits = student(x)
        with torch.no_grad():
            cl_logits = cl_teacher(x)
            ul_logits = ul_teacher(x)

        buffer_mask = (y >= 10) & (y < 50)
        forget_mask = (y < 10)

        loss = uniclun_loss(student_logits, y, cl_logits, ul_logits, buffer_mask, forget_mask, GAMMA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    acc = evaluate(student, val_loader)
    print(f"✅ Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Acc: {acc * 100:.2f}%")

# Save final model
torch.save(student.state_dict(), "checkpoints/10_59.pth")
print("✅ Saved student model at checkpoints/10_59.pth")