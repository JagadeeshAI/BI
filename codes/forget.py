import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from codes.utils import get_model
from data import get_dynamic_loader

# --------------------- Hyperparameters ---------------------
alpha = 2.0
beta = 1.0
gamma = 0.5
warmup_epochs = 5
total_epochs = 15
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
def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

# --------------------- Main Training ---------------------
def main():
    forget_classes = list(range(0, 10))
    retain_classes = list(range(10, 50))

    model = get_model(num_classes=100, use_lora=True, lora_rank=2, pretrained=False).to(device)
    state_dict = torch.load("0_59.pth", map_location=device)

    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    missing_keys = [k for k in model.state_dict() if k not in filtered_state_dict]
    print(f"⚠️ Skipping keys: {missing_keys[:5]}... (+{len(missing_keys)-5} more)" if len(missing_keys) > 5 else f"⚠️ Skipping keys: {missing_keys}")
    model.load_state_dict(filtered_state_dict, strict=False)

    baseline = deepcopy(model).eval()

    retain_loader = get_dynamic_loader(class_range=(10, 59), mode="train", batch_size=64)
    forget_loader = get_dynamic_loader(class_range=(0, 9), mode="train", batch_size=64)
    val_retain_loader = get_dynamic_loader(class_range=(10, 59), mode="val", batch_size=64)
    val_forget_loader = get_dynamic_loader(class_range=(0, 9), mode="val", batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Attach BND hooks
    hooks, base_hooks = [], []
    for i in range(12):
        h = FeatureHook()
        bh = FeatureHook()
        model.blocks[i].mlp.register_forward_hook(h)
        baseline.blocks[i].mlp.register_forward_hook(bh)
        hooks.append(h)
        base_hooks.append(bh)

    for epoch in range(total_epochs):
        model.train()
        r_loss_sum, f_loss_sum, bnd_sum = 0.0, 0.0, 0.0

        loop = tqdm(zip(retain_loader, forget_loader), total=min(len(retain_loader), len(forget_loader)), desc=f"📉 Epoch {epoch+1}")
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

            total_loss = beta * loss_r + alpha * loss_f + gamma * loss_bnd

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            r_loss_sum += loss_r.item()
            f_loss_sum += loss_f if isinstance(loss_f, float) else loss_f.item()
            bnd_sum += loss_bnd if isinstance(loss_bnd, float) else loss_bnd.item()

            loop.set_postfix(Retention=f"{r_loss_sum:.2f}", Forget=f"{f_loss_sum:.2f}", BND=f"{bnd_sum:.2f}")

        retain_acc = compute_accuracy(model, val_retain_loader)
        forget_acc = compute_accuracy(model, val_forget_loader)
        print(f"✅ Retain Acc: {retain_acc:.2f}% | ❌ Forget Acc: {forget_acc:.2f}%")

    torch.save(model.state_dict(), "forget_0_9.pth")
    print("🚀 Done. Model saved to 'forget_0_9.pth'")

if __name__ == "__main__":
    main()
