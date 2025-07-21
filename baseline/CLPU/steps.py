# CLPU_STEPS.py (WITH TQDM PROGRESS BARS AND DETAILED SPECS)

"""
Enhanced CLPU Continual Learning Pipeline with TQDM Status Bars

Adds live per-action and per-step feedback using tqdm progress bars,
showing exactly which class or operation (learn, promote, forget)
is in progress during each training or update phase.

Fully compatible with get_dynamic_loader from data.py.
"""

import os
import copy
import logging
import torch
import yaml
from tqdm import tqdm

from codes.utils import get_model
from baseline.pkgs.CLPU.model.clpu_derpp import CLPU_Derpp
from codes.data import get_dynamic_loader

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state": model.net.state_dict(),
        "memory": model.memory,
        "prev_tasks": model.prev_tasks,
        "side_nets": {k: v.state_dict() for k, v in model.side_nets.items()},
        "task_status": model.task_status,
    }
    torch.save(checkpoint, path)
    print(f"💾 Saved model checkpoint to: {path}")

def load_config(yaml_path):
    assert os.path.isfile(yaml_path), f"Config file not found: {yaml_path}"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"📄 Loaded config: {yaml_path}")
    return DotDict(config)

class DotDict(dict):
    """dot-access dict"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

STEP_WINDOWS = [
    (10, 59),
    (20, 69),
    (30, 79),
    (40, 89),
    (50, 99),
]

SAVE_DIR = "./clpu_steps"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clpu_log(msg):
    logging.info('=' * 60)
    logging.info(msg)
    logging.info('=' * 60)

def get_class_loader(cls, config, mode="train"):
    return get_dynamic_loader(
        class_range=(cls, cls),
        mode=mode,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers
    )

def pretrain_temp_0_49(model, config):
    clpu_log("Pretraining classes 0–49 as Temporary (T)")
    for cls in tqdm(range(0, 50), desc="TEMP Pretraining: [0–49]"):
        loader = get_class_loader(cls, config, mode="train")
        tqdm.write(f"Training TEMPORARY class {cls} (0–49)")
        model.temporarily_learn(cls, loader)
        model.task_status[cls] = 'T'
    save_model(model, os.path.join(SAVE_DIR, "pretrain_temp_0_49.pth"))
    clpu_log(f"Saved pretrained model to {os.path.join(SAVE_DIR, 'pretrain_temp_0_49.pth')}")

def clpu_run_step(model, config, step_id, win_start, win_end):
    step_name = f"step{step_id + 1}"
    clpu_log(f"==> {step_name}: Active range [{win_start}–{win_end}]")
    # 1. PROMOTE (temporary → permanent) for current window
    for cls in tqdm(range(win_start, win_end + 1), desc=f"{step_name}: Promote [R] [{win_start}–{win_end}]"):
        if model.task_status.get(cls, 'T') == 'T':
            loader = get_class_loader(cls, config, mode="train")
            tqdm.write(f"Promoting class {cls} to PERMANENT (R)")
            model.finally_learn(cls, loader)
            model.task_status[cls] = 'R'
    # 2. ADD next 10 classes as TEMPORARY
    add_start = win_end + 1
    add_end = min(win_end + 10, 99)
    for cls in tqdm(range(add_start, add_end + 1), desc=f"{step_name}: Add [T] [{add_start}–{add_end}]"):
        if cls < 100:
            loader = get_class_loader(cls, config, mode="train")
            tqdm.write(f"Training TEMPORARY class {cls} (add phase)")
            model.temporarily_learn(cls, loader)
            model.task_status[cls] = 'T'
    # 3. FORGET (remove) previous 10 temp classes
    forget_start = win_start - 10
    forget_end = win_start - 1
    for cls in tqdm(range(forget_start, forget_end + 1), desc=f"{step_name}: Forget [F] [{forget_start}–{forget_end}]"):
        if 0 <= cls < 100:
            if model.task_status.get(cls) == 'T':
                tqdm.write(f"Forgetting TEMPORARY class {cls}")
                model.forget(cls)
                del model.task_status[cls]
            elif model.task_status.get(cls) == 'R':
                tqdm.write(f"Class {cls} is already permanent — cannot unlearn.")
    # 4. Save
    save_path = os.path.join(SAVE_DIR, f"{step_name}.pth")
    save_model(model, save_path)
    clpu_log(f"[✔] Saved {step_name} checkpoint: {save_path}")

def main():
    config = load_config("configs/cifar100_clpu.yaml")
    vit_model = get_model(num_classes=100, pretrained=config.use_pretrain)
    vit_model.to(DEVICE)
    model = CLPU_Derpp(config)
    model.net = vit_model
    model.device = DEVICE
    if not hasattr(model, 'task_status'):
        model.task_status = {}
    pretrain_temp_0_49(model, config)
    for i, (start, end) in enumerate(STEP_WINDOWS):
        clpu_run_step(model, config, i, start, end)
    clpu_log("✅ All CLPU steps completed. Final checkpoints saved!")

if __name__ == "__main__":
    main()
