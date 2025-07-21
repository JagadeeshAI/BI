import os
import torch
import json
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from codes.utils import get_model
from codes.data import get_dynamic_loader

def evaluate_teacher(model_path, class_ranges, device):
    """Evaluate a teacher model on specified class ranges"""
    model = get_model(num_classes=100, use_lora=False, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for class_range in class_ranges:
        val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=32)
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Evaluating {class_range}", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Teacher configurations
    teacher_configs = [
        {
            "student": "10_59",
            "good_ranges": [(10, 59)],
            "bad_ranges": [(0, 9), (60, 99)]
        },
        {
            "student": "20_69", 
            "good_ranges": [(20, 69)],
            "bad_ranges": [(0, 19), (70, 99)]
        },
        {
            "student": "30_79",
            "good_ranges": [(30, 79)],
            "bad_ranges": [(0, 29), (80, 99)]
        },
        {
            "student": "40_89",
            "good_ranges": [(40, 89)],
            "bad_ranges": [(0, 39), (90, 99)]
        },
        {
            "student": "50_99",
            "good_ranges": [(50, 99)],
            "bad_ranges": [(0, 49)]
        }
    ]
    
    for config in teacher_configs:
        student = config["student"]
        print(f"\n🔍 Evaluating teachers for student {student}")
        
        # Good teacher
        good_path = f"./baseline/teachers/student_{student}_good_teacher.pth"
        good_acc = evaluate_teacher(good_path, config["good_ranges"], device)
        
        # Bad teacher  
        bad_path = f"./baseline/teachers/student_{student}_bad_teacher.pth"
        bad_acc = evaluate_teacher(bad_path, config["bad_ranges"], device)
        
        results[f"student_{student}"] = {
            "good_teacher_acc": round(good_acc * 100, 2),
            "bad_teacher_acc": round(bad_acc * 100, 2),
            "good_ranges": config["good_ranges"],
            "bad_ranges": config["bad_ranges"]
        }
        
        print(f"✅ Good Teacher: {good_acc*100:.2f}% | Bad Teacher: {bad_acc*100:.2f}%")
    
    # Save results
    os.makedirs("/home/jag/codes/Bi/baseline/results", exist_ok=True)
    with open("/home/jag/codes/Bi/baseline/results/teacher_accuracies.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to /home/jag/codes/Bi/baseline/results/teacher_accuracies.json")

if __name__ == "__main__":
    main()