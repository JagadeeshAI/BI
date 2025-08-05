import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from codes.utils import get_model
from codes.data import get_dynamic_loader

def evaluate_student_step(model_path, target_range, device='cuda'):
    """Evaluate student model on target range classes only"""
    
    # Load model
    student = get_model(num_classes=100, use_lora=False, pretrained=False).to(device)
    student.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    student.eval()
    
    # Get test loader for target range only
    test_loader = get_dynamic_loader(class_range=target_range, mode="val", batch_size=32)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    # Initialize counters for each class in target range
    for cls in range(target_range[0], target_range[1] + 1):
        class_correct[cls] = 0
        class_total[cls] = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {target_range}"):
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)
            
            # Overall accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label in class_correct:
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1
    
    # Calculate overall accuracy
    overall_acc = 100.0 * correct / total if total > 0 else 0.0
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for cls in range(target_range[0], target_range[1] + 1):
        if class_total[cls] > 0:
            class_accuracies[cls] = 100.0 * class_correct[cls] / class_total[cls]
        else:
            class_accuracies[cls] = 0.0
    
    return overall_acc, class_accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step configurations - only evaluate on target ranges
    steps_config = {
        1: (10, 59),
        2: (20, 69), 
        3: (30, 79),
        4: (40, 89),
        5: (50, 99)
    }
    
    results = {}
    
    print("UniCLUN Student Model Evaluation")
    print("=" * 50)
    
    for step_num, target_range in steps_config.items():
        model_path = f"baseline/tables/UniCLUN/students/step{step_num}.pth"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"\nStep {step_num}: Evaluating on classes {target_range[0]}-{target_range[1]}")
        
        try:
            overall_acc, class_accuracies = evaluate_student_step(model_path, target_range, device)
            results[step_num] = {
                'target_range': target_range,
                'overall_acc': overall_acc,
                'class_accuracies': class_accuracies
            }
            
            print(f"Overall Accuracy: {overall_acc:.2f}%")
            
            # Show per-class breakdown (first 10 and last 10 classes)
            classes = list(range(target_range[0], target_range[1] + 1))
            print(f"Per-class accuracies:")
            print(f"First 10 classes: {[f'{cls}:{class_accuracies[cls]:.1f}%' for cls in classes[:10]]}")
            if len(classes) > 10:
                print(f"Last 10 classes:  {[f'{cls}:{class_accuracies[cls]:.1f}%' for cls in classes[-10:]]}")
            
            # Statistics
            accs_list = list(class_accuracies.values())
            print(f"Mean: {np.mean(accs_list):.2f}% | Std: {np.std(accs_list):.2f}% | Min: {np.min(accs_list):.2f}% | Max: {np.max(accs_list):.2f}%")
            
        except Exception as e:
            print(f"Error evaluating step {step_num}: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Step':<6} {'Target Range':<15} {'Overall Acc':<12} {'Mean±Std':<15}")
    print("-" * 70)
    
    for step_num in sorted(results.keys()):
        data = results[step_num]
        target_range = data['target_range']
        overall_acc = data['overall_acc']
        accs_list = list(data['class_accuracies'].values())
        mean_acc = np.mean(accs_list)
        std_acc = np.std(accs_list)
        
        print(f"{step_num:<6} {target_range[0]}-{target_range[1]:<12} {overall_acc:<11.2f}% {mean_acc:.2f}±{std_acc:.2f}%")
    
    # LaTeX table format
    print("\n" + "=" * 50)
    print("LATEX TABLE FORMAT")
    print("=" * 50)
    
    for step_num in sorted(results.keys()):
        data = results[step_num]
        overall_acc = data['overall_acc']
        print(f"Step-{step_num} & {overall_acc:.2f}\\% \\\\")

if __name__ == "__main__":
    main()